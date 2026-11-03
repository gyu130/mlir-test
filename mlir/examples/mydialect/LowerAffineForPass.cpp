// LowerAffineForPass.cpp：--mydialect-for-to-affine
// 将 mydialect.for 降级为 affine.for，从而复用 affine 方言的循环分析
// （依赖分析、trip count 推导）与变换能力（tiling / fusion / vectorize）。
//
// 转换条件（不满足则保留 mydialect.for，由 --lower-mydialect-print 走 scf.for）：
//   1. step 是正的编译期常量 —— affine.for 的 step 只能是整数常量；
//   2. lb/ub 可以写成 affine 作用域内的合法 dim —— i32 边界在作用域顶层做
//      index_cast 后即可作为 dim；若边界定义在嵌套 region 内（如另一个
//      mydialect.for 的 body 中算出的值），则放弃转换。
//
// 转换后 affine.for 的 IV 是 index 类型，body 内再 index_cast 回 i32 供
// mydialect 算子使用；iter_args 的类型与数量保持不变，mydialect.yield
// 替换为 affine.yield。
#include "MyOptPass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mydialect {
namespace {

// 把 i32 边界值转成 index。若它本身就是 index→i32 的 index_cast 结果，
// 直接复用源 index 值（嵌套循环的内层边界常是外层 IV 的 cast）。
static Value toIndex(PatternRewriter &rewriter, Location loc, Value v) {
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
    Value src = cast.getIn();
    if (src.getType().isIndex() && cast.getOut().getType() == v.getType())
      return src;
  }
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), v);
}

// 判断边界值转换成 index 后能否作为 affine dim 使用：
//   - 已可折叠成 index 值时，直接用 affine 的合法 dim 判定；
//   - 否则该值必须在 affine 作用域顶层（这样新建的 index_cast 位于顶层，
//     是合法 symbol/dim）。
static bool boundCanBeAffineDim(Value v, Region *scope) {
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
    Value src = cast.getIn();
    if (src.getType().isIndex())
      return affine::isValidDim(src, scope);
  }
  return affine::isTopLevelValue(v, scope);
}

struct ForToAffinePattern : OpRewritePattern<mydialect::ForOp> {
  using OpRewritePattern<mydialect::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mydialect::ForOp op,
                                PatternRewriter &rewriter) const override {
    // affine.for 的 step 必须是正的编译期常量
    auto stepCst = getConstantIntValue(op.getStep());
    if (!stepCst || *stepCst <= 0)
      return failure();

    Region *scope = affine::getAffineScope(op);
    if (!scope)
      return failure();
    if (!boundCanBeAffineDim(op.getLower(), scope) ||
        !boundCanBeAffineDim(op.getUpper(), scope)) {
      // 边界不是 affine 作用域内可分析的维度 → 保持 mydialect.for
      return failure();
    }

    Location loc = op.getLoc();
    Value lower = toIndex(rewriter, loc, op.getLower());
    Value upper = toIndex(rewriter, loc, op.getUpper());
    AffineMap idMap = AffineMap::get(/*numDims=*/1, /*numSymbols=*/0,
                                    rewriter.getAffineDimExpr(0),
                                    rewriter.getContext());

    auto affineFor = affine::AffineForOp::create(
        rewriter, loc, ValueRange{lower}, idMap, ValueRange{upper}, idMap,
        *stepCst, op.getIterArgs());

    Block *dstBody = affineFor.getBody();
    // 无 iter_args 时 builder 会自动插入 affine.yield，删掉后由原 yield 顶替；
    // 有 iter_args 时 builder 不插终止符。
    if (!dstBody->empty())
      rewriter.eraseOp(dstBody->getTerminator());

    rewriter.setInsertionPointToStart(dstBody);
    // affine IV 是 index，转回 i32 供 mydialect body 使用
    Value ivI32 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(),
                                             affineFor.getInductionVar());

    // 块参数 [0] = iv(index)，[1...] = iter_args（类型沿用原循环）
    SmallVector<Value, 8> bodyReplacements;
    bodyReplacements.push_back(ivI32);
    for (unsigned i = 1; i < dstBody->getNumArguments(); ++i)
      bodyReplacements.push_back(dstBody->getArgument(i));

    Block *srcBody = &op.getBody().front();
    rewriter.inlineBlockBefore(srcBody, dstBody, dstBody->end(),
                               bodyReplacements);

    // mydialect.yield → affine.yield
    auto yieldOp = cast<mydialect::YieldOp>(dstBody->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    affine::AffineYieldOp::create(rewriter, loc, yieldOp.getValues());
    rewriter.eraseOp(yieldOp);

    rewriter.replaceOp(op, affineFor.getResults());
    return success();
  }
};

struct ForToAffinePass
    : public PassWrapper<ForToAffinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ForToAffinePass)

  ForToAffinePass() = default;
  ForToAffinePass(const ForToAffinePass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "mydialect-for-to-affine"; }
  StringRef getDescription() const final {
    return "Lower mydialect.for to affine.for when the step is a positive "
           "constant";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ForToAffinePattern>(&getContext());
    GreedyRewriteConfig config;
    config.setMaxIterations(16);
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
  }
};

static PassRegistration<ForToAffinePass> regForToAffine;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createForToAffinePass() {
  return std::make_unique<ForToAffinePass>();
}
} // namespace mydialect
