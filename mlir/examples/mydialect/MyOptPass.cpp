#include "MyOptPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // 新增：贪婪重写驱动
#include "MyDialect.h"
#include "llvm/ADT/SmallVector.h"
using namespace mlir;
namespace mydialect {

// 保留原有 TableGen 生成逻辑（Base模板、注册宏）
#define GEN_PASS_DEF_MYOPTPASS
#include "MyOptPass.h.inc"
#define GEN_PASS_REGISTRATION_MYOPTPASS
#include "MyOptPass.h.inc"

namespace {

// ==============================
// Pattern 1：删除连续冗余 mydialect.print
// 规则：如果当前 print 的前一个 op 也是 print，且操作数完全相同，删除当前 print
// ==============================
struct RemoveRedundantPrintPattern : public OpRewritePattern<PrintOp> {
  using OpRewritePattern<PrintOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PrintOp printOp,
                                PatternRewriter &rewriter) const override {
    Block *block = printOp->getBlock();
    auto opIt = printOp->getIterator();
    // 块开头没有前一个op，直接不匹配
    if (opIt == block->begin())
      return failure();

    // 往前看一个op
    --opIt;
    Operation *prevOp = &*opIt;
    auto prevPrint = dyn_cast<PrintOp>(prevOp);
    if (!prevPrint)
      return failure();

    // 前一个print和当前print打印同一个值，删除当前冗余打印
    if (prevPrint.getValue() == printOp.getValue()) {
      rewriter.eraseOp(printOp);
      return success();
    }
    return failure();
  }
};

// ==============================
// Pattern 2：块内 arith.addi 公共子表达式消除（CSE）
// 规则：如果当前 addi 之前已经有完全相同(lhs,rhs)的addi，复用之前的结果，删除当前addi
// ==============================
struct AddiCSEPattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();
    Block *block = addOp->getBlock();
    auto opIt = addOp->getIterator();

    // 向前遍历当前块中所有在 addOp 之前的op
    for (auto it = opIt; it != block->begin();) {
      --it;
      Operation *prevOp = &*it;
      auto prevAdd = dyn_cast<arith::AddIOp>(prevOp);
      if (!prevAdd)
        continue;
      // 找到相同操作数的addi，替换当前结果为之前addi的结果
      if (prevAdd.getLhs() == lhs && prevAdd.getRhs() == rhs) {
        rewriter.replaceOp(addOp, prevAdd.getResult());
        return success();
      }
    }
    return failure();
  }
};

// ==============================
// Pattern 3：identity_tensor(x) → x
// identity 语义即透传；verifier 已保证输入/结果类型相同，直接替换
// ==============================
struct IdentityTensorFoldPattern
    : public OpRewritePattern<IdentityTensorOp> {
  using OpRewritePattern<IdentityTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IdentityTensorOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getIn());
    return success();
  }
};

// ==============================
// Pattern 4：transpose(transpose(x, p1), p2) → x
// 条件：两次置换复合为恒等（out2 维度 m ← out1 维度 p2[m] ← x 维度 p1[p2[m]]，
// 要求 p1[p2[m]] == m）。结果类型与原始输入一致时方可折叠。
// ==============================
struct DoubleTransposeFoldPattern
    : public OpRewritePattern<TransposeTensorOp> {
  using OpRewritePattern<TransposeTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getData().getDefiningOp<TransposeTensorOp>();
    if (!inner)
      return failure();
    // 恒等置换必然使结果类型等于原始输入类型，此处兼作安全校验
    if (op.getResult().getType() != inner.getData().getType())
      return failure();

    auto p2 = op.getPerm();
    auto p1 = inner.getPerm();
    if (p1.size() != p2.size())
      return failure();
    for (size_t m = 0; m < p2.size(); ++m) {
      int64_t outerIdx = cast<IntegerAttr>(p2[m]).getInt();
      int64_t xIdx = cast<IntegerAttr>(p1[outerIdx]).getInt();
      if (xIdx != static_cast<int64_t>(m))
        return failure();
    }

    Value src = inner.getData();
    rewriter.replaceOp(op, src);
    // 内层 transpose 可能只剩本次使用，变死代码时顺手删除
    if (inner->use_empty())
      rewriter.eraseOp(inner);
    return success();
  }
};

// ==============================
// Pattern 5：reshape(reshape(x, s1), s2) → reshape(x, s2)
// reshape 只按元素总数重解释布局，内层 reshape 的中间 shape 对外层无影响
// ==============================
struct ChainedReshapeFoldPattern
    : public OpRewritePattern<ReshapeTensorOp> {
  using OpRewritePattern<ReshapeTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getData().getDefiningOp<ReshapeTensorOp>();
    if (!inner)
      return failure();

    auto newOp = rewriter.create<ReshapeTensorOp>(
        op.getLoc(), op.getResult().getType(), inner.getData(), op.getShape());
    rewriter.replaceOp(op, newOp.getResult());
    if (inner->use_empty())
      rewriter.eraseOp(inner);
    return success();
  }
};

// ==============================
// 主Pass：继承TableGen生成的Base，内部用Pattern驱动优化
// ==============================
struct MyOptPass : public impl::MyOptPassBase<MyOptPass> {
  void runOnOperation() override;
};

// 保留原有自动注册逻辑
static mlir::PassRegistration<MyOptPass> reg;

} // anonymous namespace

void MyOptPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();

  // 1. 构建Pattern集合：所有优化规则在这里add，新增优化只需要新增Pattern类即可
  RewritePatternSet patterns(ctx);
  patterns.add<RemoveRedundantPrintPattern, AddiCSEPattern,
               IdentityTensorFoldPattern, DoubleTransposeFoldPattern,
               ChainedReshapeFoldPattern>(ctx);

  // 2. 贪婪重写：自动迭代直到没有Pattern可以匹配，自动处理use-def安全替换/删除
  GreedyRewriteConfig config;
  config.setMaxIterations(10);  // 防止死循环
  config.setUseTopDownTraversal(true);

  if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
    signalPassFailure();
  }
}

} // namespace mydialect

