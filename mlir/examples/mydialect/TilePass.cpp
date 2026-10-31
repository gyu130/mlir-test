// TilePass.cpp：--mydialect-tile
// 对 lowering 后生成的 elementwise scf.for 做 strip-mining 分块：
//
//   scf.for %i = lb to ub step 1 {  body(%i)  }
// =>
//   scf.for %io = lb to ub step T {
//     %rem = affine.min (d0)[s0] -> (T, s0 - d0)(%io)[%ub]
//     scf.for %ii = 0 to %rem step 1 {
//       %i = %io + %ii
//       body(%i)
//     }
//   }
//
// 合法性（只挑真正可安全分块的循环）：
//   1. 无 iter_args（结果数为 0）——elementwise 循环没有跨迭代携带依赖；
//   2. 只处理最外层循环（父 op 不是 scf.for），避免对分块产生的内层再分块；
//   3. step 为常量 1；
//   4. 常量上界 <= tile-size 时不值得分块，直接跳过。
#include "MyOptPass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace mlir;

namespace mydialect {
namespace {

struct TileLoopPattern : public OpRewritePattern<scf::ForOp> {
  TileLoopPattern(MLIRContext *ctx, int64_t tileSize)
      : OpRewritePattern<scf::ForOp>(ctx), tileSize(tileSize) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // 1. 循环携带值存在 → 有跨迭代依赖，不做分块
    if (op.getNumResults() != 0)
      return failure();
    // 2. 只分块最外层循环（内层交给后续多维 tiling 或保持原样）
    if (isa<scf::ForOp>(op->getParentOp()))
      return failure();
    // 3. step 必须为常量 1
    auto stepCst = getConstantIntValue(op.getStep());
    if (!stepCst || *stepCst != 1)
      return failure();
    // 4. 静态上界不超过 tile 大小 → 分块无收益
    if (auto ubCst = getConstantIntValue(op.getUpperBound());
        ubCst && *ubCst <= tileSize)
      return failure();

    auto loc = op.getLoc();
    Value lb = op.getLowerBound();
    Value ub = op.getUpperBound();
    Value cTile = arith::ConstantIndexOp::create(rewriter, loc, tileSize);
    Value cZero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value cOne = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // 外层循环：步长 = tile-size
    auto outer = scf::ForOp::create(rewriter, loc, lb, ub, cTile);
    rewriter.setInsertionPointToStart(outer.getBody());

    // 内层上界：min(tile-size, ub - io)，用 affine.min 表达分块边界
    AffineMap ubMap = AffineMap::get(
        /*dimCount=*/1, /*symbolCount=*/1,
        {getAffineConstantExpr(tileSize, getContext()),
         getAffineSymbolExpr(0, getContext()) - getAffineDimExpr(0, getContext())},
        getContext());
    Value innerUb = affine::AffineMinOp::create(
        rewriter, loc, ubMap, ValueRange{outer.getInductionVar(), ub});

    // 内层循环：从 0 到 min(tile, 剩余)，步长 1
    auto inner = scf::ForOp::create(rewriter, loc, cZero, innerUb, cOne);

    // 用 io + ii 替换原循环 IV，把原循环体整体搬进内层循环
    Block *innerBody = inner.getBody();
    rewriter.setInsertionPointToStart(innerBody);
    Value idx = arith::AddIOp::create(rewriter, loc, outer.getInductionVar(),
                                      inner.getInductionVar());
    rewriter.eraseOp(innerBody->getTerminator()); // 去掉内层自动插入的 scf.yield

    Block *srcBody = op.getBody();
    rewriter.inlineBlockBefore(srcBody, innerBody, innerBody->end(), idx);

    rewriter.eraseOp(op);
    return success();
  }

  int64_t tileSize;
};

struct TilePass : public PassWrapper<TilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilePass)
  TilePass() = default;
  TilePass(const TilePass &other) : PassWrapper(other) {}

  Option<int64_t> tileSize{
      *this, "tile-size", llvm::cl::desc("分块大小（外层循环步长）"),
      llvm::cl::init(32)};

  StringRef getArgument() const final { return "mydialect-tile"; }
  StringRef getDescription() const final {
    return "Strip-mine outermost elementwise scf.for loops with tile-size";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect,
                    affine::AffineDialect>();
  }

  void runOnOperation() override {
    if (tileSize <= 0) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<TileLoopPattern>(&getContext(), tileSize.getValue());
    GreedyRewriteConfig config;
    config.setMaxIterations(8);
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
  }
};

static PassRegistration<TilePass> regTile;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createTilePass() {
  return std::make_unique<TilePass>();
}
} // namespace mydialect
