// ParallelPass.cpp：--mydialect-parallelize
// 把可安全并行的 elementwise scf.for 提升为 scf.forall（并行循环）：
//
//   scf.for %i = lb to ub step s { body(%i) }
// =>
//   scf.forall (%i) in (lb to ub step s) { body(%i) }
//
// 合法性（只挑真正可安全并行的循环）：
//   1. 无 iter_args（结果数为 0）——没有跨迭代携带依赖；
//   2. 只处理最外层循环（父 op 不是 scf.for / scf.forall），
//      tile 后的外层分块循环（step = tile-size）因此可以被并行化；
//   3. 循环体内不允许 CallOpInterface 等外部调用（打印/函数调用有副作用，
//      必须保持串行执行顺序）。
//
// 后续降级：
//   - 串行执行：--convert-scf-to-cf 会将 scf.forall → scf.parallel → scf.for；
//   - 多线程后端：--convert-scf-to-openmp 生成 omp.parallel / omp.wsloop。
#include "MyOptPass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace mlir;

namespace mydialect {
namespace {

struct ForToForallPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // 1. 循环携带值存在 → 有跨迭代依赖，不能并行
    if (op.getNumResults() != 0)
      return failure();
    // 2. 只并行化最外层循环（内层留在并行块内部串行执行）
    if (isa<scf::ForOp, scf::ForallOp>(op->getParentOp()))
      return failure();
    // 3. 循环体内的外部调用（打印等）有副作用，必须保持串行
    bool hasCall = false;
    op.getBody()->walk([&](Operation *inner) {
      if (isa<CallOpInterface>(inner))
        hasCall = true;
    });
    if (hasCall)
      return failure();

    auto loc = op.getLoc();
    SmallVector<OpFoldResult> lbs{op.getLowerBound()};
    SmallVector<OpFoldResult> ubs{op.getUpperBound()};
    SmallVector<OpFoldResult> steps{op.getStep()};

    // 无 outputs 的 forall；builder 会自动插入 scf.forall.in_parallel 终止符
    auto forall = scf::ForallOp::create(rewriter, loc, lbs, ubs, steps,
                                        ValueRange{}, std::nullopt);

    // 把原循环体（去掉 scf.yield）搬到 in_parallel 之前，
    // 原循环 IV 由 forall 的 induction var 替换
    rewriter.eraseOp(op.getBody()->getTerminator());
    rewriter.inlineBlockBefore(op.getBody(), forall.getBody(),
                               Block::iterator(forall.getTerminator()),
                               forall.getInductionVars());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ParallelPass : public PassWrapper<ParallelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelPass)
  ParallelPass() = default;
  ParallelPass(const ParallelPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "mydialect-parallelize"; }
  StringRef getDescription() const final {
    return "Promote legal elementwise scf.for loops to scf.forall";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ForToForallPattern>(&getContext());
    GreedyRewriteConfig config;
    config.setMaxIterations(8);
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
  }
};

static PassRegistration<ParallelPass> regParallel;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createParallelPass() {
  return std::make_unique<ParallelPass>();
}
} // namespace mydialect
