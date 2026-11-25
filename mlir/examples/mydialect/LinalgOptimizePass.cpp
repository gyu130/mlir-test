// LinalgOptimizePass.cpp：--mydialect-linalg-opt
// 对 LowerMyDialectPrint 生成的 linalg 操作做标准 Linalg 变换：
//   1. 逐元素 linalg.generic（全部 parallel 迭代）→ linalg::tileLinalgOp 分块；
//   2. 分块只作用于最外层循环（其余 tile size 置 0，Linalg 约定 0 = 不分块），
//      与旧 --mydialect-tile 的 strip-mining 语义一致；
//   3. 分块后产生的内层 linalg.generic 由流水线中的
//      --convert-linalg-to-loops 统一降回 scf.for，后续 schedule /
//      SCFToCF / LLVM 链路保持不变。
//
// 说明：MLIR 的 linalg::fuseElementwiseOps 只支持 tensor 语义（producer 必须
// hasPureTensorSemantics）。本流水线全程 buffer 语义（MyTensorType 已被
// TypeConverter 转成 memref），因此逐元素融合仍由 --mydialect-fuse-elementwise
// 在 mydialect 层完成，其结果降级后就是「单个 linalg.generic + 无中间 memref」，
// 与 linalg 融合的收益相同。
#include "MyOptPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace mlir;

namespace mydialect {
namespace {

struct LinalgOptimizePass
    : public PassWrapper<LinalgOptimizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgOptimizePass)
  LinalgOptimizePass() = default;
  LinalgOptimizePass(const LinalgOptimizePass &other) : PassWrapper(other) {}

  Option<int64_t> tileSize{
      *this, "tile-size", llvm::cl::desc("分块大小（最外层循环的 tile size）"),
      llvm::cl::init(32)};
  Option<bool> tileEnabled{
      *this, "tile", llvm::cl::desc("是否对逐元素 linalg.generic 做分块"),
      llvm::cl::init(true)};

  StringRef getArgument() const final { return "mydialect-linalg-opt"; }
  StringRef getDescription() const final {
    return "Tile elementwise linalg.generic ops (buffer semantics) before "
           "lowering to loops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    if (!tileEnabled)
      return;
    if (tileSize <= 0) {
      signalPassFailure();
      return;
    }

    ModuleOp module = getOperation();

    // 先收集候选，再统一改写；只处理未被其它 linalg op 包裹的顶层 op，
    // 避免对分块产生的内层 op 再次分块。
    SmallVector<linalg::LinalgOp> targets;
    module.walk([&](Operation *op) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp)
        return;
      // 只分块逐元素 generic（全部 parallel 迭代）
      if (!isa<linalg::GenericOp>(op) || linalgOp.getNumLoops() == 0 ||
          linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
        return;
      if (op->getParentOfType<linalg::LinalgOp>())
        return;
      // 静态首维不超过 tile size 时分块无收益，跳过
      Value out = linalgOp.getDpsInitOperand(0)->get();
      if (auto memTy = dyn_cast<MemRefType>(out.getType())) {
        int64_t d0 = memTy.getDimSize(0);
        if (!ShapedType::isDynamic(d0) && d0 <= tileSize)
          return;
      }
      targets.push_back(linalgOp);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : targets) {
      // 逐个分块，每个 op 用独立的 tile sizes：仅最外层非 0
      SmallVector<int64_t> sizes(op.getNumLoops(), 0);
      sizes[0] = tileSize;
      linalg::LinalgTilingOptions options;
      options.setTileSizes(sizes);
      // tileLinalgOp 只负责生成分块循环，不删除原 op；buffer 语义下原 op
      // 无结果，分块成功后直接删除，否则会保留一份未分块的重复计算。
      FailureOr<linalg::TiledLinalgOp> tiled =
          linalg::tileLinalgOp(rewriter, op, options);
      if (succeeded(tiled) && op->getNumResults() == 0)
        rewriter.eraseOp(op);
    }
  }
};

static PassRegistration<LinalgOptimizePass> regLinalgOpt;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createLinalgOptimizePass() {
  return std::make_unique<LinalgOptimizePass>();
}
} // namespace mydialect
