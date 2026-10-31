// LayoutPropagationPass.cpp：--mydialect-layout-propagation
// Phase 6.3 layout 传播：
//   1. 把 elementwise 算子输入端的布局转换（transpose）提升到消费端，
//      add_tensor/mul_tensor/cmpi_tensor(transpose(x,p), transpose(y,p))
//        → transpose(add_tensor(x,y), p)
//      N 个输入转换合并成 1 个，并把转换推到图的下游，暴露可融合的
//      elementwise 链（供 FusionPass 使用）；
//   2. 相邻布局转换复合为单个：transpose(transpose(x,p1),p2)
//        → transpose(x, p1∘p2)，复合为恒等时直接消去；
//   3. 恒等置换（perm == identity）是纯拷贝，直接消去。
// 布局转换只在无法继续传播的边界（print/reshape/reduce 等非 elementwise
// 消费端）真正落地，即「按需物化」。
#include "MyOptPass.h"
#include "MyDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <type_traits>

using namespace mlir;

namespace mydialect {
namespace {

struct LayoutStats {
  unsigned hoisted = 0;   // elementwise 输入端布局转换被提升合并
  unsigned composed = 0;  // 相邻布局转换复合为单个 non-identity
  unsigned cancelled = 0; // 复合为恒等 / 恒等置换直接消去
};

static bool samePerm(ArrayAttr a, ArrayAttr b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (cast<IntegerAttr>(a[i]).getInt() != cast<IntegerAttr>(b[i]).getInt())
      return false;
  return true;
}

static bool isIdentityPerm(ArrayAttr perm) {
  for (size_t i = 0; i < perm.size(); ++i)
    if (cast<IntegerAttr>(perm[i]).getInt() != static_cast<int64_t>(i))
      return false;
  return true;
}

// ==============================
// 恒等置换消去：transpose(x, identity) → x
// ==============================
struct IdentityPermFold : public OpRewritePattern<TransposeTensorOp> {
  IdentityPermFold(MLIRContext *ctx, LayoutStats &stats)
      : OpRewritePattern<TransposeTensorOp>(ctx), stats(stats) {}

  LogicalResult matchAndRewrite(TransposeTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (!isIdentityPerm(op.getPerm()))
      return failure();
    // 恒等置换下结果类型必定等于输入类型（verifier 已保证 perm 合法）
    rewriter.replaceOp(op, op.getData());
    ++stats.cancelled;
    return success();
  }

  LayoutStats &stats;
};

// ==============================
// 布局转换复合：transpose(transpose(x, p1), p2) → transpose(x, p1∘p2)
// 语义：out2 维 m ← out1 维 p2[m] ← x 维 p1[p2[m]]，故 composed[m]=p1[p2[m]]。
// 复合为恒等时直接消去，两次转换都不再落地。
// ==============================
struct ComposeTransposePattern : public OpRewritePattern<TransposeTensorOp> {
  ComposeTransposePattern(MLIRContext *ctx, LayoutStats &stats)
      : OpRewritePattern<TransposeTensorOp>(ctx), stats(stats) {}

  LogicalResult matchAndRewrite(TransposeTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getData().getDefiningOp<TransposeTensorOp>();
    if (!inner)
      return failure();
    ArrayAttr p1 = inner.getPerm();
    ArrayAttr p2 = op.getPerm();
    if (p1.size() != p2.size())
      return failure();

    MLIRContext *ctx = rewriter.getContext();
    SmallVector<Attribute> composed;
    for (size_t m = 0; m < p2.size(); ++m) {
      int64_t mid = cast<IntegerAttr>(p2[m]).getInt();
      // 与输入 IR 中 perm 的 i64 整型属性保持一致，避免打印成 index 形式
      composed.push_back(IntegerAttr::get(
          IntegerType::get(ctx, 64), cast<IntegerAttr>(p1[mid]).getInt()));
    }
    ArrayAttr composedPerm = ArrayAttr::get(ctx, composed);

    if (isIdentityPerm(composedPerm)) {
      rewriter.replaceOp(op, inner.getData());
      ++stats.cancelled;
    } else {
      auto newOp = rewriter.create<TransposeTensorOp>(
          op.getLoc(), op.getResult().getType(), inner.getData(), composedPerm);
      rewriter.replaceOp(op, newOp.getResult());
      ++stats.composed;
    }
    if (inner->use_empty())
      rewriter.eraseOp(inner);
    return success();
  }

  LayoutStats &stats;
};

// ==============================
// 布局传播：把 elementwise 输入端的同构 transpose 提升到结果之后
//   elementwise(transpose(x,p), transpose(y,p))
//     → transpose(elementwise(x,y), p)
// elementwise 逐元素运算与布局置换可交换，因此该重写语义等价；
// 多个输入转换合并成一个，转换数量严格减少，可继续向下游传播。
// ==============================
template <typename OpT>
struct HoistLayoutPattern : public OpRewritePattern<OpT> {
  HoistLayoutPattern(MLIRContext *ctx, LayoutStats &stats)
      : OpRewritePattern<OpT>(ctx), stats(stats) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();

    ArrayAttr perm;
    SmallVector<Value> data;
    for (Value operand : op->getOperands()) {
      auto tr = operand.getDefiningOp<TransposeTensorOp>();
      if (!tr)
        return failure();
      if (!perm)
        perm = tr.getPerm();
      else if (!samePerm(perm, tr.getPerm()))
        return failure();
      data.push_back(tr.getData());
    }

    // 自然布局（未置换）下的结果类型
    auto data0Ty = dyn_cast<MyTensorType>(data[0].getType());
    if (!data0Ty)
      return failure();

    Type resTy;
    if constexpr (std::is_same_v<OpT, CmpITensorOp>) {
      // cmpi_tensor 结果元素类型为 i1，shape 与输入一致
      if (cast<MyTensorType>(data[1].getType()).getShape() != data0Ty.getShape())
        return failure();
      resTy = MyTensorType::get(rewriter.getContext(), data0Ty.getShape(),
                                IntegerType::get(rewriter.getContext(), 1));
    } else {
      // add/mul 的输入与结果同类型
      if (data[1].getType() != data[0].getType())
        return failure();
      resTy = data[0].getType();
    }

    Operation *newOp;
    if constexpr (std::is_same_v<OpT, CmpITensorOp>) {
      newOp = rewriter.create<OpT>(op.getLoc(), resTy, data[0], data[1],
                                   op.getPredicate());
    } else {
      newOp = rewriter.create<OpT>(op.getLoc(), resTy, data[0], data[1]);
    }

    auto newTrans = rewriter.create<TransposeTensorOp>(
        op.getLoc(), op.getResult().getType(), newOp->getResult(0), perm);
    rewriter.replaceOp(op, newTrans.getResult());
    ++stats.hoisted;
    return success();
  }

  LayoutStats &stats;
};

struct LayoutPropagationPass
    : public PassWrapper<LayoutPropagationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutPropagationPass)
  LayoutPropagationPass() = default;
  LayoutPropagationPass(const LayoutPropagationPass &other)
      : PassWrapper(other) {}

  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Print layout propagation statistics"),
                       llvm::cl::init(false)};

  StringRef getArgument() const final {
    return "mydialect-layout-propagation";
  }
  StringRef getDescription() const final {
    return "Propagate tensor layouts: hoist layout conversions out of "
           "elementwise ops and compose adjacent conversions";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<MyDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LayoutStats stats;
    RewritePatternSet patterns(ctx);
    patterns.add<IdentityPermFold, ComposeTransposePattern>(ctx, stats);
    patterns.add<HoistLayoutPattern<AddTensorOp>, HoistLayoutPattern<MulTensorOp>,
                 HoistLayoutPattern<CmpITensorOp>>(ctx, stats);

    GreedyRewriteConfig config;
    config.setMaxIterations(10);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
      return;
    }

    if (verbose)
      llvm::errs() << "[mydialect-layout-propagation] hoisted=" << stats.hoisted
                   << ", composed=" << stats.composed
                   << ", cancelled=" << stats.cancelled << "\n";
  }
};

static PassRegistration<LayoutPropagationPass> regLayoutPropagation;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createLayoutPropagationPass() {
  return std::make_unique<LayoutPropagationPass>();
}
} // namespace mydialect
