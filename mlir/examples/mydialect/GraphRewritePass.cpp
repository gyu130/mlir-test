// GraphRewritePass.cpp：--mydialect-graph-rewrite
// 通用「算子替换规则库」驱动的图重写 pass。规则以具名 RewriteRule 组织
// （见 RewriteRuleSet.h），通过 --rules=a,b,c 单独开关，--max-iter 控制
// 迭代轮数，--fold 控制贪婪驱动是否启用常量折叠。
//
// 规则来源分两类：
//   1. 复用既有优化里的代数化简/布局置换规则（与 --mydialect-opt、
//      --mydialect-layout-propagation 语义一致），在此登记为稳定名称；
//   2. 新增算子替换规则：relu(relu)、reshape 恒等、add 0 / mul 1、
//      matmul(I,x)、concat 合并、transpose 复合。
#include "RewriteRuleSet.h"
#include "MyDialect.h"
#include "MyDialectOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

using namespace mlir;

namespace mydialect {
namespace {

// ---- 常量张量判定辅助 ----

// 若 v 由 constant_tensor 定义则返回其 ElementsAttr
static bool getConstantAttr(Value v, ElementsAttr &attr) {
  auto c = v.getDefiningOp<ConstantTensorOp>();
  if (!c)
    return false;
  attr = c.getValue();
  return true;
}

static bool isAllZero(Value v) {
  ElementsAttr attr;
  if (!getConstantAttr(v, attr))
    return false;
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return false;
  Attribute splat = dense.getSplatValue<Attribute>();
  if (auto ia = dyn_cast<IntegerAttr>(splat))
    return ia.getValue().isZero();
  if (auto fa = dyn_cast<FloatAttr>(splat))
    return fa.getValue().isZero();
  return false;
}

static bool isAllOne(Value v) {
  ElementsAttr attr;
  if (!getConstantAttr(v, attr))
    return false;
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return false;
  Attribute splat = dense.getSplatValue<Attribute>();
  if (auto ia = dyn_cast<IntegerAttr>(splat))
    return ia.getValue().isOne();
  if (auto fa = dyn_cast<FloatAttr>(splat))
    return fa.getValue().isExactlyValue(1.0);
  return false;
}

// 判定 v 是否为 2-D 整数单位矩阵常量
static bool isIdentityMatrix(Value v) {
  ElementsAttr attr;
  if (!getConstantAttr(v, attr))
    return false;
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  if (!dense)
    return false;
  auto ty = dyn_cast<ShapedType>(dense.getType());
  if (!ty || !ty.hasStaticShape() || ty.getRank() != 2)
    return false;
  int64_t n = ty.getDimSize(0), m = ty.getDimSize(1);
  if (n != m || n <= 0 || !ty.getElementType().isIntOrIndex())
    return false;
  int64_t idx = 0;
  for (APInt val : dense.getValues<APInt>()) {
    bool diag = (idx / n) == (idx % n);
    if (diag ? !val.isOne() : !val.isZero())
      return false;
    ++idx;
  }
  return true;
}

// ================= 复用既有代数化简 / 布局置换规则 =================

// 删除连续冗余 print（同 --mydialect-opt）
struct RemoveRedundantPrintPattern : public OpRewritePattern<PrintOp> {
  RemoveRedundantPrintPattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<PrintOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(PrintOp op,
                                PatternRewriter &rewriter) const override {
    Block *block = op->getBlock();
    auto it = op->getIterator();
    if (it == block->begin())
      return failure();
    --it;
    auto prev = dyn_cast<PrintOp>(&*it);
    if (prev && prev.getValue() == op.getValue()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// 块内 arith.addi 公共子表达式消除（同 --mydialect-opt）
struct AddiCSEPattern : public OpRewritePattern<arith::AddIOp> {
  AddiCSEPattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<arith::AddIOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    Block *block = op->getBlock();
    for (auto it = op->getIterator(); it != block->begin();) {
      --it;
      auto prev = dyn_cast<arith::AddIOp>(&*it);
      if (prev && prev.getLhs() == op.getLhs() && prev.getRhs() == op.getRhs()) {
        rewriter.replaceOp(op, prev.getResult());
        return success();
      }
    }
    return failure();
  }
};

// identity_tensor(x) → x
struct IdentityTensorPattern : public OpRewritePattern<IdentityTensorOp> {
  IdentityTensorPattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<IdentityTensorOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(IdentityTensorOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getIn());
    return success();
  }
};

// transpose(transpose(x,p1),p2) → x（两次置换复合为恒等）
struct DoubleTransposePattern : public OpRewritePattern<TransposeTensorOp> {
  DoubleTransposePattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<TransposeTensorOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(TransposeTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getData().getDefiningOp<TransposeTensorOp>();
    if (!inner || op.getResult().getType() != inner.getData().getType())
      return failure();
    auto p2 = op.getPerm();
    auto p1 = inner.getPerm();
    if (p1.size() != p2.size())
      return failure();
    for (size_t m = 0; m < p2.size(); ++m) {
      int64_t oi = cast<IntegerAttr>(p2[m]).getInt();
      int64_t xi = cast<IntegerAttr>(p1[oi]).getInt();
      if (xi != static_cast<int64_t>(m))
        return failure();
    }
    rewriter.replaceOp(op, inner.getData());
    if (inner->use_empty())
      rewriter.eraseOp(inner);
    return success();
  }
};

// reshape(reshape(x,s1),s2) → reshape(x,s2)
struct ChainedReshapePattern : public OpRewritePattern<ReshapeTensorOp> {
  ChainedReshapePattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<ReshapeTensorOp>(ctx, benefit) {}
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

// ===================== 新增算子替换规则 =====================

// relu(relu(x)) → relu(x)
struct ReluOfReluPattern : public OpRewritePattern<ReluTensorOp> {
  ReluOfReluPattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<ReluTensorOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(ReluTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getX().getDefiningOp<ReluTensorOp>();
    if (!inner)
      return failure();
    rewriter.replaceOp(op, inner.getResult());
    if (inner->use_empty())
      rewriter.eraseOp(inner);
    return success();
  }
};

// reshape(x, s) 且结果类型与 x 相同 → x
struct ReshapeIdentityPattern : public OpRewritePattern<ReshapeTensorOp> {
  ReshapeIdentityPattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<ReshapeTensorOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(ReshapeTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getResult().getType() != op.getData().getType())
      return failure();
    rewriter.replaceOp(op, op.getData());
    return success();
  }
};

// add_tensor(x, 0) / add_tensor(0, x) → x
struct AddZeroPattern : public OpRewritePattern<AddTensorOp> {
  AddZeroPattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<AddTensorOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(AddTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (isAllZero(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if (isAllZero(op.getLhs())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

// mul_tensor(x, 1) / mul_tensor(1, x) → x
struct MulOnePattern : public OpRewritePattern<MulTensorOp> {
  MulOnePattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<MulTensorOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(MulTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (isAllOne(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if (isAllOne(op.getLhs())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

// matmul(A, I) → A，matmul(I, B) → B（I 为 2-D 单位矩阵常量）
struct MatMulIdentityPattern : public OpRewritePattern<MatMulTensorOp> {
  MatMulIdentityPattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<MatMulTensorOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(MatMulTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (isIdentityMatrix(op.getB()) &&
        op.getResult().getType() == op.getA().getType()) {
      rewriter.replaceOp(op, op.getA());
      return success();
    }
    if (isIdentityMatrix(op.getA()) &&
        op.getResult().getType() == op.getB().getType()) {
      rewriter.replaceOp(op, op.getB());
      return success();
    }
    return failure();
  }
};

// concat(concat(a,b,axis), c, axis) → concat(a,b,c,axis)
struct ConcatMergePattern : public OpRewritePattern<ConcatTensorOp> {
  ConcatMergePattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<ConcatTensorOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(ConcatTensorOp op,
                                PatternRewriter &rewriter) const override {
    for (Value in : op.getInputs()) {
      auto inner = in.getDefiningOp<ConcatTensorOp>();
      if (!inner || inner.getAxis() != op.getAxis())
        continue;
      SmallVector<Value> newInputs;
      for (Value other : op.getInputs()) {
        if (other == in)
          llvm::append_range(newInputs, inner.getInputs());
        else
          newInputs.push_back(other);
      }
      auto newOp = rewriter.create<ConcatTensorOp>(
          op.getLoc(), op.getResult().getType(), newInputs, op.getAxis());
      rewriter.replaceOp(op, newOp.getResult());
      if (inner->use_empty())
        rewriter.eraseOp(inner);
      return success();
    }
    return failure();
  }
};

// transpose(transpose(x,p1),p2) → transpose(x, q)，q[m] = p1[p2[m]]（非恒等）
struct ComposeTransposePattern : public OpRewritePattern<TransposeTensorOp> {
  ComposeTransposePattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<TransposeTensorOp>(ctx, benefit) {}
  LogicalResult matchAndRewrite(TransposeTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getData().getDefiningOp<TransposeTensorOp>();
    if (!inner)
      return failure();
    auto p2 = op.getPerm();
    auto p1 = inner.getPerm();
    if (p1.size() != p2.size())
      return failure();

    SmallVector<Attribute> perm;
    bool identity = true;
    for (size_t m = 0; m < p2.size(); ++m) {
      int64_t oi = cast<IntegerAttr>(p2[m]).getInt();
      int64_t xi = cast<IntegerAttr>(p1[oi]).getInt();
      if (xi != static_cast<int64_t>(m))
        identity = false;
      perm.push_back(rewriter.getIndexAttr(xi));
    }
    // 恒等复合交给 double-transpose 直接折成 x，避免生成 identity transpose
    if (identity)
      return failure();

    auto newOp = rewriter.create<TransposeTensorOp>(
        op.getLoc(), op.getResult().getType(), inner.getData(),
        rewriter.getArrayAttr(perm));
    rewriter.replaceOp(op, newOp.getResult());
    if (inner->use_empty())
      rewriter.eraseOp(inner);
    return success();
  }
};

// ===================== 规则集定义 =====================

struct GraphRewritePass
    : public PassWrapper<GraphRewritePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GraphRewritePass)
  GraphRewritePass() = default;
  GraphRewritePass(const GraphRewritePass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "mydialect-graph-rewrite"; }
  StringRef getDescription() const final {
    return "Apply the named mydialect graph rewrite rule set";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  Option<int> maxIter{
      *this, "max-iter",
      llvm::cl::desc("Max greedy rewrite iterations"), llvm::cl::init(10)};
  Option<bool> fold{
      *this, "fold",
      llvm::cl::desc("Enable constant folding in the greedy driver"),
      llvm::cl::init(true)};
  Option<std::string> rules{
      *this, "rules",
      llvm::cl::desc("Comma-separated rule names to enable (empty = defaults)"),
      llvm::cl::init("")};

  void runOnOperation() override {
    RewriteRuleSet set = getMyDialectGraphRewriteRuleSet();

    std::string rulesStr = rules.getValue();
    SmallVector<StringRef> selected;
    if (!rulesStr.empty())
      StringRef(rulesStr).split(selected, ',');

    RewritePatternSet patterns(&getContext());
    for (const RewriteRule &r : set.rules) {
      bool enabled = rulesStr.empty()
                         ? r.defaultOn
                         : llvm::is_contained(selected, r.name);
      if (enabled)
        r.populate(patterns, &getContext(), PatternBenefit(r.benefit));
    }

    GreedyRewriteConfig config;
    config.setMaxIterations(maxIter);
    config.setUseTopDownTraversal(true);
    config.enableFolding(fold);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
  }
};

static PassRegistration<GraphRewritePass> regGraphRewrite;

} // anonymous namespace

RewriteRuleSet getMyDialectGraphRewriteRuleSet() {
  return RewriteRuleSet{
      "mydialect-graph",
      {
          // 复用既有代数化简 / 布局置换规则
          {"remove-redundant-print", 1, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<RemoveRedundantPrintPattern>(ctx, b);
           }},
          {"addi-cse", 2, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<AddiCSEPattern>(ctx, b);
           }},
          {"identity-tensor", 3, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<IdentityTensorPattern>(ctx, b);
           }},
          {"double-transpose", 4, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<DoubleTransposePattern>(ctx, b);
           }},
          {"chained-reshape", 4, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<ChainedReshapePattern>(ctx, b);
           }},
          // 新增算子替换规则
          {"relu-of-relu", 5, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<ReluOfReluPattern>(ctx, b);
           }},
          {"reshape-identity", 3, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<ReshapeIdentityPattern>(ctx, b);
           }},
          {"add-zero", 5, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<AddZeroPattern>(ctx, b);
           }},
          {"mul-one", 5, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<MulOnePattern>(ctx, b);
           }},
          {"matmul-identity", 5, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<MatMulIdentityPattern>(ctx, b);
           }},
          {"concat-merge", 4, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<ConcatMergePattern>(ctx, b);
           }},
          {"compose-transpose", 4, true,
           [](RewritePatternSet &p, MLIRContext *ctx, PatternBenefit b) {
             p.add<ComposeTransposePattern>(ctx, b);
           }},
      }};
}

} // namespace mydialect
