// FusionPass.cpp：--mydialect-fuse-elementwise
// 把 add_tensor/mul_tensor 的 producer-consumer 链融合成单个
// mydialect.fused_elementwise（变参输入张量 + region 内标量计算）。
// 融合条件：producer 单使用（hasOneUse）——多用 producer 保留为叶子输入，
// 避免复制被其他消费者共享的计算。shape/elem 一致性由 add/mul/fused 各自
// 的 verifier 沿链传导保证，融合时无需重复检查。
#include "MyOptPass.h"
#include "MyDialect.h"
#include "MyDialectOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <functional>

using namespace mlir;

namespace mydialect {
namespace {

// 可被吸收进 fused_elementwise 的逐元素 producer
static bool isFusibleOp(Operation *op) {
  return isa<AddTensorOp, MulTensorOp, FusedElementwiseOp>(op);
}

// 从 v 出发递归收集叶子输入（leaves/leafIndex 去重保序）与被吸收的
// producer（consumed，先序：使用者在前，其操作数 producer 在后）。
static void collectLeaves(Value v, SmallVectorImpl<Value> &leaves,
                          DenseMap<Value, unsigned> &leafIndex,
                          SmallVectorImpl<Operation *> &consumed,
                          SmallPtrSetImpl<Operation *> &consumedSet) {
  Operation *def = v.getDefiningOp();
  if (def && isFusibleOp(def) && v.hasOneUse()) {
    consumed.push_back(def);
    consumedSet.insert(def);
    for (Value operand : def->getOperands())
      collectLeaves(operand, leaves, leafIndex, consumed, consumedSet);
    return;
  }
  if (leafIndex.try_emplace(v, leaves.size()).second)
    leaves.push_back(v);
}

// 匹配根：add_tensor / mul_tensor（模板对两个具体 op 分别实例化；
// MLIR 22 不允许 OpRewritePattern<Operation*>，与基类 final 重载冲突）。
// 只要至少吸收一个 producer 就融合，
// 链式结构经 greedy 迭代逐层合并（add(fused, x) 会再次命中本 pattern）。
template <typename OpTy>
struct FuseElementwisePattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy rootOp,
                                PatternRewriter &rewriter) const override {

    auto loc = rootOp->getLoc();
    auto resTy = cast<MyTensorType>(rootOp->getResult(0).getType());
    Type elemTy = resTy.getElementType();
    bool isInt = isa<IntegerType>(elemTy);

    // ---- 1. 分析：收集叶子输入与被吸收 op ----
    SmallVector<Value> leaves;
    DenseMap<Value, unsigned> leafIndex;
    SmallVector<Operation *> consumed;
    SmallPtrSet<Operation *, 8> consumedSet;
    for (Value operand : rootOp->getOperands())
      collectLeaves(operand, leaves, leafIndex, consumed, consumedSet);

    // 未吸收任何 producer（如 add_tensor %arg0, %arg1）则无融合收益，跳过
    if (consumed.empty())
      return failure();

    // ---- 2. 构造 fused_elementwise：变参输入 + 单块 region ----
    auto fused = FusedElementwiseOp::create(rewriter, loc, resTy, leaves);
    Block *blk = rewriter.createBlock(&fused.getBody());
    blk->addArguments(SmallVector<Type>(leaves.size(), elemTy),
                      SmallVector<Location>(leaves.size(), loc));
    rewriter.setInsertionPointToStart(blk);

    // ---- 3. 递归发射标量表达式（遍历结构与 collectLeaves 一致）----
    // consumed 中的 op 内联展开为标量 arith；叶子映射到对应 block 参数
    std::function<Value(Value)> emit = [&](Value v) -> Value {
      Operation *def = v.getDefiningOp();
      if (!def || !consumedSet.contains(def)) {
        auto it = leafIndex.find(v);
        assert(it != leafIndex.end() && "value must be a collected leaf");
        return blk->getArgument(it->second);
      }
      if (auto fusedDef = dyn_cast<FusedElementwiseOp>(def)) {
        // 内联已有 fused 的 body：其块参数映射到 inputs 的 emit 结果
        IRMapping mapping;
        Block &innerBlk = fusedDef.getBody().front();
        for (auto [arg, in] :
             llvm::zip(innerBlk.getArguments(), fusedDef.getInputs()))
          mapping.map(arg, emit(in));
        Value yielded;
        for (Operation &inner : innerBlk) {
          if (auto y = dyn_cast<FusedYieldOp>(inner)) {
            yielded = mapping.lookup(y.getValue());
            continue;
          }
          rewriter.clone(inner, mapping);
        }
        return yielded;
      }
      // add/mul 张量算子 → 标量 arith.add/addf 或 mul/mulf
      Value l = emit(def->getOperand(0));
      Value r = emit(def->getOperand(1));
      if (isa<AddTensorOp>(def))
        return isInt ? Value(rewriter.create<arith::AddIOp>(loc, l, r).getResult())
                     : Value(rewriter.create<arith::AddFOp>(loc, l, r).getResult());
      return isInt ? Value(rewriter.create<arith::MulIOp>(loc, l, r).getResult())
                   : Value(rewriter.create<arith::MulFOp>(loc, l, r).getResult());
    };

    // 根 op 自身的标量计算作为 body 表达式之根
    Value l = emit(rootOp->getOperand(0));
    Value r = emit(rootOp->getOperand(1));
    Value result;
    if (isa<AddTensorOp>(rootOp))
      result = isInt ? Value(rewriter.create<arith::AddIOp>(loc, l, r).getResult())
                     : Value(rewriter.create<arith::AddFOp>(loc, l, r).getResult());
    else
      result = isInt ? Value(rewriter.create<arith::MulIOp>(loc, l, r).getResult())
                     : Value(rewriter.create<arith::MulFOp>(loc, l, r).getResult());
    FusedYieldOp::create(rewriter, loc, result);

    rewriter.replaceOp(rootOp, fused.getResult());
    // consumed 先序收集（使用者在先），正序擦除保证先删使用者再删其操作数
    for (Operation *c : consumed)
      if (c->use_empty())
        rewriter.eraseOp(c);
    return success();
  }
};

struct FusionElementwisePass
    : public PassWrapper<FusionElementwisePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionElementwisePass)
  StringRef getArgument() const final { return "mydialect-fuse-elementwise"; }
  StringRef getDescription() const final {
    return "Fuse producer-consumer elementwise tensor ops (add/mul chains) "
           "into mydialect.fused_elementwise";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseElementwisePattern<AddTensorOp>,
                 FuseElementwisePattern<MulTensorOp>>(&getContext());
    GreedyRewriteConfig config;
    config.setMaxIterations(16); // 防死循环，链式融合逐层收敛
    config.setUseTopDownTraversal(true);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

static PassRegistration<FusionElementwisePass> regFusion;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createFusionElementwisePass() {
  return std::make_unique<FusionElementwisePass>();
}
} // namespace mydialect
