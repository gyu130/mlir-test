// FusionPass.cpp：--mydialect-fuse-elementwise
// 把 add_tensor/mul_tensor 的 producer-consumer 链融合成单个
// mydialect.fused_elementwise（变参输入张量 + region 内标量计算）。
//
// 融合判定（代价模型）：
//   - 组内多用免复制：若 producer 的所有使用者都在本次融合组内（含根 op），
//     吸收它不产生任何复制——body 内该表达式只计算一次并被多处复用，
//     因此总是融合。
//   - 跨组多用需代价判定：若 producer 仍被融合组之外的使用者引用，吸收它会把
//     表达式复制进 body（原 op 仍需为组外使用者保留）。只有当
//     --fusion-max-dup 允许复制（>=2），且代价模型认为「复制一份子树的周期
//     小于省下的中间张量一次 store+load」时才融合。默认 --fusion-max-dup=1
//     表示不复制，保持原有单使用融合语义。
//   - --fusion-live-regs=N（N>0）用 Sethi-Ullman 数估计融合表达式的寄存器
//     压力，超过上限则放弃本次融合，避免过深表达式造成寄存器溢出。
//   - --fusion-model=heuristic|hw 选择周期模型：heuristic 对各算子延迟直接累加；
//     hw 复用 OpCountCycleEstimatePass 的 FU/延迟流水线模型
//     ceil(N/fu)+latency-1，加法/乘法两条流水线取最大值。
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
#include <algorithm>
#include <cassert>
#include <functional>
#include <string>

using namespace mlir;

namespace mydialect {
namespace {

// 可被吸收进 fused_elementwise 的逐元素 producer
static bool isFusibleOp(Operation *op) {
  return op && isa<AddTensorOp, MulTensorOp, FusedElementwiseOp>(op);
}

// 从 v 出发收集根 op 可达的所有可融合 producer（不限制使用次数）。
// order 为先序：使用者在前、其操作数 producer 在后。
static void collectCandidates(Value v, SmallVectorImpl<Operation *> &order,
                              SmallPtrSetImpl<Operation *> &set) {
  Operation *def = v.getDefiningOp();
  if (!isFusibleOp(def) || !set.insert(def).second)
    return;
  order.push_back(def);
  for (Value operand : def->getOperands())
    collectCandidates(operand, order, set);
}

// producer 在「融合组外」的使用次数。root 连同组内使用者都视为组内。
static unsigned externalUseCount(Operation *def,
                                 const SmallPtrSetImpl<Operation *> &absorbed,
                                 Operation *rootOp) {
  unsigned n = 0;
  for (Value res : def->getResults())
    for (Operation *user : res.getUsers())
      if (user != rootOp && !absorbed.contains(user))
        ++n;
  return n;
}

// 融合代价模型：估计逐元素表达子树的计算周期，判定跨组复制是否划算。
struct FusionCostModel {
  // true=硬件 FU/延迟流水线模型；false=简单延迟累加
  bool hw;

  // 与 OpCountCycleEstimatePass 的默认参数保持一致
  static constexpr int64_t addLatency = 1;
  static constexpr int64_t mulLatency = 4;
  static constexpr int64_t addFU = 2;
  static constexpr int64_t mulFU = 1;
  // 中间张量一次 store 或 load 的估计周期
  static constexpr int64_t memOpCycles = 2;

  // 统计 def 子树内（仅组内已吸收 op 会被复制）的加法/乘法数量
  static void count(Operation *op,
                    const SmallPtrSetImpl<Operation *> &inGroup,
                    int64_t &adds, int64_t &muls) {
    if (!op || !inGroup.contains(op))
      return;
    if (isa<AddTensorOp>(op)) {
      ++adds;
    } else if (isa<MulTensorOp>(op)) {
      ++muls;
    } else if (auto fused = dyn_cast<FusedElementwiseOp>(op)) {
      for (Operation &inner : fused.getBody().front()) {
        if (isa<arith::AddIOp, arith::AddFOp>(inner))
          ++adds;
        else if (isa<arith::MulIOp, arith::MulFOp>(inner))
          ++muls;
      }
      for (Value operand : fused.getInputs())
        count(operand.getDefiningOp(), inGroup, adds, muls);
      return;
    }
    for (Value operand : op->getOperands())
      count(operand.getDefiningOp(), inGroup, adds, muls);
  }

  // N 个同类操作占用 fu 条流水线、latency 拍延迟时所需的拍数
  static int64_t pipelineCycles(int64_t n, int64_t fu, int64_t lat) {
    if (n <= 0)
      return 0;
    if (fu <= 0)
      fu = 1;
    return (n + fu - 1) / fu + lat - 1;
  }

  int64_t estimate(Operation *def,
                   const SmallPtrSetImpl<Operation *> &inGroup) const {
    int64_t adds = 0, muls = 0;
    count(def, inGroup, adds, muls);
    if (!hw)
      return adds * addLatency + muls * mulLatency;
    return std::max(pipelineCycles(adds, addFU, addLatency),
                    pipelineCycles(muls, mulFU, mulLatency));
  }

  // 复制一份 producer 省下中间张量的一次 store+load；复制成本低于收益才划算
  bool duplicationWorthwhile(
      Operation *def, const SmallPtrSetImpl<Operation *> &inGroup) const {
    return estimate(def, inGroup) < 2 * memOpCycles;
  }
};

// Sethi-Ullman 数：评估二元表达子树所需的最少寄存器数。
// 仅用于 --fusion-live-regs 的门槛判断，fused 体近似为单值。
static int sethiUllman(Value v, const SmallPtrSetImpl<Operation *> &inGroup) {
  Operation *def = v.getDefiningOp();
  if (!isFusibleOp(def) || !inGroup.contains(def))
    return 1;
  if (isa<FusedElementwiseOp>(def))
    return 1;
  int a = sethiUllman(def->getOperand(0), inGroup);
  int b = sethiUllman(def->getOperand(1), inGroup);
  return a == b ? a + 1 : std::max(a, b);
}

// 匹配根：add_tensor / mul_tensor（模板对两个具体 op 分别实例化；
// MLIR 22 不允许 OpRewritePattern<Operation*>，与基类 final 重载冲突）。
// 只要至少吸收一个 producer 就融合，
// 链式结构经 greedy 迭代逐层合并（add(fused, x) 会再次命中本 pattern）。
template <typename OpTy>
struct FuseElementwisePattern : public OpRewritePattern<OpTy> {
  FuseElementwisePattern(MLIRContext *ctx, int maxDup, int liveRegs,
                         bool hwModel)
      : OpRewritePattern<OpTy>(ctx), maxDup(maxDup),
        fusionLiveRegs(liveRegs), costModel{hwModel} {}

  int maxDup;           // 允许的 producer 副本数；1=不复制
  int fusionLiveRegs;   // 0=不限制
  FusionCostModel costModel;

  LogicalResult matchAndRewrite(OpTy rootOp,
                                PatternRewriter &rewriter) const override {

    auto loc = rootOp->getLoc();
    auto resTy = cast<MyTensorType>(rootOp->getResult(0).getType());
    Type elemTy = resTy.getElementType();
    bool isInt = isa<IntegerType>(elemTy);

    // ---- 1. 收集候选 producer（不限使用次数）----
    SmallVector<Operation *> candidates;
    SmallPtrSet<Operation *, 8> candidateSet;
    for (Value operand : rootOp->getOperands())
      collectCandidates(operand, candidates, candidateSet);
    if (candidates.empty())
      return failure();

    // ---- 2. 吸收决策：组内多用免复制；跨组多用按代价模型判定 ----
    SmallPtrSet<Operation *, 8> absorbed(candidateSet.begin(),
                                         candidateSet.end());
    // 从 absorbed 中剔除不划算的 producer 会影响其上游 producer 的「组外使用」
    // 判定，故迭代到不动点。
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation *def : candidates) {
        if (!absorbed.contains(def))
          continue;
        bool duplicated = externalUseCount(def, absorbed, rootOp) > 0;
        bool allow;
        if (!duplicated)
          allow = true;               // 组内多用：无复制成本
        else if (maxDup < 2)
          allow = false;              // 未开启跨组复制（默认）
        else
          allow = costModel.duplicationWorthwhile(def, absorbed);
        if (!allow) {
          absorbed.erase(def);
          changed = true;
        }
      }
    }
    if (absorbed.empty())
      return failure();

    // ---- 3. 收集叶子输入与被吸收 op（先序：使用者在前）----
    SmallVector<Value> leaves;
    DenseMap<Value, unsigned> leafIndex;
    SmallVector<Operation *> consumed;
    SmallPtrSet<Operation *, 8> consumedSet;
    std::function<void(Value)> collect = [&](Value v) {
      Operation *def = v.getDefiningOp();
      if (isFusibleOp(def) && absorbed.contains(def)) {
        if (consumedSet.insert(def).second) {
          consumed.push_back(def);
          for (Value operand : def->getOperands())
            collect(operand);
        }
        return;
      }
      if (leafIndex.try_emplace(v, leaves.size()).second)
        leaves.push_back(v);
    };
    for (Value operand : rootOp->getOperands())
      collect(operand);

    // ---- 4. 寄存器压力门槛（Sethi-Ullman）----
    if (fusionLiveRegs > 0) {
      int a = sethiUllman(rootOp->getOperand(0), consumedSet);
      int b = sethiUllman(rootOp->getOperand(1), consumedSet);
      int su = (a == b) ? a + 1 : std::max(a, b);
      if (su > fusionLiveRegs)
        return failure();
    }

    // ---- 5. 构造 fused_elementwise：变参输入 + 单块 region ----
    auto fused = FusedElementwiseOp::create(rewriter, loc, resTy, leaves);
    Block *blk = rewriter.createBlock(&fused.getBody());
    blk->addArguments(SmallVector<Type>(leaves.size(), elemTy),
                      SmallVector<Location>(leaves.size(), loc));
    rewriter.setInsertionPointToStart(blk);

    // ---- 6. 递归发射标量表达式，组内多用经 memo 只算一次 ----
    DenseMap<Value, Value> emitted;
    std::function<Value(Value)> emit = [&](Value v) -> Value {
      auto hit = emitted.find(v);
      if (hit != emitted.end())
        return hit->second;

      Operation *def = v.getDefiningOp();
      if (!isFusibleOp(def) || !consumedSet.contains(def)) {
        auto it = leafIndex.find(v);
        assert(it != leafIndex.end() && "value must be a collected leaf");
        Value arg = blk->getArgument(it->second);
        emitted[v] = arg;
        return arg;
      }

      Value result;
      if (auto fusedDef = dyn_cast<FusedElementwiseOp>(def)) {
        // 内联已有 fused 的 body：其块参数映射到 inputs 的 emit 结果
        IRMapping mapping;
        Block &innerBlk = fusedDef.getBody().front();
        for (auto [arg, in] :
             llvm::zip(innerBlk.getArguments(), fusedDef.getInputs()))
          mapping.map(arg, emit(in));
        for (Operation &inner : innerBlk) {
          if (auto y = dyn_cast<FusedYieldOp>(inner)) {
            result = mapping.lookup(y.getValue());
            continue;
          }
          rewriter.clone(inner, mapping);
        }
      } else {
        // add/mul 张量算子 → 标量 arith.add/addf 或 mul/mulf
        Value l = emit(def->getOperand(0));
        Value r = emit(def->getOperand(1));
        if (isa<AddTensorOp>(def))
          result =
              isInt ? Value(rewriter.create<arith::AddIOp>(loc, l, r).getResult())
                    : Value(rewriter.create<arith::AddFOp>(loc, l, r).getResult());
        else
          result =
              isInt ? Value(rewriter.create<arith::MulIOp>(loc, l, r).getResult())
                    : Value(rewriter.create<arith::MulFOp>(loc, l, r).getResult());
      }
      emitted[v] = result;
      return result;
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

    // 擦除被吸收的 op：迭代到不动点，保证先删使用者再删其 producer；
    // 复制场景下仍被组外使用的原 op 因 use_empty 为假而保留。
    SmallVector<Operation *> toErase(consumed.begin(), consumed.end());
    bool erased = true;
    while (erased) {
      erased = false;
      for (auto it = toErase.begin(); it != toErase.end();) {
        Operation *c = *it;
        if (c->use_empty()) {
          rewriter.eraseOp(c);
          it = toErase.erase(it);
          erased = true;
        } else {
          ++it;
        }
      }
    }
    return success();
  }
};

struct FusionElementwisePass
    : public PassWrapper<FusionElementwisePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionElementwisePass)
  FusionElementwisePass() = default;
  // Pass::Option 不可拷贝，需显式提供拷贝构造（选项值由 clone() 单独拷贝）
  FusionElementwisePass(const FusionElementwisePass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const final { return "mydialect-fuse-elementwise"; }
  StringRef getDescription() const final {
    return "Fuse producer-consumer elementwise tensor ops (add/mul chains) "
           "into mydialect.fused_elementwise with a fusion cost model";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  // 允许的 producer 副本数：1 表示不复制，只有所有使用者都在融合组内才吸收
  Option<int> fusionMaxDup{
      *this, "fusion-max-dup",
      llvm::cl::desc("Max copies of a producer allowed (1 = no duplication: "
                     "fuse only when all uses are inside the fusion group)"),
      llvm::cl::init(1)};
  // 融合表达式寄存器压力上限（Sethi-Ullman 数）；0 表示不限制
  Option<int> fusionLiveRegs{
      *this, "fusion-live-regs",
      llvm::cl::desc("Max Sethi-Ullman register pressure of a fused expression "
                     "(0 = unlimited)"),
      llvm::cl::init(0)};
  // 跨组复制的代价模型
  Option<std::string> fusionModel{
      *this, "fusion-model",
      llvm::cl::desc("Cost model for cross-group duplication: heuristic|hw"),
      llvm::cl::init("heuristic")};

  void runOnOperation() override {
    bool hwModel = (fusionModel == "hw");
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseElementwisePattern<AddTensorOp>,
                 FuseElementwisePattern<MulTensorOp>>(
        &getContext(), static_cast<int>(fusionMaxDup),
        static_cast<int>(fusionLiveRegs), hwModel);
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
