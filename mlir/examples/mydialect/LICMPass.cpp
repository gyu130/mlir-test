// LICMPass.cpp：--mydialect-licm
// 通用循环不变量外提（Loop-Invariant Code Motion）。
//
// 规则：若某个叶子 op（无嵌套 region）同时满足
//   - 纯/可推测（isMemoryEffectFree && isSpeculatable；Pure 同时含
//     AlwaysSpeculatable + NoMemoryEffect）
//   - 所有操作数都定义在循环之外（既非循环块参数，也非循环内定义的值，
//     且定义点支配循环 op）
// 则把它上提到循环 op 之前。
//
// 与既有 ConstantFoldMyDialectPass 里的 LoopInvariantHoistPattern 的关系：
//   - 后者只针对 mydialect.for，且额外处理「循环内无别名写时的 invariant
//     load」；它服务于常量折叠流水线。
//   - 本 pass 是独立、聚焦的 LICM：覆盖 scf.for / scf.while / affine.for /
//     mydialect.for / mydialect.loop 等所有常见循环，只外提纯/可推测叶子 op；
//     通过多轮不动点把内层先外提、再逐层提到外层循环之外。
//   transpose/reshape 等纯张量算子因此可被提到循环外，配合 layout 传播复用。
#include "MyDialect.h"
#include "MyDialectOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mydialect {
namespace {

static bool isLoopLike(Operation *op) {
  return isa<scf::ForOp, scf::WhileOp, affine::AffineForOp, mydialect::ForOp,
             mydialect::LoopOp>(op);
}

// v 是否定义在 loop 之外（loop 及其嵌套子区域之外）
static bool isInsideLoop(Operation *loop, Value v) {
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    Block *owner = ba.getOwner();
    // 块的父 op：循环体自身块 → loop；函数入口块 → func
    return loop->isAncestor(owner->getParentOp()) || owner->getParentOp() == loop;
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  return def == loop || loop->isProperAncestor(def);
}

// v 定义在 loop 之外，且其定义点支配 loop op（保守结构判断，无需 DominanceInfo）
static bool valueDefinedOutsideAndDominates(Value v, Operation *loop) {
  if (isInsideLoop(loop, v))
    return false;

  Block *loopBlock = loop->getBlock();
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    Block *owner = ba.getOwner();
    return owner == loopBlock ||
           owner->getParent()->isProperAncestor(loopBlock->getParent());
  }
  Operation *def = v.getDefiningOp();
  Block *defBlock = def->getBlock();
  if (defBlock == loopBlock)
    return def->isBeforeInBlock(loop);
  // 外层块：要求 def 所在区域是 loop 所在区域的严格祖先
  return defBlock->getParent()->isProperAncestor(loopBlock->getParent());
}

// 把 loop 各 region 中满足条件的 op 上提到 loop 之前；返回是否有改动
static bool hoistLoopInvariants(Operation *loop) {
  bool hoisted = false;
  for (Region &region : loop->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (op.hasTrait<OpTrait::IsTerminator>())
          continue;
        // 只外提叶子 op，避免跨 region 语义（如 device_exec/scf.if）
        if (op.getNumRegions() != 0)
          continue;
        // 纯 + 可推测
        if (!isMemoryEffectFree(&op) || !isSpeculatable(&op))
          continue;
        bool allOutside = true;
        for (Value operand : op.getOperands()) {
          if (!valueDefinedOutsideAndDominates(operand, loop)) {
            allOutside = false;
            break;
          }
        }
        if (!allOutside)
          continue;
        op.moveBefore(loop);
        hoisted = true;
      }
    }
  }
  return hoisted;
}

struct MyDialectLICMPass
    : public PassWrapper<MyDialectLICMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyDialectLICMPass)
  MyDialectLICMPass() = default;
  MyDialectLICMPass(const MyDialectLICMPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "mydialect-licm"; }
  StringRef getDescription() const final {
    return "Hoist loop-invariant pure/speculatable ops out of loops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, affine::AffineDialect>();
  }

  Option<int> maxIter{
      *this, "max-iter",
      llvm::cl::desc("Max fixpoint iterations for nested hoisting"),
      llvm::cl::init(8)};

  void runOnOperation() override {
    bool changed = true;
    int iter = 0;
    while (changed && iter++ < maxIter) {
      changed = false;
      // 后序遍历：内层循环先处理，便于同轮把不变量逐层提到外层
      SmallVector<Operation *> loops;
      getOperation().walk<WalkOrder::PostOrder>([&](Operation *op) {
        if (isLoopLike(op))
          loops.push_back(op);
      });
      for (Operation *loop : loops)
        changed |= hoistLoopInvariants(loop);
    }
  }
};

static PassRegistration<MyDialectLICMPass> regLICM;

} // anonymous namespace
} // namespace mydialect
