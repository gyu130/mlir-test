// SchedulePass.cpp：--mydialect-schedule
// 基本块级 list scheduling（指令调度）：
//
// 复用 OpCountCycleEstimate 的硬件模型（功能单元 FU 数量 + 各算子延迟），
// 在每个基本块内按依赖 DAG 做表调度，重排无副作用的算子以缩短关键路径：
//
//   - 可调度：纯算术算子（arith.addi/muli/divsi...）与 memref.load；
//   - 屏障：store / call / 带 region 的 op / 终止符等有副作用或结构复杂的 op，
//     其前后顺序保持不变，调度只在屏障之间分段进行；
//   - 模型：每类 FU（add / mul / mem）有 N 个流水线单元，
//     每个单元每拍可发射一条新指令；另受全局发射宽度 issue-width 限制；
//   - 优先级：关键路径长度（当前节点到块尾的最长延迟和），长的先发。
//
// 重排只改变同一段内互不依赖的算子顺序，SSA 依赖与屏障顺序均由 DAG 保证。
#include "MyOptPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

using namespace mlir;

namespace mydialect {
namespace {

// 功能单元类别；Barrier 表示不参与调度
enum class FU { Add, Mul, Mem, Free, Barrier };

static FU classifyOp(Operation *op) {
  // 内存读取可以参与调度（段内不会有 store，不会改变读序语义）
  if (isa<memref::LoadOp>(op))
    return FU::Mem;
  // 带 region 的 op、终止符、有副作用 op 都是屏障
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>() ||
      !isMemoryEffectFree(op))
    return FU::Barrier;
  if (isa<arith::ConstantOp>(op))
    return FU::Free;
  if (isa<arith::AddIOp, arith::SubIOp, arith::AndIOp, arith::OrIOp,
          arith::XOrIOp, arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp,
          arith::CmpIOp, arith::CmpFOp, arith::SelectOp, arith::ExtSIOp,
          arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp, arith::AddFOp,
          arith::SubFOp>(op))
    return FU::Add;
  if (isa<arith::MulIOp, arith::DivSIOp,
          arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::MulFOp,
          arith::DivFOp>(op))
    return FU::Mul;
  // 其余纯算子保守视为屏障，不重排
  return FU::Barrier;
}

struct Model {
  int64_t addLatency = 1;
  int64_t mulLatency = 4;
  int64_t memLatency = 3;
  int64_t addFU = 2;
  int64_t mulFU = 1;
  int64_t memFU = 1;
  int64_t issueWidth = 2;

  int64_t latencyOf(FU cls) const {
    switch (cls) {
    case FU::Add: return addLatency;
    case FU::Mul: return mulLatency;
    case FU::Mem: return memLatency;
    default: return 0;
    }
  }
};

struct Node {
  Operation *op = nullptr;
  FU cls = FU::Free;
  int64_t lat = 0;
  SmallVector<int, 4> preds;
  SmallVector<int, 4> succs;
  int64_t prio = 0;
};

// 段内表调度。byPriority=true 时按关键路径优先，否则按原程序顺序（基线）。
// order 输出调度后的发射顺序，返回总完成周期。
static int64_t listSchedule(SmallVectorImpl<Node> &nodes, const Model &m,
                            bool byPriority, SmallVectorImpl<int> &order) {
  int64_t n = static_cast<int64_t>(nodes.size());
  order.clear();
  if (n == 0)
    return 0;

  SmallVector<int64_t> fuAdd(std::max<int64_t>(m.addFU, 1), 0);
  SmallVector<int64_t> fuMul(std::max<int64_t>(m.mulFU, 1), 0);
  SmallVector<int64_t> fuMem(std::max<int64_t>(m.memFU, 1), 0);
  SmallVector<int64_t> finish(n, 0);
  SmallVector<bool> done(n, false);

  // 关键路径优先级：从后往前累加
  for (int64_t i = n - 1; i >= 0; --i) {
    int64_t best = 0;
    for (int s : nodes[i].succs)
      best = std::max(best, nodes[s].prio);
    nodes[i].prio = nodes[i].lat + best;
  }

  int64_t t = 0;
  int64_t remaining = n;
  // 少量保护：正常一定在 n + 所有延迟之和内结束
  int64_t guard = 0;

  // 判断节点 i 在 t 拍是否可发射（依赖满足 + 对应 FU 有空闲单元）
  auto issuable = [&](int64_t i) {
    for (int p : nodes[i].preds) {
      if (!done[p])
        return false;
      if (finish[p] > t)
        return false;
    }
    int64_t avail = t;
    switch (nodes[i].cls) {
    case FU::Add: avail = *std::min_element(fuAdd.begin(), fuAdd.end()); break;
    case FU::Mul: avail = *std::min_element(fuMul.begin(), fuMul.end()); break;
    case FU::Mem: avail = *std::min_element(fuMem.begin(), fuMem.end()); break;
    default: break; // Free：无资源限制
    }
    return avail <= t;
  };

  while (remaining > 0 && guard < 1000000) {
    int64_t issuedThisCycle = 0;
    while (issuedThisCycle < std::max<int64_t>(m.issueWidth, 1)) {
      int best = -1;
      if (byPriority) {
        for (int64_t i = 0; i < n; ++i) {
          if (done[i] || !issuable(i))
            continue;
          if (best < 0 || nodes[i].prio > nodes[best].prio ||
              (nodes[i].prio == nodes[best].prio && i < best))
            best = i;
        }
      } else {
        // 基线：严格按程序顺序发射（in-order 派发），
        // 最早未发射的算子若被阻塞则本拍空转，不越过它发射后面的算子
        for (int64_t i = 0; i < n; ++i) {
          if (done[i])
            continue;
          if (issuable(i))
            best = i;
          break;
        }
      }
      if (best < 0)
        break;
      // 占用最早空闲的 FU 单元
      switch (nodes[best].cls) {
      case FU::Add:
        *std::min_element(fuAdd.begin(), fuAdd.end()) = t + 1;
        break;
      case FU::Mul:
        *std::min_element(fuMul.begin(), fuMul.end()) = t + 1;
        break;
      case FU::Mem:
        *std::min_element(fuMem.begin(), fuMem.end()) = t + 1;
        break;
      default:
        break;
      }
      finish[best] = t + nodes[best].lat;
      done[best] = true;
      order.push_back(best);
      --remaining;
      ++issuedThisCycle;
    }
    if (issuedThisCycle == 0)
      ++t;
    ++guard;
  }

  int64_t total = 0;
  for (int64_t i = 0; i < n; ++i)
    total = std::max(total, finish[i]);
  return total;
}

// 对一个连续的无副作用段做表调度并按结果重排 IR
static void scheduleSegment(SmallVectorImpl<Operation *> &ops, const Model &m,
                            bool verbose, int64_t &baseline, int64_t &scheduled) {
  int64_t n = static_cast<int64_t>(ops.size());
  if (n <= 1)
    return;

  llvm::DenseMap<Operation *, int> idx;
  for (int64_t i = 0; i < n; ++i)
    idx[ops[i]] = static_cast<int>(i);

  SmallVector<Node> nodes(n);
  for (int64_t i = 0; i < n; ++i) {
    nodes[i].op = ops[i];
    nodes[i].cls = classifyOp(ops[i]);
    nodes[i].lat = m.latencyOf(nodes[i].cls);
  }
  // 依赖边：仅段内 def-use
  for (int64_t i = 0; i < n; ++i) {
    for (Value operand : ops[i]->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      auto it = idx.find(def);
      if (it == idx.end() || it->second >= i)
        continue;
      nodes[i].preds.push_back(it->second);
      nodes[it->second].succs.push_back(static_cast<int>(i));
    }
  }

  SmallVector<Node> copy = nodes;
  SmallVector<int> baselineOrder;
  baseline = listSchedule(copy, m, /*byPriority=*/false, baselineOrder);
  SmallVector<int> order;
  scheduled = listSchedule(nodes, m, /*byPriority=*/true, order);

  if (verbose)
    llvm::errs() << "[mydialect-schedule] block segment: " << n
                 << " ops, cycles " << baseline << " -> " << scheduled
                 << "\n";

  // 顺序未变则无需改动 IR
  bool changed = false;
  for (int64_t i = 0; i < n; ++i) {
    if (order[i] != i) {
      changed = true;
      break;
    }
  }
  if (!changed)
    return;

  // 把所有段内 op 依次移动到段的末尾（锚点为段后的屏障 op），
  // 由于段内 op 全部无副作用，移动不改变语义
  Operation *anchor = ops.back()->getNextNode();
  if (!anchor)
    anchor = ops.back()->getBlock()->getTerminator();
  for (int i : order)
    ops[i]->moveBefore(anchor);
}

static void scheduleBlock(Block &block, const Model &m, bool verbose,
                          int64_t &baseline, int64_t &scheduled) {
  SmallVector<Operation *> seg;
  auto flush = [&]() {
    scheduleSegment(seg, m, verbose, baseline, scheduled);
    seg.clear();
  };
  for (Operation &op : block) {
    if (classifyOp(&op) == FU::Barrier) {
      flush();
      continue;
    }
    seg.push_back(&op);
  }
  flush();
}

static void scheduleRegion(Region &region, const Model &m, bool verbose,
                           int64_t &baseline, int64_t &scheduled) {
  for (Block &block : region) {
    scheduleBlock(block, m, verbose, baseline, scheduled);
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        scheduleRegion(nested, m, verbose, baseline, scheduled);
  }
}

struct SchedulePass : public PassWrapper<SchedulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SchedulePass)
  SchedulePass() = default;
  SchedulePass(const SchedulePass &other) : PassWrapper(other) {}

  Option<int64_t> addLatency{*this, "add-latency",
                             llvm::cl::desc("Latency of add-class ops"),
                             llvm::cl::init(1)};
  Option<int64_t> mulLatency{*this, "mul-latency",
                             llvm::cl::desc("Latency of mul-class ops"),
                             llvm::cl::init(4)};
  Option<int64_t> memLatency{*this, "mem-latency",
                             llvm::cl::desc("Latency of memref.load"),
                             llvm::cl::init(3)};
  Option<int64_t> addFU{*this, "add-fu",
                        llvm::cl::desc("Number of add functional units"),
                        llvm::cl::init(2)};
  Option<int64_t> mulFU{*this, "mul-fu",
                        llvm::cl::desc("Number of mul functional units"),
                        llvm::cl::init(1)};
  Option<int64_t> memFU{*this, "mem-fu",
                        llvm::cl::desc("Number of load functional units"),
                        llvm::cl::init(1)};
  Option<int64_t> issueWidth{*this, "issue-width",
                             llvm::cl::desc("Max ops issued per cycle"),
                             llvm::cl::init(2)};
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Print per-block schedule statistics"),
                       llvm::cl::init(false)};

  StringRef getArgument() const final { return "mydialect-schedule"; }
  StringRef getDescription() const final {
    return "Block-level list scheduling with a FU/latency hardware model";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    Model m;
    m.addLatency = addLatency;
    m.mulLatency = mulLatency;
    m.memLatency = memLatency;
    m.addFU = addFU;
    m.mulFU = mulFU;
    m.memFU = memFU;
    m.issueWidth = issueWidth;

    int64_t baseline = 0, scheduled = 0;
    ModuleOp module = getOperation();
    for (Region &region : module->getRegions())
      scheduleRegion(region, m, verbose, baseline, scheduled);

    if (verbose)
      llvm::errs() << "[mydialect-schedule] total: cycles " << baseline
                   << " -> " << scheduled << "\n";
  }
};

static PassRegistration<SchedulePass> regSchedule;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createSchedulePass() {
  return std::make_unique<SchedulePass>();
}
} // namespace mydialect
