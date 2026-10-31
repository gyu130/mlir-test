// PlaceDevicePass.cpp：--mydialect-place-device
// 异构部署的 device placement：
//   1. 以 mydialect.hw_mul（硬件加速原语）为种子，沿纯 mydialect 算子的
//      def-use 边做双向闭包，得到「设备子图」；
//   2. 把同一基本块内连续的设备算子包进 mydialect.device_exec region，
//      region 内产生、region 外仍要用的值通过 device_yield 提升为结果，
//      形成 host ↔ device 的流水线边界。
//
// 下游 --mydialect-lower-device=target 决定边界如何落地：
//   cpu    : 内联 region（设备算子退回 host 执行，JIT 可直接跑）
//   device : 保留边界，交给设备后端/kernel 生成
#include "MyOptPass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;

namespace mydialect {
namespace {

// 可作为设备算子的候选：mydialect 纯算子（无副作用、无 region、非终止符）
// print 系列虽然不带副作用 trait，但会调用外部运行时，必须留在 host
static bool isPureMyDialectOp(Operation *op) {
  StringRef ns = op->getName().getDialectNamespace();
  if (ns != "mydialect")
    return false;
  if (isa<PrintOp, PrintStrOp>(op))
    return false;
  return isMemoryEffectFree(op) && op->getNumRegions() == 0 &&
         !op->hasTrait<OpTrait::IsTerminator>();
}

// 以 hw_mul 为种子，沿纯算子 def-use 边做双向闭包（仅同一基本块内）
static void collectDeviceOps(Block &block, llvm::DenseSet<Operation *> &dev) {
  llvm::SmallVector<Operation *, 16> worklist;
  for (Operation &op : block)
    if (isa<HwMulOp>(op))
      worklist.push_back(&op);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!dev.insert(op).second)
      continue;
    for (Value v : op->getOperands()) {
      Operation *def = v.getDefiningOp();
      if (def && def->getBlock() == &block && isPureMyDialectOp(def) &&
          !dev.count(def))
        worklist.push_back(def);
    }
    for (Value r : op->getResults()) {
      for (Operation *user : r.getUsers()) {
        if (user->getBlock() == &block && isPureMyDialectOp(user) &&
            !dev.count(user))
          worklist.push_back(user);
      }
    }
  }
}

// 把一个连续设备算子段 [lo, hi) 包进 device_exec
static void wrapRun(Block &block, ArrayRef<Operation *> ops, size_t lo,
                    size_t hi, StringAttr deviceAttr) {
  Operation *first = ops[lo];
  Operation *after = ops[hi - 1]->getNextNode();

  llvm::SmallPtrSet<Operation *, 8> runSet;
  for (size_t i = lo; i < hi; ++i)
    runSet.insert(ops[i]);

  // 收集 region 内定义、region 外使用的值（去重）
  SmallVector<Value> outs;
  for (size_t i = lo; i < hi; ++i) {
    for (Value r : ops[i]->getResults()) {
      for (OpOperand &use : r.getUses()) {
        if (!runSet.count(use.getOwner())) {
          outs.push_back(r);
          break;
        }
      }
    }
  }

  SmallVector<Type> outTypes;
  for (Value v : outs)
    outTypes.push_back(v.getType());

  // 用 OperationState 显式建 region，避免依赖 ODS 生成 builder 的 region 语义。
  // 注意 Operation::create 会把 state 里的 region 通过 takeBody 搬进新 op，
  // 因此必须用 op 自己的 region（state.addRegion() 返回的 region 随后即被掏空）
  OpBuilder b(first);
  OperationState state(first->getLoc(), DeviceExecOp::getOperationName());
  state.addTypes(outTypes);
  state.addAttribute("device", deviceAttr);
  state.addRegion();
  DeviceExecOp devOp = cast<DeviceExecOp>(b.create(state));

  Block *body = new Block();
  devOp.getBody().push_back(body);

  // 把整段算子搬进 region（region 可直接引用外层值，无需块参数）
  // 注意 splice 的接收者必须是目标 list（body），否则 Operation 的 block
  // 指针不会被 transferNodesFromList 更新，会留下悬空的 block 归属
  body->getOperations().splice(body->end(), block.getOperations(),
                               Block::iterator(first),
                               after ? Block::iterator(after) : block.end());

  OpBuilder atEnd = OpBuilder::atBlockEnd(body);
  DeviceYieldOp::create(atEnd, first->getLoc(), outs);

  // region 外的使用改指向 device_exec 的结果
  for (size_t k = 0; k < outs.size(); ++k) {
    outs[k].replaceUsesWithIf(devOp.getResult(k), [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      return owner->getParentRegion() != &devOp.getBody();
    });
  }
}

static void wrapDeviceRuns(Block &block, llvm::DenseSet<Operation *> &dev,
                           StringAttr deviceAttr) {
  SmallVector<Operation *> ops;
  for (Operation &op : block)
    ops.push_back(&op);
  size_t i = 0;
  while (i < ops.size()) {
    if (!dev.count(ops[i])) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < ops.size() && dev.count(ops[j]))
      ++j;
    wrapRun(block, ops, i, j, deviceAttr);
    i = j;
  }
}

static void placeRegion(Region &region, StringAttr deviceAttr,
                        llvm::DenseSet<Operation *> &allDev, bool verbose) {
  for (Block &block : region) {
    llvm::DenseSet<Operation *> dev;
    collectDeviceOps(block, dev);
    if (!dev.empty()) {
      wrapDeviceRuns(block, dev, deviceAttr);
      allDev.insert(dev.begin(), dev.end());
    }
    for (Operation &op : block) {
      // 已经包好的 device_exec 不再深入，避免把设备算子重复嵌套
      if (isa<DeviceExecOp>(op))
        continue;
      for (Region &nested : op.getRegions())
        placeRegion(nested, deviceAttr, allDev, verbose);
    }
  }
}

struct PlaceDevicePass
    : public PassWrapper<PlaceDevicePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlaceDevicePass)
  PlaceDevicePass() = default;
  PlaceDevicePass(const PlaceDevicePass &other) : PassWrapper(other) {}

  Option<std::string> deviceName{*this, "device-name",
                                 llvm::cl::desc("Target device name label"),
                                 llvm::cl::init("hw")};
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Print placement statistics"),
                       llvm::cl::init(false)};

  StringRef getArgument() const final { return "mydialect-place-device"; }
  StringRef getDescription() const final {
    return "Place hw_mul closure ops on the target device and wrap them in "
           "mydialect.device_exec regions";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    StringAttr attr =
        StringAttr::get(&getContext(), deviceName.getValue());
    llvm::DenseSet<Operation *> allDev;
    for (Region &r : module->getRegions())
      placeRegion(r, attr, allDev, verbose);

    if (verbose)
      llvm::errs() << "[mydialect-place-device] placed " << allDev.size()
                   << " ops on device '" << deviceName << "'\n";
  }
};

static PassRegistration<PlaceDevicePass> regPlaceDevice;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createPlaceDevicePass() {
  return std::make_unique<PlaceDevicePass>();
}
} // namespace mydialect
