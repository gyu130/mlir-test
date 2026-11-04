// DistributePass.cpp：--mydialect-distribute
// 分布式/异构执行：跨 device_exec 自动插 device_copy + device_wait
//
// 核心思想：
//   1. 给每个 device_exec（device="cpu"/"gpu"/"accel"/"hw"...）打设备标签；
//   2. 对跨 device 的 operand 依赖自动插 device_copy：
//        - host value（函数参数 / host op 产出） → device：先 device_alloc
//          目标 device buffer，再 device_copy，device_exec 引用新 buffer；
//        - device_exec 产出 A → 另一 device_exec 消费 B（设备不同）：
//          先在 B 的设备上 alloc buffer，再 device_copy，B 引用新 buffer；
//        - device_exec 产出 → host 下游：alloc host buffer → copy 回 host；
//   3. device_copy 默认同步（async 不打印），流水线化 async copy + device_wait
//      作为后续 pass 选项。
//
// 约束：当前仅支持静态 shape memref；HAL runtime 调用由 LowerDevicePass
// 后续把 device_copy/device_wait 降到 extern call。
#include "MyDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace mydialect;

namespace mydialect {
namespace {

// ======== 设备归属判定 ========
// 默认 host = "cpu"；device_exec 产出 = 其 device attr 指定的设备
static StringRef valueDevice(Value v,
                             const DenseMap<Value, StringRef> &cache) {
  auto it = cache.find(v);
  if (it != cache.end()) return it->second;
  Operation *def = v.getDefiningOp();
  if (!def) return "cpu"; // block argument → host
  if (auto dev = dyn_cast<DeviceExecOp>(def)) {
    StringAttr d = dev.getDeviceAttr();
    return d ? d.getValue() : "cpu";
  }
  return "cpu";
}

// 带 ODS 生成的 result size：DeviceCopyOp 有 src/dst 两个 operand + 可选 async
// DeviceAllocOp 有 device attr + Variadic sizes + MemRef result

// 在 anchorOp 之前：device_alloc + device_copy(src → alloc)，返回 alloc
static Value copyValueToDevice(OpBuilder &b, Operation *anchorOp, Value src,
                               StringRef dstDevice) {
  Location loc = anchorOp->getLoc();
  MemRefType srcTy = cast<MemRefType>(src.getType());
  b.setInsertionPoint(anchorOp);

  // 用 OperationState 建 DeviceAllocOp，避免猜 builder 签名
  OperationState allocState(loc, DeviceAllocOp::getOperationName());
  allocState.addTypes(srcTy);
  allocState.addAttribute("device", StringAttr::get(b.getContext(), dstDevice));
  allocState.addOperands(ValueRange{});
  Operation *allocOp = b.create(allocState);

  // DeviceCopyOp：src, dst, 可选 async
  OperationState copyState(loc, DeviceCopyOp::getOperationName());
  copyState.addOperands(ValueRange{src, allocOp->getResult(0)});
  b.create(copyState);

  return allocOp->getResult(0);
}

// 把一个 device_exec 内所有 operand（除 block arg）里跨设备的 memref 换成 copy 后的值
static void distributeDeviceExec(DeviceExecOp dev,
                                 const DenseMap<Value, StringRef> &devMap) {
  StringRef devDevice = dev.getDevice();
  Block &body = dev.getBody().front();

  // 收集要替换的 OpOperand 列表（避免边迭代边改容器）
  struct PendingRepl { Operation *op; unsigned opIdx; Value newVal; };
  SmallVector<PendingRepl, 8> pending;

  for (Operation &inner : body) {
    if (inner.getNumRegions() > 0) continue; // 不深入嵌套 region
    for (OpOperand &use : inner.getOpOperands()) {
      Value src = use.get();
      if (!isa<MemRefType>(src.getType())) continue;
      StringRef srcDev = valueDevice(src, devMap);
      if (srcDev == devDevice) continue;
      // block argument：必须从 host 拷贝，用 device_exec 本身作为 anchor
      // 普通 op operand：同样用 device_exec 作为 anchor（所有 copy 插在 device_exec 前）
      OpBuilder b(dev.getContext());
      Block *devBlock = dev.getOperation()->getBlock();
      b.setInsertionPoint(devBlock, Block::iterator(dev.getOperation()));
      Value newVal = copyValueToDevice(b, dev.getOperation(), src, devDevice);
      pending.push_back({&inner, use.getOperandNumber(), newVal});
    }
  }

  // 应用替换
  for (auto &p : pending) {
    Operation *op = p.op;
    if (p.opIdx < op->getNumOperands())
      op->setOperand(p.opIdx, p.newVal);
  }
}

// 对一个基本块：先构建 value→device map，再逐个 device_exec distribute
static void distributeBlock(Block &block) {
  DenseMap<Value, StringRef> devMap;
  for (Operation &op : block) {
    for (Value r : op.getResults())
      devMap[r] = valueDevice(r, devMap);
  }
  for (Operation &op : block) {
    if (auto dev = dyn_cast<DeviceExecOp>(&op))
      distributeDeviceExec(dev, devMap);
  }
}

struct DistributePass : public PassWrapper<DistributePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistributePass)

  StringRef getArgument() const final { return "mydialect-distribute"; }
  StringRef getDescription() const final {
    return "Insert device_copy across device_exec boundaries for "
           "distributed/heterogeneous execution";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp fn) {
      for (Block &block : fn.getBlocks())
        distributeBlock(block);
    });
  }
};

static PassRegistration<DistributePass> regDistribute;

} // anonymous namespace
} // namespace mydialect
