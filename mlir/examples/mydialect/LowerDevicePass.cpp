// LowerDevicePass.cpp：--mydialect-lower-device
// 异构部署的边界落地（与 --mydialect-place-device 配对）：
//   target=cpu    : 内联 mydialect.device_exec region，设备算子退回 host 执行
//                   （JIT 路径走这一档）
//   target=device : 保留 device_exec 边界，交给设备后端/kernel 生成
#include "MyOptPass.h"
#include "MyDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;

namespace mydialect {
namespace {

// 内联 device_exec：yield 操作数替换结果，region 内算子搬回父块
static void inlineDevice(DeviceExecOp op) {
  Block &body = op.getBody().front();
  auto yield = cast<DeviceYieldOp>(body.getTerminator());
  for (unsigned i = 0; i < op.getNumResults(); ++i)
    op.getResult(i).replaceAllUsesWith(yield.getOperand(i));
  yield->erase();
  op->getBlock()->getOperations().splice(Block::iterator(op),
                                         body.getOperations());
  op->erase();
}

static void lowerRegion(Region &region, bool inlineAll, unsigned &count) {
  for (Block &block : region) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (auto dev = dyn_cast<DeviceExecOp>(&op)) {
        if (inlineAll) {
          inlineDevice(dev);
          ++count;
        }
        continue; // 保留边界时不再深入（放置阶段不会嵌套 device_exec）
      }
      for (Region &nested : op.getRegions())
        lowerRegion(nested, inlineAll, count);
    }
  }
}

struct LowerDevicePass
    : public PassWrapper<LowerDevicePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDevicePass)
  LowerDevicePass() = default;
  LowerDevicePass(const LowerDevicePass &other) : PassWrapper(other) {}

  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("Deployment target: cpu=inline device_exec, "
                     "otherwise keep the host/device boundary"),
      llvm::cl::init("cpu")};
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Print lowering statistics"),
                       llvm::cl::init(false)};

  StringRef getArgument() const final { return "mydialect-lower-device"; }
  StringRef getDescription() const final {
    return "Lower mydialect.device_exec boundaries according to the "
           "deployment target (cpu inlines the region)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool inlineAll = target == "cpu";
    unsigned count = 0;
    for (Region &r : module->getRegions())
      lowerRegion(r, inlineAll, count);

    if (verbose)
      llvm::errs() << "[mydialect-lower-device] target=" << target
                   << ", inlined " << count << " device_exec ops\n";
  }
};

static PassRegistration<LowerDevicePass> regLowerDevice;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createLowerDevicePass(std::string target,
                                                  bool verbose) {
  auto pass = std::make_unique<LowerDevicePass>();
  pass->target = std::move(target);
  pass->verbose = verbose;
  return pass;
}
} // namespace mydialect
