#include "OpCountCycleEstimate.h"
#include "MyDialect.h"
#include "MyDialectOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace mydialect {

// TableGen 声明/定义：生成 impl::OpCountCycleEstimatePassBase 及工厂函数
#define GEN_PASS_DECL_OPCOUNTCYCLEESTIMATEPASS
#define GEN_PASS_DEF_OPCOUNTCYCLEESTIMATEPASS
#include "MyDialectPasses.h.inc"

struct OpCountCycleEstimatePass
    : public impl::OpCountCycleEstimatePassBase<OpCountCycleEstimatePass> {
  using Base::Base;
  using Base::addLatency;
  using Base::mulLatency;
  using Base::addFU;
  using Base::mulFU;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // arith.addi/muli 也要统计：hw_mul 在流水线里先被 lower 成 arith.muli，
    // 只看 mydialect.muli 会漏掉硬件乘法
    int64_t addCount = 0;
    int64_t mulCount = 0;
    module.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      if (opName == "mydialect.addi" || opName == "arith.addi") {
        ++addCount;
      } else if (opName == "mydialect.muli" || opName == "arith.muli" ||
                 opName == "mydialect.hw_mul") {
        ++mulCount;
      }
    });

    // 硬件资源模型：fu 个流水线功能单元，latency 拍延迟。
    // N 个同类操作占用 ceil(N / fu) + latency - 1 拍；
    // 加法/乘法两条流水线并行，总周期取两者最大值（理想无数据相关）。
    auto pipelineCycles = [](int64_t n, int fu, int latency) -> int64_t {
      if (n <= 0)
        return 0;
      if (fu <= 0)
        fu = 1;
      return (n + fu - 1) / fu + latency - 1;
    };

    int64_t addCycle = pipelineCycles(addCount, addFU, addLatency);
    int64_t mulCycle = pipelineCycles(mulCount, mulFU, mulLatency);
    int64_t totalCycle = std::max(addCycle, mulCycle);

    // 报告走 stderr，与驱动里的 IR 输出（stdout）分离
    llvm::errs() << "\n===== Op Count & Cycle Estimation Report =====\n";
    llvm::errs() << "mydialect.addi count : " << addCount << "\n";
    llvm::errs() << "mydialect.muli count : " << mulCount << "\n";
    llvm::errs() << "HW FU config: add_fu=" << addFU << ", mul_fu=" << mulFU
                 << "\n";
    llvm::errs() << "addi latency=" << addLatency
                 << ", muli latency=" << mulLatency << "\n";
    llvm::errs() << "add pipeline cycle : " << addCycle << "\n";
    llvm::errs() << "mul pipeline cycle : " << mulCycle << "\n";
    llvm::errs() << "Estimated total cycle (ideal no dependency): "
                 << totalCycle << "\n";
    llvm::errs() << "==============================================\n\n";
  }
};

// Pass 注册：与 LowerMyDialectPrint 一致，在自身 TU 内静态注册，
// 这样 mlir-opt 只要编译本文件即可使用 --mydialect-count-op-cycle-est，
// 无需依赖 MyDialect.cpp 的 GEN_PASS_REGISTRATION 汇总入口
#define GEN_PASS_REGISTRATION_OPCOUNTCYCLEESTIMATEPASS
#include "MyDialectPasses.h.inc"

static bool opCountCycleEstimatePassRegistered = []() {
  registerOpCountCycleEstimatePass();
  return true;
}();

} // namespace mydialect
