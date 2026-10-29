
#ifndef MYDIALECT_OPCOUNTCYCLEESTIMATE_H
#define MYDIALECT_OPCOUNTCYCLEESTIMATE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mydialect {
/// 创建 OpCountCycleEstimatePass（TableGen 生成声明，见 MyDialectPasses.td）
/// 命令行用法：mlir-opt --mydialect-count-op-cycle-est
std::unique_ptr<mlir::Pass> createOpCountCycleEstimatePass();
} // namespace mydialect

#endif
