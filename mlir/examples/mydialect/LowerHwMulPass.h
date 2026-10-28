#ifndef MYDIALECT_LOWER_HWMUL_PASS_H
#define MYDIALECT_LOWER_HWMUL_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

// Pass 由 MyDialectPasses.td (TableGen) 声明并自动生成 createLowerHwMulPass()
// 及 registerLowerHwMulPass()，见 MyDialectPasses.h.inc。
// 命令行用法：mlir-opt --lower-hw-mul="use-hardware-path=true"

namespace mydialect {
// 手写工厂函数：显式指定 lowering 路径（不和 TableGen 生成的
// createLowerHwMulPass() 冲突——本函数要求 bool 参数，无默认值）。
//   useHardwarePath = false: hw_mul → arith.muli (SW / JIT)
//   useHardwarePath = true:  hw_mul → llvm.call_intrinsic "llvm.mydialect.hw.mul.i32"
std::unique_ptr<mlir::Pass> createLowerHwMulPass(bool useHardwarePath);
} // namespace mydialect

#endif
