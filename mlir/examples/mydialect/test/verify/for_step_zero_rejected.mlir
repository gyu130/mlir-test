// RUN: %mlir-opt %s 2>&1 | %FileCheck %s
//
// ForOp::verify: step 必须非零。
// step=0 时 scf.for 永不推进，是 UB，静态拒绝。

module {
  func.func @step_zero_const() -> i32 {
    %c0_lb   = arith.constant 0 : i32
    %c10_ub  = arith.constant 10 : i32
    %c0_step = arith.constant 0 : i32
    mydialect.for %iv = %c0_lb to %c10_ub step %c0_step : i32 {
    }
    return %c0_lb : i32
  }
}

// CHECK: error: 'mydialect.for' op step must not be zero
