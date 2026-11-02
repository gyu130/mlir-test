// RUN: %mlir-opt %s 2>&1 | %FileCheck %s
//
// 正常 step 应该通过 verifier，不报错。

module {
  // step=-1：合法反向循环
  func.func @step_neg_one() -> i32 {
    %c10   = arith.constant 10 : i32
    %c0    = arith.constant 0  : i32
    %c_neg1 = arith.constant -1 : i32
    mydialect.for %iv = %c10 to %c0 step %c_neg1 : i32 {
    }
    return %c0 : i32
  }

  // step=2：合法正向稀疏循环
  func.func @step_two() -> i32 {
    %c0  = arith.constant 0  : i32
    %c10 = arith.constant 10 : i32
    %c2  = arith.constant 2  : i32
    mydialect.for %iv = %c0 to %c10 step %c2 : i32 {
    }
    return %c0 : i32
  }

  // step 来自外部 SSA（动态）：允许，不做校验
  func.func @step_dynamic(%dyn_step : i32) -> i32 {
    %c0  = arith.constant 0  : i32
    %c10 = arith.constant 10 : i32
    mydialect.for %iv = %c0 to %c10 step %dyn_step : i32 {
    }
    return %c0 : i32
  }
}

// CHECK-NOT: error
// CHECK-LABEL: func.func @step_neg_one
// CHECK-LABEL: func.func @step_two
// CHECK-LABEL: func.func @step_dynamic
