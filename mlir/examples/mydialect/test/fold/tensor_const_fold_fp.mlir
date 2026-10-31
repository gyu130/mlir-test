// RUN: %mlir-opt --constant-fold-mydialect %s | %FileCheck %s
// 6.1 浮点张量常量折叠：逐元素按 IEEE 规则求值（这里用 f32 的加/乘）
module {
  func.func @float_mul() -> !mydialect.tensor<[2],f32> {
    %a = mydialect.constant_tensor dense<[1.5, 2.0]> : tensor<2xf32> : <[2], f32>
    %b = mydialect.constant_tensor dense<[2.0, 3.0]> : tensor<2xf32> : <[2], f32>
    %0 = mydialect.mul_tensor %a, %b : <[2], f32>
    return %0 : !mydialect.tensor<[2], f32>
  }
}

// CHECK-LABEL: func.func @float_mul
// CHECK: mydialect.constant_tensor dense<[3.000000e+00, 6.000000e+00]> : tensor<2xf32>
// CHECK-NOT: mydialect.mul_tensor
