// RUN: %mlir-opt --constant-fold-mydialect %s | %FileCheck %s
// 6.1 张量级常量折叠：add/mul 双常量输入逐元素求值后折叠为 constant_tensor
module {
  // 整数链式折叠：([1,2,3]+[4,5,6])*[2,3,4] = [10,21,36]
  func.func @int_chain() -> !mydialect.tensor<[3],i32> {
    %a = mydialect.constant_tensor dense<[1, 2, 3]> : tensor<3xi32> : <[3], i32>
    %b = mydialect.constant_tensor dense<[4, 5, 6]> : tensor<3xi32> : <[3], i32>
    %c = mydialect.constant_tensor dense<[2, 3, 4]> : tensor<3xi32> : <[3], i32>
    %0 = mydialect.add_tensor %a, %b : <[3], i32>
    %1 = mydialect.mul_tensor %0, %c : <[3], i32>
    return %1 : !mydialect.tensor<[3], i32>
  }

  // 0 维标量常量：7*6 = 42
  func.func @scalar_mul() -> !mydialect.tensor<[],i32> {
    %a = mydialect.constant_tensor dense<7> : tensor<i32> : <[], i32>
    %b = mydialect.constant_tensor dense<6> : tensor<i32> : <[], i32>
    %0 = mydialect.mul_tensor %a, %b : <[], i32>
    return %0 : !mydialect.tensor<[], i32>
  }

  // 非全常量：保留原 op
  func.func @non_const(%arg0 : !mydialect.tensor<[3],i32>,
                       %arg1 : !mydialect.tensor<[3],i32>) -> !mydialect.tensor<[3],i32> {
    %a = mydialect.constant_tensor dense<[1, 2, 3]> : tensor<3xi32> : <[3], i32>
    %0 = mydialect.add_tensor %arg0, %arg1 : <[3], i32>
    %1 = mydialect.mul_tensor %0, %a : <[3], i32>
    return %1 : !mydialect.tensor<[3], i32>
  }
}

// CHECK-LABEL: func.func @int_chain
// CHECK-NEXT: %[[C:.*]] = mydialect.constant_tensor dense<[10, 21, 36]> : tensor<3xi32>
// CHECK-NEXT: return %[[C]]
// CHECK-NOT: mydialect.add_tensor
// CHECK-NOT: mydialect.mul_tensor

// CHECK-LABEL: func.func @scalar_mul
// CHECK-NEXT: %[[S:.*]] = mydialect.constant_tensor dense<42> : tensor<i32>
// CHECK-NEXT: return %[[S]]

// CHECK-LABEL: func.func @non_const
// CHECK: mydialect.add_tensor %arg0, %arg1
// CHECK: mydialect.mul_tensor
