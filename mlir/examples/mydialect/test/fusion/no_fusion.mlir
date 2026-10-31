// RUN: %mlir-opt --mydialect-fuse-elementwise %s | %FileCheck %s
// 负例 1（multi_use）：mul 有两个使用者 → 不吸收进融合体，mul_tensor 保留；
//   但下游单使用的 add 链仍可彼此融合，且 mul 以叶子输入身份进入（body 无 muli）
// 负例 2（leaf_only）：两操作数都是函数参数的 add → 无融合收益，不做无意义包裹
module {
  func.func @multi_use(%a : !mydialect.tensor<[4],i32>,
                       %b : !mydialect.tensor<[4],i32>,
                       %c : !mydialect.tensor<[4],i32>) -> !mydialect.tensor<[4],i32> {
    %0 = mydialect.mul_tensor %a, %b : !mydialect.tensor<[4],i32>
    %1 = mydialect.add_tensor %0, %c : !mydialect.tensor<[4],i32>
    %2 = mydialect.add_tensor %0, %1 : !mydialect.tensor<[4],i32>
    return %2 : !mydialect.tensor<[4],i32>
  }

  func.func @leaf_only(%a : !mydialect.tensor<[4],i32>,
                       %b : !mydialect.tensor<[4],i32>) -> !mydialect.tensor<[4],i32> {
    %0 = mydialect.add_tensor %a, %b : !mydialect.tensor<[4],i32>
    return %0 : !mydialect.tensor<[4],i32>
  }
}

// CHECK-LABEL: func.func @multi_use
// CHECK: %[[M:.*]] = mydialect.mul_tensor
// CHECK: mydialect.fused_elementwise %[[M]], %arg2
// CHECK: arith.addi
// CHECK: arith.addi
// CHECK-NOT: arith.muli

// CHECK-LABEL: func.func @leaf_only
// CHECK: mydialect.add_tensor %arg0, %arg1 : <[4], i32>
// CHECK-NOT: mydialect.fused_elementwise
