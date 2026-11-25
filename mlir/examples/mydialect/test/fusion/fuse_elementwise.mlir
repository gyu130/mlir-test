// RUN: %mlir-opt --mydialect-fuse-elementwise %s | %FileCheck %s
// 基本融合：mul 的结果只被 add 使用一次 → 融合为单个 fused_elementwise，
// 输入为 (%a, %b, %c)，body 内标量表达式为 (%a*%b)+%c
// 注意：assemblyFormat 中单值 type($res) 走 printStrippedAttrOrType，
// 结果类型打印为去掉 !mydialect.tensor 前缀的类型体 <[4], i32>
module {
  func.func @main(%a : !mydialect.tensor<[4],i32>,
                  %b : !mydialect.tensor<[4],i32>,
                  %c : !mydialect.tensor<[4],i32>) -> !mydialect.tensor<[4],i32> {
    %0 = mydialect.mul_tensor %a, %b : !mydialect.tensor<[4],i32>
    %1 = mydialect.add_tensor %0, %c : !mydialect.tensor<[4],i32>
    return %1 : !mydialect.tensor<[4],i32>
  }

  // 组内多用：%0(mul) 有两个使用者（%1、%2），但都在同一融合组内，
  // 吸收它不产生复制，body 内 mul 只计算一次并被两处 add 复用。
  func.func @multi_use_in_group(%a : !mydialect.tensor<[4],i32>,
                                %b : !mydialect.tensor<[4],i32>,
                                %c : !mydialect.tensor<[4],i32>) -> !mydialect.tensor<[4],i32> {
    %0 = mydialect.mul_tensor %a, %b : !mydialect.tensor<[4],i32>
    %1 = mydialect.add_tensor %0, %c : !mydialect.tensor<[4],i32>
    %2 = mydialect.add_tensor %0, %1 : !mydialect.tensor<[4],i32>
    return %2 : !mydialect.tensor<[4],i32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %[[F:.*]] = mydialect.fused_elementwise %arg0, %arg1, %arg2
// CHECK-SAME: -> <[4], i32> {
// CHECK: ^bb0(%[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32):
// CHECK: %[[M:.*]] = arith.muli %[[A]], %[[B]] : i32
// CHECK: %[[S:.*]] = arith.addi %[[M]], %[[C]] : i32
// CHECK: mydialect.fused_yield %[[S]] : i32
// CHECK: return %[[F]] : !mydialect.tensor<[4], i32>
// CHECK-NOT: mydialect.mul_tensor
// CHECK-NOT: mydialect.add_tensor

// 组内多用：mul 只计算一次，两个 add 复用同一 muli 结果
// CHECK-LABEL: func.func @multi_use_in_group
// CHECK: mydialect.fused_elementwise %arg0, %arg1, %arg2
// CHECK: %[[M:.*]] = arith.muli
// CHECK: arith.addi %[[M]],
// CHECK: arith.addi %[[M]],
// CHECK-NOT: mydialect.mul_tensor
// CHECK-NOT: mydialect.add_tensor
