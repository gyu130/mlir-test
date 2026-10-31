// RUN: %mlir-opt %s | %FileCheck %s
// fused_elementwise 汇编格式 roundtrip：手写 IR → parse → print 稳定。
// 变参 type($inputs) 打印完整 !mydialect.tensor<...>，单值 type($res)
// 打印类型体 <[4], i32>，两种形式 parse 都要接受。
module {
  func.func @rt(%a : !mydialect.tensor<[4],i32>,
                %b : !mydialect.tensor<[4],i32>) -> !mydialect.tensor<[4],i32> {
    %0 = mydialect.fused_elementwise %a, %b : !mydialect.tensor<[4],i32>, !mydialect.tensor<[4],i32> -> <[4],i32> {
    ^bb0(%x: i32, %y: i32):
      %1 = arith.muli %x, %y : i32
      mydialect.fused_yield %1 : i32
    }
    return %0 : !mydialect.tensor<[4],i32>
  }
}

// CHECK-LABEL: func.func @rt
// CHECK: mydialect.fused_elementwise %arg0, %arg1
// CHECK-SAME: -> <[4], i32> {
// CHECK: ^bb0(%[[X:.*]]: i32, %[[Y:.*]]: i32):
// CHECK: %[[M:.*]] = arith.muli %[[X]], %[[Y]] : i32
// CHECK: mydialect.fused_yield %[[M]] : i32
