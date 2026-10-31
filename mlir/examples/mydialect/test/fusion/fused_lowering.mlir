// RUN: %mlir-opt --mydialect-fuse-elementwise --lower-mydialect-print %s | %FileCheck %s
// 融合 + lowering 端到端：(a*b)+c 只产生一个 scf.for 循环体，
// 循环体内 load 三个输入元素、muli+addi、store；中间结果不落地 memref
module {
  func.func @main(%a : !mydialect.tensor<[4],i32>,
                  %b : !mydialect.tensor<[4],i32>,
                  %c : !mydialect.tensor<[4],i32>) -> !mydialect.tensor<[4],i32> {
    %0 = mydialect.mul_tensor %a, %b : !mydialect.tensor<[4],i32>
    %1 = mydialect.add_tensor %0, %c : !mydialect.tensor<[4],i32>
    return %1 : !mydialect.tensor<[4],i32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: memref.store
// CHECK-NOT: scf.for
