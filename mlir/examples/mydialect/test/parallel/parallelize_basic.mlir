// RUN: %mlir-opt --lower-mydialect-print --mydialect-parallelize %s | %FileCheck %s
// Phase 3：无迭代携带、无外部调用的 elementwise 循环提升为 scf.forall
module {
  func.func @main(%a : !mydialect.tensor<[64],i32>,
                  %b : !mydialect.tensor<[64],i32>) -> !mydialect.tensor<[64],i32> {
    %0 = mydialect.add_tensor %a, %b : <[64], i32>
    return %0 : !mydialect.tensor<[64], i32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: scf.forall (%[[IV:.*]]) in (%c64) {
// CHECK-NEXT: %[[L0:.*]] = memref.load %arg0[%[[IV]]]
// CHECK-NEXT: %[[L1:.*]] = memref.load %arg1[%[[IV]]]
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[L0]], %[[L1]]
// CHECK-NEXT: memref.store %[[ADD]], %alloc[%[[IV]]]
// 原 scf.for 已被替换（scf.forall 前缀含 "scf.for"，需排除）
// CHECK-NOT: scf.for %
