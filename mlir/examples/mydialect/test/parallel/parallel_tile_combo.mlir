// RUN: %mlir-opt --lower-mydialect-print --mydialect-linalg-opt --convert-linalg-to-loops --mydialect-parallelize %s | %FileCheck %s
// 分块 + 并行化组合：linalg::tileLinalgOp 分块后外层循环（step = tile-size）
// 被提升为 scf.forall，内层分块循环留在并行块内部保持串行
module {
  func.func @main(%a : !mydialect.tensor<[64],i32>,
                  %b : !mydialect.tensor<[64],i32>) -> !mydialect.tensor<[64],i32> {
    %0 = mydialect.add_tensor %a, %b : <[64], i32>
    return %0 : !mydialect.tensor<[64], i32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: scf.forall (%[[IO:.*]]) = (%c0) to (%c64) step (%c32) {
// CHECK-NEXT: %[[SV0:.*]] = memref.subview %arg0[%[[IO]]] [32] [1]
// CHECK-NEXT: %[[SV1:.*]] = memref.subview %arg1[%[[IO]]] [32] [1]
// CHECK-NEXT: %[[SVO:.*]] = memref.subview %alloc[%[[IO]]] [32] [1]
// CHECK-NEXT: scf.for %[[II:.*]] = %c0 to %c32 step %c1 {
// CHECK-NEXT: %[[L0:.*]] = memref.load %[[SV0]][%[[II]]]
// CHECK-NEXT: %[[L1:.*]] = memref.load %[[SV1]][%[[II]]]
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[L0]], %[[L1]]
// CHECK-NEXT: memref.store %[[ADD]], %[[SVO]][%[[II]]]
