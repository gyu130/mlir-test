// RUN: %mlir-opt --lower-mydialect-print --mydialect-tile %s | %FileCheck %s
// Phase 2.2：一维 elementwise 循环 strip-mining
// 外层步长 = tile-size(32)，内层上界 = affine.min，索引 = io + ii
module {
  func.func @main(%a : !mydialect.tensor<[64],i32>,
                  %b : !mydialect.tensor<[64],i32>) -> !mydialect.tensor<[64],i32> {
    %0 = mydialect.add_tensor %a, %b : <[64], i32>
    return %0 : !mydialect.tensor<[64], i32>
  }
}

// CHECK: #map = affine_map<(d0)[s0] -> (32, -d0 + s0)>
// CHECK-LABEL: func.func @main
// CHECK: scf.for %[[IO:.*]] = %c0 to %c64 step %c32 {
// CHECK-NEXT: %[[UB:.*]] = affine.min #map(%[[IO]])[%c64]
// CHECK-NEXT: scf.for %[[II:.*]] = %c0 to %[[UB]] step %c1 {
// CHECK-NEXT: %[[IDX:.*]] = arith.addi %[[IO]], %[[II]] : index
// CHECK-NEXT: %[[L0:.*]] = memref.load %arg0[%[[IDX]]]
// CHECK-NEXT: %[[L1:.*]] = memref.load %arg1[%[[IDX]]]
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[L0]], %[[L1]]
// CHECK-NEXT: memref.store %[[ADD]], %alloc[%[[IDX]]]
// 分块后只有两个循环、一次 store（原循环被搬进内层）
// CHECK-NOT: memref.store
