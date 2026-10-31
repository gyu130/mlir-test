// RUN: %mlir-opt --lower-mydialect-print --mydialect-tile %s | %FileCheck %s
// Phase 2：多维 elementwise nest 只分块最外层循环，
// 内层（第二维）保持原样不分块（避免递归分块）
module {
  func.func @main(%a : !mydialect.tensor<[64,64],i32>,
                  %b : !mydialect.tensor<[64,64],i32>) -> !mydialect.tensor<[64,64],i32> {
    %0 = mydialect.add_tensor %a, %b : <[64,64], i32>
    return %0 : !mydialect.tensor<[64,64], i32>
  }
}

// CHECK-LABEL: func.func @main
// 最外层被 strip-mining（step 32 + affine.min）
// CHECK: scf.for %[[IO:.*]] = %c0 to %c64 step %c32 {
// CHECK-NEXT: %[[UB:.*]] = affine.min #map(%[[IO]])[%c64]
// CHECK-NEXT: scf.for %[[II:.*]] = %c0 to %[[UB]] step %c1 {
// CHECK-NEXT: %[[I:.*]] = arith.addi %[[IO]], %[[II]] : index
// 第二维循环保持原状（step 1 到 64，无 affine.min 上界）
// CHECK-NEXT: scf.for %[[J:.*]] = %c0 to %c64 step %c1 {
// CHECK-NEXT: memref.load %arg0[%[[I]], %[[J]]]
// CHECK-NEXT: memref.load %arg1[%[[I]], %[[J]]]
// CHECK-NEXT: arith.addi
// CHECK-NEXT: memref.store
// 只有一层 affine.min（未递归分块）
// CHECK-NOT: affine.min
// CHECK: return %alloc
