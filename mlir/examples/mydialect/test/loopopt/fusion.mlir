// RUN: %mlir-opt --mydialect-loop-optimize="fusion=true interchange=false vectorize=false" %s | %FileCheck %s
//
// 相邻、同边界、无内存依赖的两个 scf.for 融合为一个；边界不同的保持分离。

func.func @fuse(%a: memref<8xf32>, %b: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %x = memref.alloc() : memref<8xf32>
  %y = memref.alloc() : memref<8xf32>
  scf.for %i = %c0 to %c8 step %c1 {
    %0 = memref.load %a[%i] : memref<8xf32>
    memref.store %0, %x[%i] : memref<8xf32>
  }
  scf.for %i = %c0 to %c8 step %c1 {
    %1 = memref.load %b[%i] : memref<8xf32>
    memref.store %1, %y[%i] : memref<8xf32>
  }
  return
}

func.func @no_fuse_diff_bounds(%a: memref<8xf32>, %b: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %x = memref.alloc() : memref<8xf32>
  %y = memref.alloc() : memref<8xf32>
  scf.for %i = %c0 to %c8 step %c1 {
    %0 = memref.load %a[%i] : memref<8xf32>
    memref.store %0, %x[%i] : memref<8xf32>
  }
  scf.for %i = %c0 to %c4 step %c1 {
    %1 = memref.load %b[%i] : memref<8xf32>
    memref.store %1, %y[%i] : memref<8xf32>
  }
  return
}

// CHECK-LABEL: func.func @fuse
// CHECK: scf.for %[[IV:[^ ]+]] = %c0 to %c8 step %c1 {
// CHECK-NEXT: %{{.*}} = memref.load %arg0[%[[IV]]]
// CHECK-NEXT: memref.store %{{.*}}, %alloc[%[[IV]]]
// CHECK-NEXT: %{{.*}} = memref.load %arg1[%[[IV]]]
// CHECK-NEXT: memref.store %{{.*}}, %alloc_0[%[[IV]]]
// CHECK-NEXT: }
// CHECK-NOT: scf.for

// CHECK-LABEL: func.func @no_fuse_diff_bounds
// CHECK: scf.for
// CHECK: scf.for
