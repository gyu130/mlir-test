// RUN: %mlir-opt --mydialect-loop-optimize="fusion=false interchange=true vectorize=false" %s | %FileCheck %s
//
// 交换前外层 IV 出现在 memref 最内维索引（列主序访问）；交换后最内维索引
// 由最内层 IV 驱动（行主序访问），改善访存局部性。

func.func @interchange() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %a = memref.alloc() : memref<4x4xf32>
  %b = memref.alloc() : memref<4x4xf32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %0 = memref.load %a[%j, %i] : memref<4x4xf32>
      memref.store %0, %b[%j, %i] : memref<4x4xf32>
    }
  }
  return
}

// 已是最内维由内层 IV 驱动（行主序），不应交换。
func.func @no_interchange() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %a = memref.alloc() : memref<4x4xf32>
  %b = memref.alloc() : memref<4x4xf32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %0 = memref.load %a[%i, %j] : memref<4x4xf32>
      memref.store %0, %b[%i, %j] : memref<4x4xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @interchange
// CHECK: scf.for %[[O:[^ ]+]] = %c0 to %c4 step %c1 {
// CHECK-NEXT: scf.for %[[I:[^ ]+]] = %c0 to %c4 step %c1 {
// CHECK-NEXT: %{{.*}} = memref.load %{{.*}}[%[[O]], %[[I]]]
// CHECK-NEXT: memref.store %{{.*}}, %{{.*}}[%[[O]], %[[I]]]

// CHECK-LABEL: func.func @no_interchange
// CHECK: scf.for %[[O2:[^ ]+]] = %c0 to %c4 step %c1 {
// CHECK-NEXT: scf.for %[[I2:[^ ]+]] = %c0 to %c4 step %c1 {
// CHECK-NEXT: %{{.*}} = memref.load %{{.*}}[%[[O2]], %[[I2]]]
// CHECK-NEXT: memref.store %{{.*}}, %{{.*}}[%[[O2]], %[[I2]]]
