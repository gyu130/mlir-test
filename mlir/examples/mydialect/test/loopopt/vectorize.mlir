// RUN: %mlir-opt --mydialect-loop-optimize="fusion=false interchange=false vectorize=true" %s | %FileCheck %s
// RUN: %mlir-opt --mydialect-loop-optimize="fusion=false interchange=false vectorize=true vector-width=3" %s | %FileCheck %s --check-prefix=W3
//
// trip count 是 vector-width 整数倍的逐元素循环 → transfer_read/write + 向量算；
// 不是整数倍（8 % 3 != 0）→ 保持标量循环。

func.func @vectorize(%a: memref<8xf32>, %b: memref<8xf32>, %c: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %0 = memref.load %a[%i] : memref<8xf32>
    %1 = memref.load %b[%i] : memref<8xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %c[%i] : memref<8xf32>
  }
  return
}

// CHECK-LABEL: func.func @vectorize
// CHECK: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[STEP:.*]] = arith.constant 4 : index
// CHECK: scf.for %[[IV:[^ ]+]] = %c0 to %c8 step %[[STEP]] {
// CHECK-NEXT: %[[A:.*]] = vector.transfer_read %arg0[%[[IV]]], %[[PAD]] : memref<8xf32>, vector<4xf32>
// CHECK-NEXT: %[[B:.*]] = vector.transfer_read %arg1[%[[IV]]], %[[PAD]] : memref<8xf32>, vector<4xf32>
// CHECK-NEXT: %[[S:.*]] = arith.addf %[[A]], %[[B]] : vector<4xf32>
// CHECK-NEXT: vector.transfer_write %[[S]], %arg2[%[[IV]]] : vector<4xf32>, memref<8xf32>
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store

// W3-LABEL: func.func @vectorize
// W3-NOT: vector.transfer_read
// W3: scf.for
// W3: memref.load
// W3: memref.store
