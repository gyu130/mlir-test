// RUN: %mlir-opt --mydialect-schedule %s | %FileCheck %s
// 屏障语义：store 把基本块切成两段，调度不能把后面的 load 提到 store 之前
module {
  func.func @load_store(%m : memref<16xi32>, %o : memref<16xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = memref.load %m[%c0] : memref<16xi32>
    %b = arith.addi %a, %a : i32
    memref.store %b, %o[%c1] : memref<16xi32>
    %c = memref.load %m[%c1] : memref<16xi32>
    %d = arith.addi %c, %c : i32
    memref.store %d, %o[%c0] : memref<16xi32>
    return
  }
}

// CHECK-LABEL: func.func @load_store
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[L0:.*]] = memref.load %arg0[%[[C0]]]
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[A0:.*]] = arith.addi %[[L0]], %[[L0]] : i32
// CHECK-NEXT: memref.store %[[A0]], %arg1[%[[C1]]]
// CHECK-NEXT: %[[L1:.*]] = memref.load %arg0[%[[C1]]]
// CHECK-NEXT: %[[A1:.*]] = arith.addi %[[L1]], %[[L1]] : i32
// CHECK-NEXT: memref.store %[[A1]], %arg1[%[[C0]]]
// 两次 store 的先后顺序保持不变
// CHECK-NOT: memref.store
