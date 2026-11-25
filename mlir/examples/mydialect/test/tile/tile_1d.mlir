// RUN: %mlir-opt --lower-mydialect-print --mydialect-linalg-opt --convert-linalg-to-loops %s | %FileCheck %s
// 一维 elementwise 分块：--mydialect-linalg-opt 调用 linalg::tileLinalgOp
// 对 linalg.generic 的最外层循环分块（tile-size=32），再由
// --convert-linalg-to-loops 降回 scf.for。
module {
  func.func @main(%a : !mydialect.tensor<[64],i32>,
                  %b : !mydialect.tensor<[64],i32>) -> !mydialect.tensor<[64],i32> {
    %0 = mydialect.add_tensor %a, %b : <[64], i32>
    return %0 : !mydialect.tensor<[64], i32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: scf.for %[[IO:.*]] = %c0 to %c64 step %c32 {
// CHECK-NEXT: %[[SV0:.*]] = memref.subview %arg0[%[[IO]]] [32] [1]
// CHECK-NEXT: %[[SV1:.*]] = memref.subview %arg1[%[[IO]]] [32] [1]
// CHECK-NEXT: %[[SVO:.*]] = memref.subview %alloc[%[[IO]]] [32] [1]
// CHECK-NEXT: scf.for %[[II:.*]] = %c0 to %c32 step %c1 {
// CHECK-NEXT: %[[L0:.*]] = memref.load %[[SV0]][%[[II]]]
// CHECK-NEXT: %[[L1:.*]] = memref.load %[[SV1]][%[[II]]]
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[L0]], %[[L1]]
// CHECK-NEXT: memref.store %[[ADD]], %[[SVO]][%[[II]]]
// 原 linalg op 已被删除：只剩内层一次 store，无未分块的重复计算
// CHECK-NOT: memref.store
