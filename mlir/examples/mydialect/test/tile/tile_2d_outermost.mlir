// RUN: %mlir-opt --lower-mydialect-print --mydialect-linalg-opt --convert-linalg-to-loops %s | %FileCheck %s
// 多维 elementwise：linalg::tileLinalgOp 只对最外层循环分块，
// 其余维度 tile size 置 0（Linalg 约定 0 = 不分块），内层循环保持完整上界。
module {
  func.func @main(%a : !mydialect.tensor<[64,64],i32>,
                  %b : !mydialect.tensor<[64,64],i32>) -> !mydialect.tensor<[64,64],i32> {
    %0 = mydialect.add_tensor %a, %b : <[64,64], i32>
    return %0 : !mydialect.tensor<[64,64], i32>
  }
}

// CHECK-LABEL: func.func @main
// 最外层分块：步长 32，操作数换成 [32, 64] 的 subview
// CHECK: scf.for %[[IO:.*]] = %c0 to %c64 step %c32 {
// CHECK-NEXT: %[[SV0:.*]] = memref.subview %arg0[%[[IO]], 0] [32, 64] [1, 1]
// CHECK-NEXT: %[[SV1:.*]] = memref.subview %arg1[%[[IO]], 0] [32, 64] [1, 1]
// CHECK-NEXT: %[[SVO:.*]] = memref.subview %alloc[%[[IO]], 0] [32, 64] [1, 1]
// 第二维保持原样（0 到 64，步长 1）
// CHECK-NEXT: scf.for %[[II:.*]] = %c0 to %c32 step %c1 {
// CHECK-NEXT: scf.for %[[J:.*]] = %c0 to %c64 step %c1 {
// CHECK-NEXT: memref.load %[[SV0]][%[[II]], %[[J]]]
// CHECK-NEXT: memref.load %[[SV1]][%[[II]], %[[J]]]
// CHECK-NEXT: arith.addi
// CHECK-NEXT: memref.store
// 只有一层分块循环（未递归分块），且原 op 已删除
// CHECK-NOT: scf.for
// CHECK: return %alloc
