// RUN: %mlir-opt --lower-mydialect-print %s | %FileCheck %s
//
// transpose_tensor: 嵌套 scf.for + 索引置换 + memref.load/store 转置拷贝

module {
  // 2D 简单转置：perm=[1,0]，输入 [2,3] → 输出 [3,2]
  func.func @transpose_2d(%a : !mydialect.tensor<[2,3],i32>)
      -> !mydialect.tensor<[3,2],i32> {
    %r = mydialect.transpose_tensor %a, [1, 0] : !mydialect.tensor<[2,3],i32> -> !mydialect.tensor<[3,2],i32>
    return %r : !mydialect.tensor<[3,2],i32>
  }

  // 3D 一般置换：perm=[2,0,1]，输入 [2,3,4] → 输出 [4,2,3]
  func.func @transpose_3d(%a : !mydialect.tensor<[2,3,4],i32>)
      -> !mydialect.tensor<[4,2,3],i32> {
    %r = mydialect.transpose_tensor %a, [2, 0, 1] : !mydialect.tensor<[2,3,4],i32> -> !mydialect.tensor<[4,2,3],i32>
    return %r : !mydialect.tensor<[4,2,3],i32>
  }

  // perm=[0] 恒等，rank=1 时退化为拷贝
  func.func @transpose_1d(%a : !mydialect.tensor<[4],i32>)
      -> !mydialect.tensor<[4],i32> {
    %r = mydialect.transpose_tensor %a, [0] : !mydialect.tensor<[4],i32> -> !mydialect.tensor<[4],i32>
    return %r : !mydialect.tensor<[4],i32>
  }
}

// ----- 2D -----
// CHECK-LABEL: func.func @transpose_2d
// CHECK-NOT: mydialect.transpose_tensor
// CHECK: memref.alloc() : memref<3x2xi32>
// CHECK: scf.for
// CHECK-NESTED: scf.for
// CHECK-NESTED-NOT: scf.for
// CHECK-NESTED-NESTED: memref.store
// CHECK-NESTED-NESTED: memref.load
// CHECK-NOT: memref.reinterpret_cast

// ----- 3D -----
// CHECK-LABEL: func.func @transpose_3d
// CHECK: memref.alloc() : memref<4x2x3xi32>
// CHECK: scf.for
// CHECK-NESTED: scf.for
// CHECK-NESTED-NESTED: scf.for
// CHECK-NESTED-NESTED-NOT: scf.for

// ----- 1D -----
// CHECK-LABEL: func.func @transpose_1d
// CHECK: memref.alloc() : memref<4xi32>
// CHECK: scf.for
// CHECK-NESTED-NOT: scf.for
