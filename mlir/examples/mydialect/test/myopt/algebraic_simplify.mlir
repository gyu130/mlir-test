// RUN: %mlir-opt --my-opt %s | %FileCheck %s
// 6.2 代数化简：identity 透传、双重 transpose 消去、链式 reshape 合并
module {
  // identity_tensor(x) → x
  func.func @identity(%a : !mydialect.tensor<[4,4],i32>) -> !mydialect.tensor<[4,4],i32> {
    %0 = mydialect.identity_tensor %a : <[4,4], i32>
    return %0 : !mydialect.tensor<[4,4], i32>
  }

  // transpose(transpose(x, [1,0]), [1,0]) → x（互逆置换）
  func.func @double_transpose(%a : !mydialect.tensor<[2,3],i32>) -> !mydialect.tensor<[2,3],i32> {
    %0 = mydialect.transpose_tensor %a, [1, 0] : <[2,3], i32> -> <[3,2], i32>
    %1 = mydialect.transpose_tensor %0, [1, 0] : <[3,2], i32> -> <[2,3], i32>
    return %1 : !mydialect.tensor<[2,3], i32>
  }

  // 非互逆置换（[1,2,0] 复合 [1,0,2] 非恒等）→ 保留
  func.func @non_inverse_transpose(%a : !mydialect.tensor<[2,3,4],i32>) -> !mydialect.tensor<[2,4,3],i32> {
    %0 = mydialect.transpose_tensor %a, [1, 0, 2] : <[2,3,4], i32> -> <[3,2,4], i32>
    %1 = mydialect.transpose_tensor %0, [1, 2, 0] : <[3,2,4], i32> -> <[2,4,3], i32>
    return %1 : !mydialect.tensor<[2,4,3], i32>
  }

  // reshape(reshape(x, s1), s2) → reshape(x, s2)
  func.func @chained_reshape(%x : !mydialect.tensor<[2,3],i32>,
                             %s1 : !mydialect.tensor<[1],i32>,
                             %s2 : !mydialect.tensor<[2],i32>) -> !mydialect.tensor<[2,3],i32> {
    %0 = mydialect.reshape_tensor %x, %s1 : <[2,3], i32>, <[1], i32> -> <[6], i32>
    %1 = mydialect.reshape_tensor %0, %s2 : <[6], i32>, <[2], i32> -> <[2,3], i32>
    return %1 : !mydialect.tensor<[2,3], i32>
  }
}

// CHECK-LABEL: func.func @identity
// CHECK-NEXT: return %arg0
// CHECK-NOT: mydialect.identity_tensor

// CHECK-LABEL: func.func @double_transpose
// CHECK-NEXT: return %arg0
// CHECK-NOT: mydialect.transpose_tensor

// CHECK-LABEL: func.func @non_inverse_transpose
// CHECK: mydialect.transpose_tensor %arg0, [1, 0, 2]
// CHECK: mydialect.transpose_tensor %{{.*}}, [1, 2, 0]

// CHECK-LABEL: func.func @chained_reshape
// CHECK-NEXT: %[[R:.*]] = mydialect.reshape_tensor %arg0, %arg2
// CHECK-NEXT: return %[[R]]
