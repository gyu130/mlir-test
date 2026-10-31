// RUN: %mlir-opt --mydialect-layout-propagation %s | %FileCheck %s
// Phase 6.3：相邻布局转换复合为单个。
//   p1 = [1,0,2]，p2 = [0,2,1]，复合 p1∘p2 = [1,2,0]
//   另有一条恒等置换转换（perm=[0,1,2]）应被直接消去。
module {
  func.func @compose(%x : !mydialect.tensor<[2, 3, 4], i32>) -> !mydialect.tensor<[3, 4, 2], i32> {
    %0 = mydialect.transpose_tensor %x, [1, 0, 2] : !mydialect.tensor<[2, 3, 4], i32> -> !mydialect.tensor<[3, 2, 4], i32>
    %1 = mydialect.transpose_tensor %0, [0, 2, 1] : !mydialect.tensor<[3, 2, 4], i32> -> !mydialect.tensor<[3, 4, 2], i32>
    return %1 : !mydialect.tensor<[3, 4, 2], i32>
  }

  func.func @identity_perm(%y : !mydialect.tensor<[2, 3], i32>) -> !mydialect.tensor<[2, 3], i32> {
    %0 = mydialect.transpose_tensor %y, [0, 1] : !mydialect.tensor<[2, 3], i32> -> !mydialect.tensor<[2, 3], i32>
    return %0 : !mydialect.tensor<[2, 3], i32>
  }
}

// CHECK-LABEL: func.func @compose
// CHECK-NEXT: %[[T:.*]] = mydialect.transpose_tensor %arg0, [1, 2, 0] : <[2,3,4], i32> -> <[3,4,2], i32>
// CHECK-NEXT: return %[[T]] : !mydialect.tensor<[3,4,2], i32>
// CHECK-LABEL: func.func @identity_perm
// CHECK-NEXT: return %arg0 : !mydialect.tensor<[2,3], i32>
