// RUN: %mlir-opt --lower-mydialect-print %s | %FileCheck %s
//
// reshape_tensor: 零拷贝 re-interpret_cast（共享底层 buffer）
// 降级后出现 memref.reinterpret_cast，无循环

module {
  // <[2,3],i32> → <[6],i32>：6 个元素，行主序
  func.func @reshape_2d_to_1d(%a : !mydialect.tensor<[2,3],i32>,
                              %shape : !mydialect.tensor<[1],i32>)
      -> !mydialect.tensor<[6],i32> {
    %r = mydialect.reshape_tensor %a, %shape : !mydialect.tensor<[2,3],i32>, !mydialect.tensor<[1],i32> -> !mydialect.tensor<[6],i32>
    return %r : !mydialect.tensor<[6],i32>
  }

  // <[6],i32> → <[2,3],i32>：反向
  func.func @reshape_1d_to_2d(%a : !mydialect.tensor<[6],i32>,
                              %shape : !mydialect.tensor<[2],i32>)
      -> !mydialect.tensor<[2,3],i32> {
    %r = mydialect.reshape_tensor %a, %shape : !mydialect.tensor<[6],i32>, !mydialect.tensor<[2],i32> -> !mydialect.tensor<[2,3],i32>
    return %r : !mydialect.tensor<[2,3],i32>
  }

  // 0 维 → 0 维（退化 case）
  func.func @reshape_0d(%a : !mydialect.tensor<[],i32>,
                        %shape : !mydialect.tensor<[0],i32>)
      -> !mydialect.tensor<[],i32> {
    %r = mydialect.reshape_tensor %a, %shape : !mydialect.tensor<[],i32>, !mydialect.tensor<[0],i32> -> !mydialect.tensor<[],i32>
    return %r : !mydialect.tensor<[],i32>
  }
}

// ----- 2D→1D -----
// CHECK-LABEL: func.func @reshape_2d_to_1d
// CHECK-NOT: mydialect.reshape_tensor
// CHECK-NOT: scf.for
// CHECK-NOT: memref.alloc
// CHECK: memref.reinterpret_cast
// CHECK: to memref<6xi32>

// ----- 1D→2D -----
// CHECK-LABEL: func.func @reshape_1d_to_2d
// CHECK: memref.reinterpret_cast
// CHECK: to memref<2x3xi32>

// ----- 0D -----
// CHECK-LABEL: func.func @reshape_0d
// CHECK: memref.reinterpret_cast
// CHECK: to memref<i32>
