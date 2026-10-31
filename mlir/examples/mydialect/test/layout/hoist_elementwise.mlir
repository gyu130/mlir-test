// RUN: %mlir-opt --mydialect-layout-propagation %s | %FileCheck %s
// Phase 6.3：elementwise 输入端的两个同构布局转换被提升到结果之后，
// 再与结果端的逆转换复合为恒等，最终 4 个 transpose 全部消去，
// 只留下自然布局的 add_tensor。
module {
  func.func @hoist(%a : !mydialect.tensor<[2, 3], i32>, %b : !mydialect.tensor<[2, 3], i32>) -> !mydialect.tensor<[2, 3], i32> {
    %0 = mydialect.transpose_tensor %a, [1, 0] : !mydialect.tensor<[2, 3], i32> -> !mydialect.tensor<[3, 2], i32>
    %1 = mydialect.transpose_tensor %b, [1, 0] : !mydialect.tensor<[2, 3], i32> -> !mydialect.tensor<[3, 2], i32>
    %2 = mydialect.add_tensor %0, %1 : !mydialect.tensor<[3, 2], i32>
    %3 = mydialect.transpose_tensor %2, [1, 0] : !mydialect.tensor<[3, 2], i32> -> !mydialect.tensor<[2, 3], i32>
    return %3 : !mydialect.tensor<[2, 3], i32>
  }
}

// CHECK-LABEL: func.func @hoist
// CHECK-NOT: mydialect.transpose_tensor
// CHECK: %[[S:.*]] = mydialect.add_tensor %arg0, %arg1 : <[2,3], i32>
// CHECK: return %[[S]] : !mydialect.tensor<[2,3], i32>
