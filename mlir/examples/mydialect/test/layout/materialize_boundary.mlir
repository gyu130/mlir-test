// RUN: %mlir-opt --mydialect-layout-propagation %s | %FileCheck %s
// Phase 6.3：布局转换按需物化。print 不是 elementwise 消费端，布局无法继续
// 向下游传播，因此 add 的两个输入转换合并为一个，并在 print 前落地。
module {
  func.func @boundary(%a : !mydialect.tensor<[2, 3], i32>, %b : !mydialect.tensor<[2, 3], i32>) {
    %0 = mydialect.transpose_tensor %a, [1, 0] : !mydialect.tensor<[2, 3], i32> -> !mydialect.tensor<[3, 2], i32>
    %1 = mydialect.transpose_tensor %b, [1, 0] : !mydialect.tensor<[2, 3], i32> -> !mydialect.tensor<[3, 2], i32>
    %2 = mydialect.add_tensor %0, %1 : !mydialect.tensor<[3, 2], i32>
    mydialect.print %2 : !mydialect.tensor<[3, 2], i32>
    return
  }

  // 只有一个输入是布局转换（另一个已是目标布局的外部值），无法统一，
  // 转换保持原位；add 的转换被提升到 mul 之前，仍只有一次落地
  func.func @keep_one(%c : !mydialect.tensor<[2, 3], i32>, %d : !mydialect.tensor<[2, 3], i32>, %e : !mydialect.tensor<[3, 2], i32>) {
    %0 = mydialect.transpose_tensor %c, [1, 0] : !mydialect.tensor<[2, 3], i32> -> !mydialect.tensor<[3, 2], i32>
    %1 = mydialect.transpose_tensor %d, [1, 0] : !mydialect.tensor<[2, 3], i32> -> !mydialect.tensor<[3, 2], i32>
    %2 = mydialect.add_tensor %0, %1 : !mydialect.tensor<[3, 2], i32>
    %3 = mydialect.mul_tensor %2, %e : !mydialect.tensor<[3, 2], i32>
    mydialect.print %3 : !mydialect.tensor<[3, 2], i32>
    return
  }
}

// CHECK-LABEL: func.func @boundary
// CHECK-NEXT: %[[S:.*]] = mydialect.add_tensor %arg0, %arg1 : <[2,3], i32>
// CHECK-NEXT: %[[T:.*]] = mydialect.transpose_tensor %[[S]], [1, 0] : <[2,3], i32> -> <[3,2], i32>
// CHECK-NEXT: mydialect.print %[[T]] : !mydialect.tensor<[3,2], i32>
// CHECK-LABEL: func.func @keep_one
// CHECK-NEXT: %[[A:.*]] = mydialect.add_tensor %arg0, %arg1 : <[2,3], i32>
// CHECK-NEXT: %[[B:.*]] = mydialect.transpose_tensor %[[A]], [1, 0] : <[2,3], i32> -> <[3,2], i32>
// CHECK-NEXT: %[[M:.*]] = mydialect.mul_tensor %[[B]], %arg2 : <[3,2], i32>
// CHECK-NEXT: mydialect.print %[[M]] : !mydialect.tensor<[3,2], i32>
