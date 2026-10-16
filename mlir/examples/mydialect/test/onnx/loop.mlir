// RUN: %mlir-opt --convert-onnx-to-mydialect %s | %FileCheck %s

module {
  func.func @test_loop(%arg0: tensor<2x3xi32>, %init: tensor<2x3xi32>,
                       %m: i64, %c: i1) -> tensor<2x3xi32> {
    %0:1 = "onnx.Loop"(%m, %c, %init) ({
      %1 = "onnx.Add"(%arg0, %init) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
      "onnx.Yield"(%c, %1) : (i1, tensor<2x3xi32>) -> ()
    }) : (i64, i1, tensor<2x3xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}

// CHECK-LABEL: func.func @test_loop
// CHECK: mydialect.loop
// CHECK: mydialect.loop_yield
// CHECK-NOT: onnx.
