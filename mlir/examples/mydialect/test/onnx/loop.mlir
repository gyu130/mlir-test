// RUN: %mlir-opt --convert-onnx-to-mydialect %s | %FileCheck %s

module {
  func.func @test_loop(%arg0: tensor<2x3xi32>, %init: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %m = onnx.Constant dense<10> : tensor<i64>
    %c = onnx.Constant dense<true> : tensor<i1>
    %0:1 = "onnx.Loop"(%m, %c, %init) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>, %carry: tensor<2x3xi32>):
      %1 = "onnx.Add"(%carry, %arg0) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
      "onnx.Yield"(%cond_in, %1) : (tensor<i1>, tensor<2x3xi32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<2x3xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}

// CHECK-LABEL: func.func @test_loop
// CHECK: mydialect.loop
// CHECK: mydialect.loop_yield
// CHECK-NOT: onnx.
