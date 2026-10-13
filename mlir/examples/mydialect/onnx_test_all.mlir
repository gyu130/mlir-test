module {
  func.func @test_add(%arg0 : tensor<2x3xi32>, %arg1 : tensor<2x3xi32>) -> tensor<2x3xi32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  func.func @test_mul(%arg0 : tensor<2x3xi32>, %arg1 : tensor<2x3xi32>) -> tensor<2x3xi32> {
    %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  func.func @test_equal(%arg0 : tensor<2x3xi32>, %arg1 : tensor<2x3xi32>) -> tensor<2x3xi1> {
    %0 = "onnx.Equal"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    return %0 : tensor<2x3xi1>
  }

  func.func @test_identity(%arg0 : tensor<2x3xi32>) -> tensor<2x3xi32> {
    %0 = "onnx.Identity"(%arg0) : (tensor<2x3xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  func.func @test_constant() -> tensor<2x3xi32> {
    %0 = "onnx.Constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  func.func @test_reshape(%arg0 : tensor<6xi32>) -> tensor<2x3xi32> {
    %shape = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
    %0 = "onnx.Reshape"(%arg0, %shape) : (tensor<6xi32>, tensor<2xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  func.func @test_transpose(%arg0 : tensor<2x3xi32>) -> tensor<3x2xi32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3xi32>) -> tensor<3x2xi32>
    return %0 : tensor<3x2xi32>
  }

  func.func @test_reduce_sum(%arg0 : tensor<2x3xi32>) -> tensor<2xi32> {
    %0 = "onnx.ReduceSum"(%arg0) {axes = [1], keepdims = false} : (tensor<2x3xi32>) -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }

  func.func @test_reduce_sum_keep(%arg0 : tensor<2x3xi32>) -> tensor<2x1xi32> {
    %0 = "onnx.ReduceSum"(%arg0) {axes = [1], keepdims = true} : (tensor<2x3xi32>) -> tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
  }
}
