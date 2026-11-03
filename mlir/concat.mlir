module {
  func.func @t(%a : tensor<2x3xf32>, %b : tensor<2x3xf32>) -> tensor<2x6xf32> {
    %y = "onnx.Concat"(%a, %b) {axis = 1 : i64} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
    return %y : tensor<2x6xf32>
  }
}
