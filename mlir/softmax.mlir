module {
  func.func @t(%x : tensor<2x4xf32>) -> tensor<2x4xf32> {
    %y = "onnx.Softmax"(%x) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }
}
