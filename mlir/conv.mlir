module {
  func.func @t(%x : tensor<1x3x4x4xf32>, %w : tensor<5x3x2x2xf32>) -> tensor<1x5x3x3xf32> {
    %y = "onnx.Conv"(%x, %w) {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]} : (tensor<1x3x4x4xf32>, tensor<5x3x2x2xf32>) -> tensor<1x5x3x3xf32>
    return %y : tensor<1x5x3x3xf32>
  }
}
