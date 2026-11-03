module {
  func.func @relu(%x : !mydialect.tensor<[2,3],f32>) -> !mydialect.tensor<[2,3],f32> {
    %r = mydialect.relu_tensor %x : !mydialect.tensor<[2,3],f32>
    return %r : !mydialect.tensor<[2,3],f32>
  }
}
