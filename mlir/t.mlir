module {
  func.func @exp(%x : !mydialect.tensor<[2,3],f32>) -> !mydialect.tensor<[2,3],f32> {
    %r = mydialect.exp_tensor %x : !mydialect.tensor<[2,3],f32>
    return %r : !mydialect.tensor<[2,3],f32>
  }
}
