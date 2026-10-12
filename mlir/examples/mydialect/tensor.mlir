module {
  func.func @test_tensor(%arg0 : !mydialect.tensor<[16,32], i32>) {
    mydialect.print %arg0 : !mydialect.tensor<[16,32], i32>
    return
  }
}
