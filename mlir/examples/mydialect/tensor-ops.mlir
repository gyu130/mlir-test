module {
  func.func @tensor_compute(%a : !mydialect.tensor<[4,4],i32>, %b : !mydialect.tensor<[4,4],i32>) -> !mydialect.tensor<[4,4],i32> {
    %c = mydialect.add_tensor %a, %b : !mydialect.tensor<[4,4],i32>
    %d = mydialect.mul_tensor %c, %a : !mydialect.tensor<[4,4],i32>
    %cmp = mydialect.cmpi_tensor %d, %b, 0 : !mydialect.tensor<[4,4],i32>, !mydialect.tensor<[4,4],i1>
    return %d : !mydialect.tensor<[4,4],i32>
  }
}
