module {
  func.func @bad_load(%buf : memref<2x3xi32>) {
    %c0 = arith.constant 0 : i32
    %v = mydialect.load %buf, %c0 : memref<2x3xi32>, i32
    return
  }
}
