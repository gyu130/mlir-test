module {
  func.func @bad_store(%buf : memref<2x3xi32>) {
    %c0 = arith.constant 0 : i32
    mydialect.store %c0, %buf, %c0 : i32, memref<2x3xi32>
    return
  }
}
