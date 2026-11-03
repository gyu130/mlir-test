module {
  func.func @bad_alloc(%n : i32) {
    %b = mydialect.alloc %n : memref<2x3xi32>
    return
  }
}
