module {
  func.func @alloc_nd(%n : i32, %m : i32) {
    %c0 = arith.constant 0 : i32
    %buf2 = mydialect.alloc %n, %m : memref<?x?xi32>
    mydialect.store %c0, %buf2, %c0, %c0 : i32, memref<?x?xi32>
    %v = mydialect.load %buf2, %c0, %c0 : memref<?x?xi32>, i32
    %buf3 = mydialect.alloc %n : memref<?x4xi32>
    mydialect.store %v, %buf3, %c0, %c0 : i32, memref<?x4xi32>
    %w = mydialect.load %buf3, %c0, %c0 : memref<?x4xi32>, i32
    %static = mydialect.alloc : memref<2x3xi32>
    mydialect.store %w, %static, %c0, %c0 : i32, memref<2x3xi32>
    return
  }
}
