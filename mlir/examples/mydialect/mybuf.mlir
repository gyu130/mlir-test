module {
func.func @main() -> i32 {
  %c10 = arith.constant 10 : i32
  // 栈分配大小=10的i32数组
  %buf = mydialect.alloc %c10 : memref<?xi32>

  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c42 = arith.constant 42 : i32

  // store %42 to buf[0]
  mydialect.store %c42, %buf, %c0 : i32, memref<?xi32>

  // load buf[0]
  %v = mydialect.load %buf, %c0 : memref<?xi32>, i32
  return %v : i32
}
}

