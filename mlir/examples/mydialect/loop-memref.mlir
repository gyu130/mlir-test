module {
func.func @main() -> i32 {
  %c10 = arith.constant 10 : i32
  %buf = mydialect.alloc %c10 : memref<?xi32>

  %c0 = arith.constant 0 : i32
  %c10_ub = arith.constant 10 : i32
  %c1_step = arith.constant 1 : i32

  mydialect.for %iv = %c0 to %c10_ub step %c1_step : i32 {
    mydialect.store %iv, %buf, %iv : i32, memref<?xi32>
  }

  %c5 = arith.constant 5 : i32
  %res = mydialect.load %buf, %c5 : memref<?xi32>, i32
  return %res : i32
}
}

