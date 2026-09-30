module {
func.func @test_zero() -> (i32,i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c_neg1 = arith.constant -1 : i32
  %a = arith.constant 100 : i32
  %b = arith.constant 200 : i32
  %res:2 = mydialect.for %iv = %c0 to %c_neg1 step %c1 iter_args(%v0=%a, %v1=%b) -> (i32,i32) : i32 {
    mydialect.yield %v0, %v1 : i32,i32
  }
  return %res#0, %res#1 : i32,i32
}
}

