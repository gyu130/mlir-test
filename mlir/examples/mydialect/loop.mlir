module {
  func.func @main() -> i32 {
    %zero = arith.constant 0 : i32
    %one  = arith.constant 1 : i32
    %ten  = arith.constant 10 : i32

    mydialect.for %zero, %ten, %one : {
      ^bb0(%i:i32):
      mydialect.print %i : i32
      mydialect.yield
    }

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
