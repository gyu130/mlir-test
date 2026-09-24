module {
  func.func @main() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    mydialect.print %c42_i32 : i32
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
