module {
  func.func @main() -> i32 {
    %c42 = arith.constant 42 : i32
    mydialect.print %c42 : i32
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
