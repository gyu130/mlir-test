module {
  func.func @main() -> i32 {
    %c42 = arith.constant 42 : i32
    %c8  = arith.constant 8 : i32
    %sum = mydialect.addi %c42, %c8 : i32
    mydialect.print %sum : i32
    %c0  = arith.constant 0 : i32
    return %c0 : i32
  }
}

