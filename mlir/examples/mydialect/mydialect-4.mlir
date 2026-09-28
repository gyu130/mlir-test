module {
  func.func @main() -> i32 {
    %c7  = arith.constant 7 : i32
    %c6  = arith.constant 6 : i32
    %mul_res = mydialect.muli %c7, %c6 : i32
    mydialect.print %mul_res : i32

    %c10 = arith.constant 10 : i32
    %c2  = arith.constant 2 : i32
    %add_res = mydialect.addi %mul_res, %c2 : i32
    mydialect.print %add_res : i32

    return %c10 : i32
  }
}

