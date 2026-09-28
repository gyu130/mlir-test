// test_fold.mlir
module {
  func.func @main() -> i32 {
    %c7  = arith.constant 7 : i32
    %c6  = arith.constant 6 : i32
    %mul = mydialect.muli %c7, %c6 : i32
    %c2  = arith.constant 2 : i32
    %add = mydialect.addi %mul, %c2 : i32
    mydialect.print %add : i32
    return %add : i32
  }
}

