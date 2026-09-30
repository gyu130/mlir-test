module {
  func.func @main() -> i32 {
    %zero = arith.constant 0 : i32
    %one  = arith.constant 1 : i32
    %ten  = arith.constant 10 : i32
    %init_sum  = arith.constant 0 : i32
    %init_prod = arith.constant 1 : i32

    // 累加 0+1+2+...+9 = 45；累乘 1*1*...*1 = 1
    %sum, %prod = mydialect.for %i = %zero to %ten step %one
        iter_args(%acc_sum = %init_sum, %acc_prod = %init_prod) -> (i32, i32) : i32 {
      %new_sum  = mydialect.addi %acc_sum, %i  : i32
      %new_prod = mydialect.muli %acc_prod, %one : i32
      mydialect.yield %new_sum, %new_prod : i32, i32
    }

    mydialect.print %sum  : i32
    mydialect.print %prod : i32

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
