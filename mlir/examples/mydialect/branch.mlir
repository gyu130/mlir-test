module {
  func.func @main() -> i32 {
    %a = arith.constant 10 : i32
    %b = arith.constant 20 : i32
    %cond = mydialect.cmpi %a, %b, "slt" : i32
    %result = scf.if %cond -> i32 {
      %then_val = arith.constant 30 : i32
      scf.yield %then_val : i32
    } else {
      %else_val = arith.constant 0 : i32
      scf.yield %else_val : i32
    }
    mydialect.print %result : i32
    return %result : i32
  }
}
