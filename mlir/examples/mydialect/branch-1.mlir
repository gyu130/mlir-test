module {
  func.func private @get_input() -> i32
  func.func @main() -> i32 {
    %a = func.call @get_input() : () -> i32
    %b = arith.constant 50 : i32
    %cond = mydialect.cmpi %a, %b, "slt" : i32
    %res = scf.if %cond -> i32 {
      %v1 = arith.constant 99 : i32
      scf.yield %v1 : i32
    } else {
      %v2 = arith.constant 77 : i32
      scf.yield %v2 : i32
    }
    mydialect.print %res : i32
    return %res : i32
  }
}

