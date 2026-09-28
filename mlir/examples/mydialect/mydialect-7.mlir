// test_div_zero.mlir
module {
  func.func @main() -> i32 {
    %a = arith.constant 10 : i32
    %b = arith.constant 0 : i32
    %res = mydialect.divi_s %a, %b : i32
    return %res : i32
  }
}

