module {
  func.func @main() -> i32 {
    %a = arith.constant 1.0 : f32
    %b = arith.constant 2.0 : f32
    %r = mydialect.addi %a, %b : f32
    return %r : i32
  }
}
