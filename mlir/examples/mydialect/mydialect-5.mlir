module {
  func.func @main() -> i32 {
    %c100 = arith.constant 100 : i32
    %c7   = arith.constant 7 : i32

    %mul  = mydialect.muli %c100, %c7 : i32       // 700
    mydialect.print %mul : i32

    %sub  = mydialect.subi %mul, %c100 : i32       // 600
    mydialect.print %sub : i32

    %div  = mydialect.divi_s %sub, %c7 : i32       // 85 (600/7=85)
    mydialect.print %div : i32

    return %div : i32
  }
}

