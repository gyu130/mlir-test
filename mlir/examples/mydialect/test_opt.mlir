module {
  func.func @main(%arg0:i32) ->i32 {
    // 连续重复print，优化后只会保留最后一条
    mydialect.print %arg0 : i32
    mydialect.print %arg0 : i32
    mydialect.print %arg0 : i32

    // 两条完全一样 addi，CSE合并
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %arg0, %arg0 : i32
    mydialect.print %1 : i32
    return %1 : i32
  }
}

