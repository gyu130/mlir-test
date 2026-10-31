// RUN: %mlir-opt --mydialect-schedule=verbose %s 2>&1 | %FileCheck %s
// 调度前后周期统计：in-order 基线 9 拍，list scheduling 后 8 拍
module {
  func.func @sched(%a : i32, %b : i32, %c : i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    %1 = arith.muli %0, %c : i32
    %2 = arith.addi %b, %c : i32
    %3 = arith.addi %2, %2 : i32
    %4 = arith.addi %3, %3 : i32
    %5 = arith.muli %4, %c : i32
    %6 = arith.addi %1, %5 : i32
    return %6 : i32
  }
}

// CHECK: [mydialect-schedule] block segment: 7 ops, cycles 9 -> 8
// CHECK: [mydialect-schedule] total: cycles 9 -> 8
