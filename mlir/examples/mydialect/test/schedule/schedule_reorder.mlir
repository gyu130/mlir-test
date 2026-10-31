// RUN: %mlir-opt --mydialect-schedule %s | %FileCheck %s
// Phase 4：list scheduling 把独立的加法链提前，缩短关键路径
//   原顺序：addi, muli(依赖), addi, addi, addi, muli, addi
//   调度后：先发射两个互相独立的 addi，再交错发射两条链
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

// CHECK-LABEL: func.func @sched
// CHECK-NEXT: %[[A:.*]] = arith.addi %arg1, %arg2 : i32
// CHECK-NEXT: %[[B:.*]] = arith.addi %arg0, %arg1 : i32
// CHECK-NEXT: %[[C:.*]] = arith.addi %[[A]], %[[A]] : i32
// CHECK-NEXT: %[[M0:.*]] = arith.muli %[[B]], %arg2 : i32
// CHECK-NEXT: %[[D:.*]] = arith.addi %[[C]], %[[C]] : i32
// CHECK-NEXT: %[[M1:.*]] = arith.muli %[[D]], %arg2 : i32
// CHECK-NEXT: %[[E:.*]] = arith.addi %[[M0]], %[[M1]] : i32
// CHECK-NEXT: return %[[E]]
