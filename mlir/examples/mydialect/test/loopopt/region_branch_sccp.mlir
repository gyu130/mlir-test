// RUN: %mlir-opt --sccp %s | %FileCheck %s
//
// RegionBranchOpInterface：sccp 能沿 mydialect.for 的循环回边与
// mydialect.device_exec 的 host/device 边界传播常量。
// MyDialect::materializeConstant：让 mydialect 标量算子的折叠结果
// 能被物化回 SSA 常量，否则折叠结果因无法物化而被丢弃。

func.func @for_const_prop() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %r = mydialect.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %c0) -> (i32) : i32 {
    mydialect.yield %acc : i32
  }
  return %r : i32
}

func.func @device_const() -> i32 {
  %r = mydialect.device_exec "hw" : i32 {
    %c = arith.constant 3 : i32
    mydialect.device_yield %c : i32
  }
  return %r : i32
}

func.func @scalar_fold() -> i32 {
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %s = mydialect.addi %c3, %c4 : i32
  return %s : i32
}

// 循环迭代携带参数被常量替换：region 内 yield 与外层 return 均使用同一常量
// CHECK-LABEL: func.func @for_const_prop
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: mydialect.yield{{ *}}%[[C0]] : i32
// CHECK: return %[[C0]] : i32

// device_yield 常量沿 device 边界传播为 device_exec 结果
// CHECK-LABEL: func.func @device_const
// CHECK: %[[C3:.*]] = arith.constant 3 : i32
// CHECK: mydialect.device_yield %[[C3]] : i32
// CHECK: return %[[C3]] : i32

// materializeConstant：addi 折叠为 7
// CHECK-LABEL: func.func @scalar_fold
// CHECK-NOT: mydialect.addi
// CHECK: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: return %[[C7]] : i32
