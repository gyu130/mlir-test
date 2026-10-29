// RUN: %mlir-opt --mydialect-count-op-cycle-est %s 2>&1 | %FileCheck %s
// RUN: %mlir-opt --mydialect-count-op-cycle-est="add-fu=1" %s 2>&1 | %FileCheck --check-prefix=ONE-FU %s
// RUN: %mlir-opt --mydialect-count-op-cycle-est="add-latency=2 mul-latency=6" %s 2>&1 | %FileCheck --check-prefix=LATENCY %s

// 算子调度 / 资源建模：统计 addi/muli 数量并按功能单元数量/延迟估算周期
module {
  func.func @main(%a : i32, %b : i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %r0 = mydialect.addi %a, %b : i32
    %r1 = mydialect.addi %r0, %c0 : i32
    %r2 = mydialect.muli %r1, %b : i32
    %r3 = mydialect.addi %r2, %a : i32
    return %r3 : i32
  }

  // hw_mul 也计入乘法（流水线中 hw_mul 会先 lower 成 arith.muli）
  func.func @hw(%a : i32, %b : i32) -> i32 {
    %p = mydialect.hw_mul %a, %b : i32
    return %p : i32
  }
}

// 默认配置：add_fu=2 / addi latency=1，mul_fu=1 / muli latency=4
//   addi=3  -> ceil(3/2)+1-1 = 2
//   muli=1+1(hw_mul) -> ceil(2/1)+4-1 = 5
// CHECK: mydialect.addi count : 3
// CHECK: mydialect.muli count : 2
// CHECK: HW FU config: add_fu=2, mul_fu=1
// CHECK: addi latency=1, muli latency=4
// CHECK: add pipeline cycle : 2
// CHECK: mul pipeline cycle : 5
// CHECK: Estimated total cycle (ideal no dependency): 5

// add-fu=1：addi=3 -> ceil(3/1)+1-1 = 3，总周期 max(3,5) = 5
// ONE-FU: add pipeline cycle : 3
// ONE-FU: Estimated total cycle (ideal no dependency): 5

// add-latency=2 / mul-latency=6：addi=3 -> ceil(3/2)+2-1 = 3
//                            muli=2 -> ceil(2/1)+6-1 = 7
// LATENCY: add pipeline cycle : 3
// LATENCY: mul pipeline cycle : 7
// LATENCY: Estimated total cycle (ideal no dependency): 7
