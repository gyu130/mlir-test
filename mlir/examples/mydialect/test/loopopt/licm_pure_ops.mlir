// RUN: %mlir-opt --mydialect-licm %s | %FileCheck %s
//
// 通用 LICM（--mydialect-licm）：
//   - 循环内「纯且可推测」（Pure）且操作数全部定义在循环外的叶子 op 被外提；
//   - 依赖循环块参数（迭代携带值）的 op 保持原位；
//   - 覆盖 scf.for 与 mydialect.for，嵌套链（transpose→transpose）按顺序外提。
// transpose 等张量算子外提后不再随循环重复计算，可供 layout 传播复用。

func.func @hoist_transpose(%a : !mydialect.tensor<[2,3],i32>) -> !mydialect.tensor<[2,3],i32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %a) -> !mydialect.tensor<[2,3],i32> {
    %t = mydialect.transpose_tensor %a, [1 : index, 0 : index] : !mydialect.tensor<[2,3],i32> -> !mydialect.tensor<[3,2],i32>
    %u = mydialect.transpose_tensor %t, [1 : index, 0 : index] : !mydialect.tensor<[3,2],i32> -> !mydialect.tensor<[2,3],i32>
    scf.yield %u : !mydialect.tensor<[2,3],i32>
  }
  return %r : !mydialect.tensor<[2,3],i32>
}

func.func @hoist_addi(%x : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %k = arith.constant 7 : i32
  %r = mydialect.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (i32) : i32 {
    %s = mydialect.addi %x, %k : i32
    %t = mydialect.addi %acc, %s : i32
    mydialect.yield %t : i32
  }
  return %r : i32
}

func.func @no_hoist_dependent(%a : !mydialect.tensor<[4],i32>, %b : !mydialect.tensor<[4],i32>) -> !mydialect.tensor<[4],i32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %a) -> !mydialect.tensor<[4],i32> {
    %s = mydialect.add_tensor %acc, %b : !mydialect.tensor<[4],i32>
    scf.yield %s : !mydialect.tensor<[4],i32>
  }
  return %r : !mydialect.tensor<[4],i32>
}

// 两个 transpose 均被提到 scf.for 之前，循环体内不再出现 transpose
// CHECK-LABEL: func.func @hoist_transpose
// CHECK: %[[T1:.*]] = mydialect.transpose_tensor %arg0
// CHECK-NEXT: %[[T2:.*]] = mydialect.transpose_tensor %[[T1]]
// CHECK-NEXT: %[[R:.*]] = scf.for
// CHECK-NOT: mydialect.transpose_tensor
// CHECK: scf.yield %[[T2]]

// 循环不变量 mydialect.addi 被外提；依赖迭代携带值的 addi 保持原位
// CHECK-LABEL: func.func @hoist_addi
// CHECK: %[[INV:.*]] = mydialect.addi %arg0
// CHECK-NEXT: %[[R:.*]] = mydialect.for
// CHECK-NEXT: %[[ACC:.*]] = mydialect.addi
// CHECK-NEXT: mydialect.yield{{ *}}%[[ACC]] : i32

// 依赖块参数 %acc 的 add_tensor 无法外提，保留在循环体内
// CHECK-LABEL: func.func @no_hoist_dependent
// CHECK: %[[R:.*]] = scf.for
// CHECK-NEXT: %[[S:.*]] = mydialect.add_tensor
// CHECK-NEXT: scf.yield %[[S]]
