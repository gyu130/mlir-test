// RUN: %mlir-opt --mydialect-parallelize %s | %FileCheck %s
// 两类不能被并行的循环必须保持 scf.for：
//   1. 带 iter_args（跨迭代携带依赖）
//   2. 循环体含外部调用（副作用，执行顺序不能改变）
module {
  func.func @carried(%n : index, %init : i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
      %iv = arith.index_cast %i : index to i32
      %s = arith.addi %acc, %iv : i32
      scf.yield %s : i32
    }
    return %r : i32
  }

  func.func private @sink(i32)

  func.func @withcall(%buf : memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c64 step %c1 {
      %v = memref.load %buf[%i] : memref<64xi32>
      func.call @sink(%v) : (i32) -> ()
    }
    return
  }
}

// CHECK-LABEL: func.func @carried
// CHECK: scf.for {{.*}} iter_args
// CHECK-LABEL: func.func @withcall
// CHECK: scf.for
// CHECK: func.call @sink
// CHECK-NOT: scf.forall
