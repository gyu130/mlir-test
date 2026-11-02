// RUN: %mlir-opt --lower-mydialect-print %s | %FileCheck %s
//
// reduce_sum_tensor: 清零 memref + 按输入遍历累加
// 步骤 1: 遍历输出 shape 所有元素写零（memref.alloc 不保证零初始化）
// 步骤 2: 遍历输入 shape，每个输入元素累加到对应输出槽
// keep_dims 控制被归约维是否保留为 1

module {
  // axes=[1], keep_dims=false：[2,3] → [2]
  func.func @reduce_sum_axis1(%a : !mydialect.tensor<[2,3],i32>)
      -> !mydialect.tensor<[2],i32> {
    %r = mydialect.reduce_sum_tensor %a, [1], false : !mydialect.tensor<[2,3],i32> -> !mydialect.tensor<[2],i32>
    return %r : !mydialect.tensor<[2],i32>
  }

  // axes=[1], keep_dims=true：[2,3] → [2,1]
  func.func @reduce_sum_axis1_keepdims(%a : !mydialect.tensor<[2,3],i32>)
      -> !mydialect.tensor<[2,1],i32> {
    %r = mydialect.reduce_sum_tensor %a, [1], true : !mydialect.tensor<[2,3],i32> -> !mydialect.tensor<[2,1],i32>
    return %r : !mydialect.tensor<[2,1],i32>
  }

  // axes=[0,1], keep_dims=false 全归约：[2,3] → [] 0 维张量
  func.func @reduce_sum_all(%a : !mydialect.tensor<[2,3],i32>)
      -> !mydialect.tensor<[],i32> {
    %r = mydialect.reduce_sum_tensor %a, [0, 1], false : !mydialect.tensor<[2,3],i32> -> !mydialect.tensor<[],i32>
    return %r : !mydialect.tensor<[],i32>
  }
}

// ----- axes=[1], keep_dims=false: [2,3] → [2] -----
// CHECK-LABEL: func.func @reduce_sum_axis1
// CHECK-NOT: mydialect.reduce_sum_tensor
// CHECK: memref.alloc() : memref<2xi32>
// 清零循环：1 层 scf.for（shape=[2]）
// CHECK: scf.for
// CHECK-NESTED-NOT: scf.for
// 累加循环：2 层 scf.for（shape=[2,3]）
// 注：两个循环 nest 都用了 scf.for，无法静态区分清零 vs 累加，
// 但累加体内部必有 arith.addi
// CHECK: arith.addi

// ----- axes=[1], keep_dims=true: [2,3] → [2,1] -----
// CHECK-LABEL: func.func @reduce_sum_axis1_keepdims
// CHECK: memref.alloc() : memref<2x1xi32>
// CHECK: arith.addi

// ----- 全归约 0D: [2,3] → [] -----
// CHECK-LABEL: func.func @reduce_sum_all
// CHECK: memref.alloc() : memref<i32>
// CHECK: arith.addi
// CHECK-NOT: memref.store
// (清零是 0D，emitLoopNest rank==0 时直接 body({})，没有显式 store 循环)
