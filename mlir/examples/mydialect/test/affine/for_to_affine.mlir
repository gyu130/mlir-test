// RUN: %mlir-opt --mydialect-for-to-affine %s | %FileCheck %s
//
// 常量正 step 的 mydialect.for 降级为 affine.for（IV 变 index，body 内
// index_cast 回 i32）；动态 step / 负 step 保留 mydialect.for。

func.func @basic(%a: memref<16xi32>) {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  mydialect.for %iv = %c0 to %c16 step %c1 : i32 {
    %v = mydialect.load %a, %iv : memref<16xi32>, i32
    mydialect.store %v, %a, %iv : i32, memref<16xi32>
  }
  return
}

func.func @sum(%a: memref<16xi32>) -> i32 {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  %s = mydialect.for %iv = %c0 to %c16 step %c1 iter_args(%acc = %c0) -> (i32) : i32 {
    %v = mydialect.load %a, %iv : memref<16xi32>, i32
    %n = arith.addi %acc, %v : i32
    mydialect.yield %n : i32
  }
  return %s : i32
}

func.func @iv_bound(%a: memref<16xi32>) {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  mydialect.for %i = %c0 to %c16 step %c1 : i32 {
    mydialect.for %j = %c0 to %i step %c1 : i32 {
      %v = mydialect.load %a, %j : memref<16xi32>, i32
      mydialect.store %v, %a, %j : i32, memref<16xi32>
    }
  }
  return
}

func.func @dyn_step(%lb: i32, %ub: i32, %s: i32) {
  mydialect.for %iv = %lb to %ub step %s : i32 {
  }
  return
}

func.func @neg_step(%a: memref<16xi32>) {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %cm1 = arith.constant -1 : i32
  mydialect.for %i = %c16 to %c0 step %cm1 : i32 {
  }
  return
}

// CHECK-LABEL: func.func @basic
// CHECK: affine.for %[[IV:.*]] = 0 to 16 {
// CHECK-NEXT: %[[IV32:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK-NEXT: %[[V:.*]] = mydialect.load %arg0, %[[IV32]] : memref<16xi32>, i32
// CHECK-NEXT: mydialect.store %[[V]], %arg0, %[[IV32]]
// CHECK-NOT: mydialect.for

// CHECK-LABEL: func.func @sum
// CHECK: %[[S:.*]] = affine.for %[[IV2:.*]] = 0 to 16 iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32) {
// CHECK-NEXT: %[[IV32B:.*]] = arith.index_cast %[[IV2]] : index to i32
// CHECK: %[[N:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK-NEXT: affine.yield %[[N]] : i32
// CHECK-NOT: mydialect.yield

// CHECK-LABEL: func.func @iv_bound
// CHECK: affine.for %[[O:.*]] = 0 to 16 {
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to #{{.*}}(%[[O]]) {
// CHECK-NOT: mydialect.for

// CHECK-LABEL: func.func @dyn_step
// CHECK: mydialect.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32

// CHECK-LABEL: func.func @neg_step
// CHECK: mydialect.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32
