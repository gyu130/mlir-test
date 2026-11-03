// RUN: %mlir-opt --constant-fold-mydialect %s | %FileCheck %s
//
// LICM 扩展到 memref.load：
//   - 循环内只写新鲜 alloc（与读取源不别名）→ 不变 load 外提到循环外
//   - 循环写函数参数 memref（可能与读取源别名）→ 不外提
// 使用动态循环边界，避免常量循环被展开，从而可观察外提结果。

func.func @hoist_load(%src: memref<?xi32>, %lb: i32, %ub: i32, %step: i32) {
  %c2 = arith.constant 2 : i32
  %c100 = arith.constant 100 : i32
  %buf = mydialect.alloc %c100 : memref<?xi32>
  mydialect.for %iv = %lb to %ub step %step : i32 {
    %v = mydialect.load %src, %c2 : memref<?xi32>, i32
    mydialect.store %v, %buf, %iv : i32, memref<?xi32>
  }
  return
}

func.func @no_hoist_alias(%a: memref<?xi32>, %b: memref<?xi32>, %lb: i32, %ub: i32, %step: i32) {
  %c2 = arith.constant 2 : i32
  mydialect.for %iv = %lb to %ub step %step : i32 {
    %v = mydialect.load %a, %c2 : memref<?xi32>, i32
    mydialect.store %v, %b, %iv : i32, memref<?xi32>
  }
  return
}

// CHECK-LABEL: func.func @hoist_load
// CHECK: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: %[[BUF:.*]] = mydialect.alloc
// CHECK-NEXT: %[[V:.*]] = mydialect.load %arg0, %[[C2]] : memref<?xi32>, i32
// CHECK-NEXT: mydialect.for
// CHECK-NEXT: mydialect.store %[[V]], %[[BUF]]
// CHECK-NOT: mydialect.load

// CHECK-LABEL: func.func @no_hoist_alias
// CHECK: mydialect.for
// CHECK-NEXT: %[[V2:.*]] = mydialect.load %arg0, %{{.*}} : memref<?xi32>, i32
// CHECK-NEXT: mydialect.store %[[V2]], %arg1
