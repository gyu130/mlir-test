// RUN: %mlir-opt --lower-mydialect-print %s | %FileCheck %s
//
// cmpi_tensor: 逐元素整数比较 → memref 循环 + arith.cmpi
// 结果 memref 元素类型为 i1，predicate 采用 arith 枚举（0=eq, 1=ne, ..., 9=uge）

module {
  // 2D 逐元素 eq 比较
  func.func @cmpi_eq(%a : !mydialect.tensor<[2,3],i32>,
                     %b : !mydialect.tensor<[2,3],i32>)
      -> !mydialect.tensor<[2,3],i1> {
    %r = mydialect.cmpi_tensor %a, %b, 0 : !mydialect.tensor<[2,3],i32>, !mydialect.tensor<[2,3],i1>
    return %r : !mydialect.tensor<[2,3],i1>
  }

  // 2D 逐元素 slt 比较（predicate=2）
  func.func @cmpi_slt(%a : !mydialect.tensor<[2,3],i32>,
                       %b : !mydialect.tensor<[2,3],i32>)
      -> !mydialect.tensor<[2,3],i1> {
    %r = mydialect.cmpi_tensor %a, %b, 2 : !mydialect.tensor<[2,3],i32>, !mydialect.tensor<[2,3],i1>
    return %r : !mydialect.tensor<[2,3],i1>
  }

  // 0 维张量：无循环，直接 load/cmpi/store
  func.func @cmpi_0d(%a : !mydialect.tensor<[],i32>,
                     %b : !mydialect.tensor<[],i32>)
      -> !mydialect.tensor<[],i1> {
    %r = mydialect.cmpi_tensor %a, %b, 0 : !mydialect.tensor<[],i32>, !mydialect.tensor<[],i1>
    return %r : !mydialect.tensor<[],i1>
  }
}

// ----- 2D eq -----
// CHECK-LABEL: func.func @cmpi_eq
// CHECK: memref.alloc() : memref<2x3xi1>
// CHECK: scf.for
// CHECK-NESTED: scf.for
// CHECK-NESTED-NOT: scf.for
// CHECK-NESTED-NESTED: arith.cmpi eq
// CHECK-NESTED-NESTED: memref.store
// CHECK-NOT: mydialect.cmpi_tensor

// ----- 2D slt -----
// CHECK-LABEL: func.func @cmpi_slt
// CHECK: arith.cmpi slt

// ----- 0D -----
// CHECK-LABEL: func.func @cmpi_0d
// CHECK: memref.alloc() : memref<i1>
// CHECK-NOT: scf.for
// CHECK: arith.cmpi eq
// CHECK: memref.store
