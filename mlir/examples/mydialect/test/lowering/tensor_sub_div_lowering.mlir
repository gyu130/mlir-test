// RUN: %mlir-opt --lower-mydialect-print %s | %FileCheck %s
//
// sub_tensor / div_tensor: 与 add_tensor / mul_tensor 完全同构
// 整数 → arith.subi / arith.divsi
// 浮点 → arith.subf / arith.divf

module {
  // 2D 整数减法
  func.func @sub_int(%a : !mydialect.tensor<[2,3],i32>,
                     %b : !mydialect.tensor<[2,3],i32>)
      -> !mydialect.tensor<[2,3],i32> {
    %r = mydialect.sub_tensor %a, %b : !mydialect.tensor<[2,3],i32>
    return %r : !mydialect.tensor<[2,3],i32>
  }

  // 2D 整数除法（有符号）
  func.func @div_int(%a : !mydialect.tensor<[2,3],i32>,
                     %b : !mydialect.tensor<[2,3],i32>)
      -> !mydialect.tensor<[2,3],i32> {
    %r = mydialect.div_tensor %a, %b : !mydialect.tensor<[2,3],i32>
    return %r : !mydialect.tensor<[2,3],i32>
  }

  // 浮点减法 / 除法（验证 subf / divf 分支）
  func.func @sub_div_float(%a : !mydialect.tensor<[2,2],f32>,
                            %b : !mydialect.tensor<[2,2],f32>)
      -> !mydialect.tensor<[2,2],f32> {
    %s = mydialect.sub_tensor %a, %b : !mydialect.tensor<[2,2],f32>
    %d = mydialect.div_tensor %a, %b : !mydialect.tensor<[2,2],f32>
    return %s : !mydialect.tensor<[2,2],f32>
  }

  // 0 维张量：无循环，直接 load/subi/divsi/store
  func.func @sub_div_0d(%a : !mydialect.tensor<[],i32>,
                        %b : !mydialect.tensor<[],i32>)
      -> !mydialect.tensor<[],i32> {
    %s = mydialect.sub_tensor %a, %b : !mydialect.tensor<[],i32>
    %d = mydialect.div_tensor %a, %b : !mydialect.tensor<[],i32>
    return %s : !mydialect.tensor<[],i32>
  }
}

// ----- 2D 整数减法 -----
// CHECK-LABEL: func.func @sub_int
// CHECK-NOT: mydialect.sub_tensor
// CHECK: memref.alloc() : memref<2x3xi32>
// CHECK: scf.for
// CHECK-NESTED: scf.for
// CHECK-NESTED-NOT: scf.for
// CHECK-NESTED-NESTED: memref.load
// CHECK-NESTED-NESTED: memref.load
// CHECK-NESTED-NESTED: arith.subi
// CHECK-NESTED-NESTED: memref.store

// ----- 2D 整数除法 -----
// CHECK-LABEL: func.func @div_int
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
// CHECK-NOT: mydialect.div_tensor

// ----- 浮点 -----
// CHECK-LABEL: func.func @sub_div_float
// CHECK: arith.subf
// CHECK: arith.divf

// ----- 0D -----
// CHECK-LABEL: func.func @sub_div_0d
// CHECK: memref.alloc() : memref<i32>
// CHECK-NOT: scf.for
// CHECK: arith.subi
// CHECK: arith.divsi
