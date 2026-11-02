// RUN: %mlir-opt --convert-onnx-to-mydialect %s | %mlir-opt --lower-mydialect-print | %FileCheck %s
//
// ONNX Sub/Div → mydialect.sub_tensor / div_tensor → memref + scf.for 同构 lowering

module {
  func.func @test_sub(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %0 = "onnx.Sub"(%a, %b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  func.func @test_div(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %0 = "onnx.Div"(%a, %b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  // 组合: Sub(Add(a, b), Mul(c, d))
  func.func @test_combo(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
                         %c: tensor<2x2xi32>, %d: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %add = "onnx.Add"(%a, %b) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %mul = "onnx.Mul"(%c, %d) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %sub = "onnx.Sub"(%add, %mul) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %div = "onnx.Div"(%sub, %mul) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %div : tensor<2x2xi32>
  }
}

// ----- ONNX → MyDialect 验证 -----
// RUN: %mlir-opt --convert-onnx-to-mydialect %s | %FileCheck %s --check-prefix=MYDIALECT
//
// MYDIALECT-LABEL: func.func @test_sub
// MYDIALECT-NOT: onnx.Sub
// MYDIALECT: mydialect.sub_tensor
// MYDIALECT-LABEL: func.func @test_div
// MYDIALECT: mydialect.div_tensor

// ----- ONNX → MyDialect → Lowering 验证 -----
// CHECK-LABEL: func.func @test_sub
// CHECK-NOT: onnx.
// CHECK-NOT: mydialect.sub_tensor
// CHECK: arith.subi
//
// CHECK-LABEL: func.func @test_div
// CHECK: arith.divsi
//
// CHECK-LABEL: func.func @test_combo
// CHECK-NOT: onnx.
// CHECK-NOT: mydialect.sub_tensor
// CHECK-NOT: mydialect.div_tensor
// CHECK-NOT: mydialect.add_tensor
// CHECK-NOT: mydialect.mul_tensor
// CHECK: arith.subi
// CHECK: arith.divsi
// CHECK: arith.addi
// CHECK: arith.muli
