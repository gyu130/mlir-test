// RUN: /data1/windy01/mlir-test/llvm-project-22.1.8.src/build_mlir/bin/mlir-opt --my-opt %s | /data1/windy01/mlir-test/llvm-project-22.1.8.src/build_mlir/bin/FileCheck %s
module {
  func.func @main(%arg0:i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %arg0, %arg0 : i32
    mydialect.print %1 : i32
    return %1 : i32
  }
}

// CHECK-LABEL: func.func @main
// CHECK: [[RES:%.+]] = arith.addi %arg0, %arg0 : i32
// CHECK: mydialect.print [[RES]] : i32
// CHECK: return [[RES]] : i32
// CHECK-NOT: arith.addi

