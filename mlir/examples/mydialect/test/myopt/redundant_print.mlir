// RUN: /data1/windy01/mlir-test/llvm-project-22.1.8.src/build_mlir/bin/mlir-opt --my-opt %s | /data1/windy01/mlir-test/llvm-project-22.1.8.src/build_mlir/bin/FileCheck %s
module {
  func.func @main() {
    %v = arith.constant 42 : i32
    mydialect.print %v : i32
    mydialect.print %v : i32
    mydialect.print %v : i32
    return
  }
}
// CHECK-LABEL: func.func @main() {
// CHECK:   [[CST:%.+]] = arith.constant 42 : i32
// CHECK:   mydialect.print [[CST]] : i32
// CHECK-NOT: mydialect.print
// CHECK:   return

