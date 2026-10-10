// RUN: %mlir-opt --lower-mydialect-print %s | %FileCheck %s
module {
  func.func @main() {
    %c42 = arith.constant 42 : i32
    mydialect.print %c42 : i32
    return
  }
}

// CHECK-LABEL: func.func @main()
// CHECK: llvm.call @my_runtime_print_i32
// CHECK: return

