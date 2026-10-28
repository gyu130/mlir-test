// RUN: %mlir-opt --lower-hw-mul %s | %FileCheck --check-prefix=SW %s
// RUN: %mlir-opt --lower-hw-mul="use-hardware-path=true" %s | %FileCheck --check-prefix=HW %s

// 测试 mydialect.hw_mul 两条 lowering 路径
module {
  func.func @hw_test(%a : i32, %b : i32) -> i32 {
    %c = mydialect.hw_mul %a, %b : i32
    return %c : i32
  }

  // 常量折叠验证
  func.func @hw_const() -> i32 {
    %c10 = arith.constant 10 : i32
    %c20 = arith.constant 20 : i32
    %c = mydialect.hw_mul %c10, %c20 : i32
    return %c : i32
  }
}

// SW 路径: hw_mul -> arith.muli
// SW-LABEL: func.func @hw_test
// SW: arith.muli %a, %b : i32

// HW 路径: hw_mul -> llvm.call_intrinsic
// HW-LABEL: func.func @hw_test
// HW: llvm.call_intrinsic @llvm.mydialect.hw.mul.i32
