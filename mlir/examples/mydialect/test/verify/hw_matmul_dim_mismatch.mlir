// RUN: %mlir-opt %s 2>&1 | %FileCheck %s

// 测试 hw_matmul verifier：K/M/N 静态维度必须匹配

module {
  func.func @bad_k(%A : memref<2x3xi32>, %B : memref<4x5xi32>, %C : memref<2x5xi32>) {
    mydialect.hw_matmul %A, %B, %C : memref<2x3xi32>, memref<4x5xi32>, memref<2x5xi32>
    return
  }
}

// CHECK: error: 'mydialect.hw_matmul' op K dimension mismatch (3 vs 4)
