module {
  func.func @hw_matmul_gpu(%A : memref<2x3xi32>, %B : memref<3x4xi32>, %C : memref<2x4xi32>) {
    mydialect.device_exec "hw" : memref<2x4xi32> {
      mydialect.hw_matmul %A, %B, %C : memref<2x3xi32>, memref<3x4xi32>, memref<2x4xi32>
      mydialect.device_yield %C : memref<2x4xi32>
    }
    return
  }
}
