module {
  func.func @gpu_then_hw(
      %A : memref<2x3xi32>, %B : memref<3x4xi32>, %C : memref<2x4xi32>,
      %B2 : memref<4x5xi32>, %D : memref<2x5xi32>) {
    %gpu_c = mydialect.device_exec "gpu" : memref<2x4xi32> {
      mydialect.hw_matmul %A, %B, %C : memref<2x3xi32>, memref<3x4xi32>, memref<2x4xi32>
      mydialect.device_yield %C : memref<2x4xi32>
    }
    %hw_out = mydialect.device_exec "hw" : memref<2x5xi32> {
      mydialect.hw_matmul %gpu_c, %B2, %D : memref<2x4xi32>, memref<4x5xi32>, memref<2x5xi32>
      mydialect.device_yield %D : memref<2x5xi32>
    }
    return
  }
}
