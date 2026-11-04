// RUN: %mlir-opt --mydialect-lower-device=target=cpu %s | %FileCheck --check-prefix=CPU %s
// RUN: %mlir-opt --mydialect-lower-device=target=gpu %s | %FileCheck --check-prefix=GPU %s
//
// device_exec + hw_matmul 的 GPU kernel 路径（vs CPU inline）。
// GPU 路径把 hw_matmul 替换为手写 gpu.launch：
//   grid = (ceil(M/16), ceil(N/16), 1)  block = (16, 16, 1)
//   kernel body：每个 thread 算一个 C[i,j]，内层 K 循环累加乘积
// CPU 路径内联 device_exec，hw_matmul 保留在 host 侧。

module {
  func.func @hw_matmul_gpu(%A : memref<2x3xi32>, %B : memref<3x4xi32>, %C : memref<2x4xi32>) {
    mydialect.device_exec "hw" : memref<2x4xi32> {
      mydialect.hw_matmul %A, %B, %C : memref<2x3xi32>, memref<3x4xi32>, memref<2x4xi32>
      mydialect.device_yield %C : memref<2x4xi32>
    }
    return
  }
}

// ===== CPU 路径：内联 device_exec，hw_matmul 保留在 host =====
// CPU-LABEL: func.func @hw_matmul_gpu
// CPU-NOT: device_exec
// CPU-NOT: gpu.launch
// CPU: mydialect.hw_matmul %arg0, %arg1, %arg2

// ===== GPU 路径：保留 device_exec，内部换成 gpu.launch =====
// GPU-LABEL: func.func @hw_matmul_gpu
// GPU-DAG: mydialect.device_exec "hw" : memref<2x4xi32>
// GPU-DAG: gpu.launch blocks({{.*}}) in ({{.*}}) threads({{.*}}) in ({{.*}}) {
// GPU-DAG: gpu.block_id  x
// GPU-DAG: gpu.thread_id  x
// GPU-DAG: scf.if
// GPU-DAG: scf.for
// GPU-DAG: memref.load %arg0[
// GPU-DAG: memref.store %{{.*}}, %arg2[
// GPU-DAG: gpu.terminator
// GPU-NOT: mydialect.hw_matmul
