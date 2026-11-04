// RUN: %mlir-opt --mydialect-distribute %s | %FileCheck %s
//
// 两个 device_exec，device 属性分别为 "gpu" 和 "hw"：
//  1. 第一个在 gpu 上跑 hw_matmul（A*B=C），C yield 出来；
//  2. 第二个在 hw 上跑 hw_matmul（C*B2=D），消费第一个的产出；
// DistributePass 应自动：
//   - 把 host 参数 A/B/C 在 gpu 上 alloc + copy；
//   - 把第一个 device_exec 产出（gpu 上的 C）在 hw 上 alloc + copy；
//   - 把 host 参数 B2/D 在 hw 上 alloc + copy；
//   - device_copy 默认同步（不打印 async 属性）。

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

// ===== CHECK =====
// 先把 host 参数 A 在 gpu 上 alloc/copy
// CHECK-DAG: mydialect.device_alloc "gpu" : memref<2x3xi32>
// CHECK-DAG: mydialect.device_copy %arg0, {{%.*}} : memref<2x3xi32> -> memref<2x3xi32>

// gpu device_exec 保持 "gpu"
// CHECK-DAG: mydialect.device_exec "gpu"

// 跨设备：gpu 产出 → hw
// CHECK-DAG: mydialect.device_alloc "hw" : memref<2x4xi32>
// CHECK-DAG: mydialect.device_copy {{%.*}}, {{%.*}} : memref<2x4xi32> -> memref<2x4xi32>

// hw device_exec 保持 "hw"
// CHECK-DAG: mydialect.device_exec "hw"

// 不应有 "cpu" device_exec（没有 device_exec 的 device 是 "cpu"）
// CHECK-NOT: device_exec "cpu"
