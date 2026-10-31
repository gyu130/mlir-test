// RUN: %mlir-opt --mydialect-place-device --mydialect-lower-device=target=hw %s | %FileCheck %s
// Phase 5：部署目标不是 cpu 时保留 host/device 边界，device_exec 交给
// 设备后端/kernel 生成，不在本流水线内联。
module {
  func.func @main_graph() {
    %0 = mydialect.constant_tensor dense<10> : tensor<i32> : <[], i32>
    %1 = mydialect.tensor_to_scalar %0 : !mydialect.tensor<[], i32>, i32
    %2 = mydialect.hw_mul %1, %1 : i32
    %3 = mydialect.scalar_to_tensor %2 : i32, !mydialect.tensor<[], i32>
    mydialect.print %3 : !mydialect.tensor<[], i32>
    return
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-NEXT: %[[D:.*]] = mydialect.device_exec "hw" : !mydialect.tensor<[], i32> {
// CHECK: mydialect.device_yield
// CHECK-NEXT: }
// CHECK-NEXT: mydialect.print %[[D]] : !mydialect.tensor<[], i32>
// CHECK-NEXT: return
