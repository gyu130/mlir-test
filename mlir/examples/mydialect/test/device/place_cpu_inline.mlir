// RUN: %mlir-opt --mydialect-place-device --mydialect-lower-device %s | %FileCheck %s
// Phase 5：部署目标为 cpu 时内联 device_exec，设备算子退回 host 执行
// （JIT 路径走这一档），IR 中不再保留任何 device_exec/device_yield。
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
// CHECK-NEXT: %[[C:.*]] = mydialect.constant_tensor dense<10> : tensor<i32> : <[], i32>
// CHECK-NEXT: %[[S:.*]] = mydialect.tensor_to_scalar %[[C]] : !mydialect.tensor<[], i32>, i32
// CHECK-NEXT: %[[M:.*]] = mydialect.hw_mul %[[S]], %[[S]] : i32
// CHECK-NEXT: %[[T:.*]] = mydialect.scalar_to_tensor %[[M]] : i32, !mydialect.tensor<[], i32>
// CHECK-NEXT: mydialect.print %[[T]] : !mydialect.tensor<[], i32>
// CHECK-NEXT: return
// CHECK-NOT: mydialect.device_exec
// CHECK-NOT: mydialect.device_yield
