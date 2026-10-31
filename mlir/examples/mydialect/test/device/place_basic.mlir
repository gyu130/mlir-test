// RUN: %mlir-opt --mydialect-place-device %s | %FileCheck %s
// Phase 5：以 hw_mul 为种子，沿纯 mydialect 算子的 def-use 做双向闭包，
// 把整段设备子图包进 mydialect.device_exec，"hw" 是设备标签；
// region 内产生、region 外仍使用的 %5 通过 device_yield 提升为结果，
// host 侧的 print 改用它，形成 host ↔ device 边界。
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
// CHECK-NEXT: %[[C:.*]] = mydialect.constant_tensor dense<10> : tensor<i32> : <[], i32>
// CHECK-NEXT: %[[S:.*]] = mydialect.tensor_to_scalar %[[C]] : !mydialect.tensor<[], i32>, i32
// CHECK-NEXT: %[[M:.*]] = mydialect.hw_mul %[[S]], %[[S]] : i32
// CHECK-NEXT: %[[T:.*]] = mydialect.scalar_to_tensor %[[M]] : i32, !mydialect.tensor<[], i32>
// CHECK-NEXT: mydialect.device_yield %[[T]] : !mydialect.tensor<[], i32>
// CHECK-NEXT: }
// CHECK-NEXT: mydialect.print %[[D]] : !mydialect.tensor<[], i32>
// CHECK-NEXT: return
