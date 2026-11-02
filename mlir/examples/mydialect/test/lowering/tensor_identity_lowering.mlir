// RUN: %mlir-opt --lower-mydialect-print %s | %FileCheck %s
//
// identity_tensor: 零拷贝别名（不做 memref.copy）
// 降级后结果 memref 与输入 memref 指向同一 buffer

module {
  func.func @identity_2d(%a : !mydialect.tensor<[4,4],i32>)
      -> !mydialect.tensor<[4,4],i32> {
    %r = mydialect.identity_tensor %a : !mydialect.tensor<[4,4],i32>
    return %r : !mydialect.tensor<[4,4],i32>
  }
}

// CHECK-LABEL: func.func @identity_2d
// CHECK-NOT: mydialect.identity_tensor
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.store
// CHECK-NOT: memref.copy
// CHECK-NOT: scf.for
// CHECK: return %arg0
