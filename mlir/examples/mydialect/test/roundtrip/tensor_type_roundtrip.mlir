// RUN: %mlir-opt %s | %mlir-opt | %FileCheck %s
//
// 自定义张量类型在 op 汇编中的 stripped 形式必须能回读：
// 第一次 mlir-opt 打印成类型体 <[...], elemTy>，第二次 mlir-opt 必须能解析。

module {
  func.func @tensor_rt(%a : !mydialect.tensor<[2,3],i32>,
                       %b : !mydialect.tensor<[2,3],i32>)
      -> (!mydialect.tensor<[2,3],i32>, !mydialect.tensor<[2,3],i1>) {
    %0 = mydialect.add_tensor %a, %b : !mydialect.tensor<[2,3],i32>
    %1 = mydialect.cmpi_tensor %0, %b, 4 : !mydialect.tensor<[2,3],i32>, !mydialect.tensor<[2,3],i1>
    return %0, %1 : !mydialect.tensor<[2,3],i32>, !mydialect.tensor<[2,3],i1>
  }
}

// CHECK-LABEL: func.func @tensor_rt
// CHECK: mydialect.add_tensor %arg0, %arg1 : <[2,3], i32>
// CHECK: mydialect.cmpi_tensor %{{.*}}, %arg1, 4 : <[2,3], i32>, <[2,3], i1>
