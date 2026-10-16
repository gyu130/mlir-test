// RUN: %mlir-opt --convert-onnx-to-mydialect %s | %mlir-opt - | %FileCheck %s
//
// ONNX Loop 带循环携带（block 参数 iter/cond/carries）的映射：
// region 块参数类型被转换为 mydialect 张量，且转换结果可回读（管道里跑了两次 mlir-opt）。

module {
  func.func @main_graph(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %max_trip = arith.constant 5 : i64
    %cond     = arith.constant 1 : i1
    %zero     = arith.constant dense<0> : tensor<i64>

    %res:2 = "onnx.Loop"(%max_trip, %cond, %arg0, %zero) ({
    ^bb0(%iter: i64, %cond_in: i1, %carry: tensor<2x3xi32>, %acc: tensor<i64>):
      %sum  = "onnx.Add"(%carry, %arg0) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
      %acc2 = "onnx.Add"(%acc,   %zero) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "onnx.Yield"(%cond_in, %sum, %acc2) : (i1, tensor<2x3xi32>, tensor<i64>) -> ()
    }) {input_names = ["max_trip", "cond", "carry"]}
       : (i64, i1, tensor<2x3xi32>, tensor<i64>) -> (tensor<2x3xi32>, tensor<i64>)

    return %res#0 : tensor<2x3xi32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK: mydialect.constant_tensor
// CHECK: mydialect.loop
// CHECK: ^bb0(%{{.*}}: i64, %{{.*}}: i1, %{{.*}}: !mydialect.tensor<[2,3], i32>, %{{.*}}: !mydialect.tensor<[], i64>)
// CHECK: mydialect.add_tensor
// CHECK: mydialect.loop_yield
// CHECK-NOT: onnx.
