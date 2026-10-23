// ONNX Loop 测试输入（M/cond 须为 0 维张量，符合 onnx-mlir 输出格式，
// 也满足 ONNXLoopOp ODS 对 tensor 操作数的约束）
module {
  func.func @main_graph(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %max_trip = onnx.Constant dense<5> : tensor<i64>
    %cond     = onnx.Constant dense<true> : tensor<i1>
    %zero     = onnx.Constant dense<0> : tensor<i64>

    %res:2 = "onnx.Loop"(%max_trip, %cond, %arg0, %zero) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>, %carry: tensor<2x3xi32>, %acc: tensor<i64>):
      %sum  = "onnx.Add"(%carry, %arg0) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
      %acc2 = "onnx.Add"(%acc,   %zero) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "onnx.Yield"(%cond_in, %sum, %acc2) : (tensor<i1>, tensor<2x3xi32>, tensor<i64>) -> ()
    }) {input_names = ["max_trip", "cond", "carry"]}
       : (tensor<i64>, tensor<i1>, tensor<2x3xi32>, tensor<i64>) -> (tensor<2x3xi32>, tensor<i64>)

    return %res#0 : tensor<2x3xi32>
  }
}
