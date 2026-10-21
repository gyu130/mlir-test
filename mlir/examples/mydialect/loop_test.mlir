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
