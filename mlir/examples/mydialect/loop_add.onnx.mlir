module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.compile_options" = "--EmitONNXIR loop_add.onnx", "onnx-mlir.compiler_version" = "onnx-mlir version 0.5.1 (1e017c9f)", "onnx-mlir.symbol-postfix" = "loop_add"} {
  func.func @main_graph(%arg0: tensor<2x3xi32> {onnx.name = "x"}) -> (tensor<2x3xi32> {onnx.dim_params = "0:Loopout_dim_0,1:Loopout_dim_1", onnx.name = "out"}) {
    %0 = onnx.Constant dense<1> : tensor<i64>
    %1 = onnx.Constant dense<0> : tensor<i64>
    %2 = onnx.Constant dense<true> : tensor<i1>
    %3 = onnx.Constant dense<5> : tensor<i64>
    %4:2 = "onnx.Loop"(%3, %2, %arg0, %1) ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i1>, %arg3: tensor<2x3xi32>, %arg4: tensor<i64>):
      %5 = "onnx.Add"(%arg3, %arg0) {onnx_node_name = "/Add"} : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
      %6 = "onnx.Add"(%arg4, %0) {onnx_node_name = "/Add_1"} : (tensor<i64>, tensor<i64>) -> tensor<i64>
      onnx.Yield {onnx_node_name = "onnx.Yield_0"} %arg2, %5, %6 : tensor<i1>, tensor<2x3xi32>, tensor<i64>
    }) {input_dim_params = ["", "", "", ""], input_names = ["6", "cond", "carry.9", "i.11"], onnx_node_name = "/Loop", output_names = ["/Less_output_0", "/Add_output_0", "/Add_1_output_0"]} : (tensor<i64>, tensor<i1>, tensor<2x3xi32>, tensor<i64>) -> (tensor<2x3xi32>, tensor<i64>)
    return %4#0 : tensor<2x3xi32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
