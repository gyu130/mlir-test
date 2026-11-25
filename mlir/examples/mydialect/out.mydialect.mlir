module {
  func.func @main_graph(%arg0: !mydialect.tensor<[2,3], i32>) -> !mydialect.tensor<[2,3], i32> {
    %true = arith.constant true
    %c5_i64 = arith.constant 5 : i64
    %0 = mydialect.constant_tensor dense<0> : tensor<i64> : <[], i64>
    %1:2 = mydialect.loop %c5_i64, %true, %arg0, %0 : !mydialect.tensor<[2,3], i32>, !mydialect.tensor<[], i64> -> !mydialect.tensor<[2,3], i32>, !mydialect.tensor<[], i64> {
    ^bb0(%arg1: !mydialect.tensor<[], i64>, %arg2: !mydialect.tensor<[], i1>, %arg3: !mydialect.tensor<[2,3], i32>, %arg4: !mydialect.tensor<[], i64>):
      %2 = mydialect.add_tensor %arg3, %arg0 : <[2,3], i32>
      %3 = mydialect.add_tensor %arg4, %0 : <[], i64>
      mydialect.loop_yield %arg2, %2, %3 : !mydialect.tensor<[], i1>, !mydialect.tensor<[2,3], i32>, !mydialect.tensor<[], i64>
    }
    return %1#0 : !mydialect.tensor<[2,3], i32>
  }
}

