module {
func.func @hoist_demo(%in: i32) -> (i32,i32) {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  %global_val = arith.constant 999 : i32
  %r:2 = mydialect.for %iv = %c0 to %c10 step %c1 iter_args(%acc=%in, %unused=%c1) -> (i32,i32) : i32 {
    %hoist_add = mydialect.addi %global_val, %c1 : i32
    %new_acc = mydialect.addi %acc, %iv : i32
    mydialect.yield %new_acc, %hoist_add : i32, i32
  }
  return %r#0, %r#1 : i32, i32
}
}

