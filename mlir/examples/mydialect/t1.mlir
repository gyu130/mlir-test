llvm.func @my_runtime_print_i32(i32)
llvm.func @main() -> i64 {
  %c = llvm.mlir.constant(111 : i32) : i32
  llvm.call @my_runtime_print_i32(%c) : (i32) -> ()
  %r = llvm.mlir.constant(111 : i64) : i64
  llvm.return %r : i64
}
