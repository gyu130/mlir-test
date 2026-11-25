module {
  llvm.func @main() -> i64 {
    %0 = llvm.mlir.constant(42 : i32) : i32
    %1 = llvm.zext %0 : i32 to i64
    llvm.return %1 : i64
  }
}
