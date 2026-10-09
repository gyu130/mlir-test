module {
  func.func @main() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    mydialect.print %c42_i32 : i32

    // 字符串全局常量，得到 char* 指针
    %str_ptr = llvm.mlir.addressof @str_const : !llvm.ptr
    mydialect.print_str %str_ptr : !llvm.ptr

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
  llvm.mlir.global internal constant @str_const("Hello MyDialect!\00") : !llvm.array<17 x i8>
}

