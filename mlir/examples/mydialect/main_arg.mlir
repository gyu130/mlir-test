module {
  // main 接收一个 i32 参数，打印它
  func.func @main(%arg0 : i32) -> i32 {
    mydialect.print %arg0 : i32
    %0 = llvm.mlir.addressof @str_const : !llvm.ptr
    mydialect.print_str %0 : !llvm.ptr
    return %arg0 : i32
  }
  llvm.mlir.global internal constant @str_const("Hello from main arg!\00") {addr_space = 0 : i32}
}

