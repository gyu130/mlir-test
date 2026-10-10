// 自定义类型 wrapped_i32 的端到端验证：
//   wrap: i32 -> !mydialect.wrapped_i32
//   print 直接接受 wrapped_i32（lowering 时经 MyDialectTypeConverter 转 i32）
//   unwrap: !mydialect.wrapped_i32 -> i32
// 运行: mydriver typeConverter.mlir 42  =>  打印 42，main 返回 42
// 注意：自定义类型在汇编中必须带 `!` 前缀（!mydialect.wrapped_i32）
module {
  func.func @main(%arg0: i32) -> i32 {
    %w = mydialect.wrap %arg0 : i32, !mydialect.wrapped_i32
    mydialect.print %w : !mydialect.wrapped_i32
    %u = mydialect.unwrap %w : !mydialect.wrapped_i32, i32
    return %u : i32
  }
}
