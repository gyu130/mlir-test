#ifndef MYDIALECT_TD
#define MYDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

def MyDialect : Dialect {
  let name = "mydialect";
  let summary = "A minimal custom MLIR dialect example";
  let description = "Minimal demo dialect with print op";
  let cppNamespace = "::mydialect";
  // 声明自定义类型的 parse/print 钩子（C++ 实现中 override）
  let extraClassDeclaration = [{
    /// 解析/打印自定义类型（wrapped_i32）
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

// ========== 自定义类型 WrappedI32Type：包装一个 i32 ==========
// TypeDef 让 TableGen 生成 WrappedI32Type C++ 类（经 -gen-typedef-defs）
// 注意：TD 定义后仍需在 Dialect::initialize() addTypes<...> 注册，
// 并手写 parseType/printType 支持汇编中的 mydialect.wrapped_i32
def WrappedI32Type : TypeDef<MyDialect, "WrappedI32"> {
  let mnemonic = "wrapped_i32";        // 汇编中显示为 mydialect.wrapped_i32
}

// 自定义张量类型：shape 是 ArrayAttr，elementType 是 Type
def MyTensorType : TypeDef<MyDialect, "MyTensor"> {
  let mnemonic = "tensor"; // IR文本: mydialect.tensor<...>
  let parameters = (ins
    AttrParameter<"::mlir::ArrayAttr", "shape">:$shape,
    TypeParameter<"::mlir::Type", "elementType">:$elementType
  );
  //let assemblyFormat = "`<` $shape `,` $elementType `>`";
  let hasCustomAssemblyFormat = 1;
}

// unwrap 算子：mydialect.unwrap : wrapped_i32 -> i32（显式去包装，供测试用）
def UnwrapOp : Op<MyDialect, "unwrap"> {
  let summary = "Unwrap wrapped_i32 to raw i32";
  let arguments = (ins AnyType:$input);
  let results = (outs I32:$out);
  let assemblyFormat = "$input attr-dict `:` type($input) `,` type($out)";
}

// wrap 算子：mydialect.wrap : i32 -> wrapped_i32（显式包装，供测试用）
def WrapOp : Op<MyDialect, "wrap"> {
  let summary = "Wrap i32 into wrapped_i32";
  let arguments = (ins I32:$input);
  let results = (outs AnyType:$out);
  let assemblyFormat = "$input attr-dict `:` type($input) `,` type($out)";
}

// 【类型约束】用于 Op 定义中类型检查
def WrappedI32TypeConstraint : Type<CPred<"$_self.isa<::mydialect::WrappedI32Type>()">, "wrapped_i32 type">;

def PrintOp : Op<MyDialect, "print"> {
  let summary = "Print integer value op";
  let description = "Print an integer value to stdout";
  let arguments = (ins AnyType:$value);
  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}

def PrintStrOp : Op<MyDialect, "print_str"> {
  let summary = "print string literal";
  let arguments = (ins AnyType:$str);
  let results = (outs);
  let assemblyFormat = "$str `:` type($str) attr-dict";
}

def AddI32Op : Op<MyDialect, "addi"> {
  let summary = "i32 add for mydialect";
  let description = "Add two i32 operands";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$sum);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($sum)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def MuliOp : Op<MyDialect, "muli"> {
  let summary = "i32 multiply for mydialect";
  let description = "Multiply two i32 operands";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$product);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($product)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def SubI32Op : Op<MyDialect, "subi"> {
  let summary = "i32 subtract for mydialect";
  let description = "Subtract two i32 operands";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$diff);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($diff)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def DivSIOp : Op<MyDialect, "divi_s"> {
  let summary = "signed i32 divide for mydialect";
  let description = "Signed divide two i32 operands";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$quotient);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($quotient)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// mydialect.cmpi slt, %a, %b : i32  →  i1
def CmpiOp : Op<MyDialect, "cmpi"> {
  let arguments = (ins I32:$lhs, I32:$rhs, StrAttr:$predicate);
  let results = (outs I1:$res);
  let assemblyFormat = "$lhs `,` $rhs `,` $predicate `:` type($lhs) attr-dict";
  let hasVerifier = 1;
}

def ForOp : Op<MyDialect, "for"> {
  // 普通循环边界 + 可选 iter_args 初始值
  let arguments = (ins I32:$lower, I32:$upper, I32:$step,
                   Variadic<I32>:$iter_args);

  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<I32>:$results);

  // 自定义汇编格式（parse/print 在 MyDialect.cpp 中实现）
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    mlir::Value getInductionVar();
    mlir::ValueRange getRegionIterArgs();
  }];
}

// YieldOp：支持多个返回值，作为ForOp的terminator
def YieldOp : Op<MyDialect, "yield",
      [Terminator, ParentOneOf<["ForOp"]>]> {
  let arguments = (ins Variadic<I32>:$values);
  let results = (outs);
  let assemblyFormat = "(` ` $values^ `:` type($values))? attr-dict";
}

//===== 内存操作：alloc / load / store =====

// mydialect.alloc : 栈上分配一维数组
// 用法：%buf = mydialect.alloc(%size) : memref<?xi32>
def AllocOp : Op<MyDialect, "alloc"> {
  let arguments = (ins I32:$size);
  let results = (outs AnyMemRef:$res);

  let assemblyFormat = "$size attr-dict `:` type($res)";
  let hasVerifier = 1;
}

// mydialect.load : 数组读取
// %val = mydialect.load(%buf, %idx) : i32
def LoadOp : Op<MyDialect, "load"> {
  let arguments = (ins AnyMemRef:$memref, I32:$index);
  let results = (outs I32:$res);

  let assemblyFormat = "$memref `,` $index attr-dict `:` type($memref) `,` type($res)";
  let hasVerifier = 1;
}

// mydialect.store : 数组写入
// mydialect.store %val, %buf, %idx : i32
def StoreOp : Op<MyDialect, "store"> {
  let arguments = (ins I32:$value, AnyMemRef:$memref, I32:$index);
  let results = (outs);

  let assemblyFormat = "$value `,` $memref `,` $index attr-dict `:` type($value) `,` type($memref)";
  let hasVerifier = 1;
}


#endif

