#ifndef MYDIALECT_TD
#define MYDIALECT_TD

include "mlir/IR/OpBase.td"

def MyDialect : Dialect {
  let name = "mydialect";
  let summary = "A minimal custom MLIR dialect example";
  let description = "Minimal demo dialect with print op";
  let cppNamespace = "::mydialect";
}

def PrintOp : Op<MyDialect, "print"> {
  let summary = "Print integer value op";
  let description = "Print an integer value to stdout";
  let arguments = (ins I32:$value);
  let results = (outs);

  let hasCustomAssemblyFormat = 1;  

  let hasVerifier = 1;
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

#endif

