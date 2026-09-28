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
}

def MuliOp : Op<MyDialect, "muli"> {
  let summary = "i32 multiply for mydialect";
  let description = "Multiply two i32 operands";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$product);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($product)";
  let hasFolder = 1;
}

def SubI32Op : Op<MyDialect, "subi"> {
  let summary = "i32 subtract for mydialect";
  let description = "Subtract two i32 operands";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$diff);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($diff)";
  let hasFolder = 1;
}

def DivSIOp : Op<MyDialect, "divi_s"> {
  let summary = "signed i32 divide for mydialect";
  let description = "Signed divide two i32 operands";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$quotient);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($quotient)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

#endif

