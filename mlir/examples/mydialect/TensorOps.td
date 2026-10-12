#ifndef TENSOR_OPS_TD
#define TENSOR_OPS_TD

include "mlir/IR/OpBase.td"
include "mydialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===================== 张量逐元素加法 add_tensor =====================
def AddTensorOp : Op<MyDialect, "add_tensor", [
  Pure, SameOperandsAndResultType
]> {
  let arguments = (ins
    MyTensorType:$lhs,
    MyTensorType:$rhs
  );
  let results = (outs MyTensorType:$res);

  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

//===================== 张量逐元素乘法 mul_tensor =====================
def MulTensorOp : Op<MyDialect, "mul_tensor", [
  Pure, SameOperandsAndResultType
]> {
  let arguments = (ins
    MyTensorType:$lhs,
    MyTensorType:$rhs
  );
  let results = (outs MyTensorType:$res);

  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

//===================== 张量逐元素整数比较 cmpi_tensor =====================
def CmpITensorOp : Op<MyDialect, "cmpi_tensor", [
  Pure, SameTypeOperands
]> {
  let arguments = (ins
    MyTensorType:$lhs,
    MyTensorType:$rhs,
    I64Attr:$predicate
  );
  let results = (outs MyTensorType:$res);

  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($lhs) `,` type($res)";
}

#endif

