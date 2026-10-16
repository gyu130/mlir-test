#ifndef TENSOR_OPS_TD
#define TENSOR_OPS_TD
include "mlir/IR/OpBase.td"
include "mydialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===================== 张量逐元素加法 add_tensor =====================
// SameOperandsAndResultType 自动提供 InferTypeOpInterface 的默认 impl，
// 无需 DeclareOpInterfaceMethods，也无需手写 inferReturnTypes。
def AddTensorOp : Op<MyDialect, "add_tensor", [
  Pure,
  SameOperandsAndResultType
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
  Pure,
  SameOperandsAndResultType
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
// CmpITensorOp 结果为 i1 张量，与输入张量类型不同，无法用 SameOperandsAndResultType，
// 需手写 inferReturnTypes 显式构造 i1 张量结果类型。
def CmpITensorOp : Op<MyDialect, "cmpi_tensor", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  SameTypeOperands
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

//===================== 张量常量 constant_tensor =====================
// 结果类型由 ElementsAttr 的 shape+elementType 推导，需手写 inferReturnTypes。
def ConstantTensorOp : Op<MyDialect, "constant_tensor", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  ConstantLike
]> {
  let arguments = (ins
    ElementsAttr:$value
  );
  let results = (outs MyTensorType:$res);
  let hasVerifier = 1;
  let assemblyFormat = "$value attr-dict `:` type($res)";
}

//===================== 张量 identity identity_tensor =====================
def IdentityTensorOp : Op<MyDialect, "identity_tensor", [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins
    MyTensorType:$in
  );
  let results = (outs MyTensorType:$res);
  let hasVerifier = 1;
  let assemblyFormat = "$in attr-dict `:` type($res)";
}

//===================== 张量 reshape reshape_tensor =====================
// shape 是运行时张量，无法静态推导结果类型；结果类型由 IR 显式指定，
// pattern 转换时也显式构建。不实现 InferTypeOpInterface。
def ReshapeTensorOp : Op<MyDialect, "reshape_tensor", [
  Pure
]> {
  let arguments = (ins
    MyTensorType:$data,
    MyTensorType:$shape
  );
  let results = (outs MyTensorType:$res);
  let hasVerifier = 1;
  let assemblyFormat = "$data `,` $shape attr-dict `:` type($data) `,` type($shape) `->` type($res)";
}

//===================== 张量 transpose transpose_tensor =====================
def TransposeTensorOp : Op<MyDialect, "transpose_tensor", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let arguments = (ins
    MyTensorType:$data,
    ArrayAttr:$perm
  );
  let results = (outs MyTensorType:$res);
  let hasVerifier = 1;
  let assemblyFormat = "$data `,` $perm attr-dict `:` type($data) `->` type($res)";
}

//===================== 张量 reduce_sum reduce_sum_tensor =====================
def ReduceSumTensorOp : Op<MyDialect, "reduce_sum_tensor", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let arguments = (ins
    MyTensorType:$data,
    ArrayAttr:$axes,
    BoolAttr:$keep_dims
  );
  let results = (outs MyTensorType:$res);
  let hasVerifier = 1;
  let assemblyFormat = "$data `,` $axes `,` $keep_dims attr-dict `:` type($data) `->` type($res)";
}

// ===================== 循环算子：对标 ONNX Loop =====================
// 注意：本算子不同于 mydialect.td 中既有的标量 ForOp/YieldOp，必须使用
// 独立的 def 名称与 mnemonic，否则 TableGen 会报 "def already exists"。
// 设计：
//   - maxTripCount(i64) 最大迭代次数，initCond(i1) 初始条件
//   - initArgs 循环携带张量初始值，results 为其最终值
//   - body 单块 region，隐式终止符 LoopYieldOp 产出 (newCond, newCarryArgs)
// 说明：不能用 SingleBlockImplicitTerminator —— 该 trait 要求终止符能用
// 零操作数默认构建，而 LoopYieldOp 的 cond 是必需操作数。改用 SingleBlock，
// 终止符在 IR 中显式书写，由 LoopOp::verify() 校验其为 LoopYieldOp。
def LoopOp : Op<MyDialect, "loop", [
  Pure,
  SingleBlock
]> {
  let arguments = (ins
    I64:$maxTripCount,               // 最大迭代次数，标量 i64
    I1:$initCond,                    // 初始条件，标量 i1
    Variadic<MyTensorType>:$initArgs // 迭代 carry 初始张量参数
  );
  let results = (outs
    Variadic<MyTensorType>:$results
  );
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  // 汇编格式：mydialect.loop %m, %cond, %a, %b : (tensor...) -> (tensor...) { ... }
  let assemblyFormat = "$maxTripCount `,` $initCond `,` $initArgs attr-dict `:` type($initArgs) `->` type($results) $body";
}

// 循环体终止算子 loop_yield，产出 (newCond, newCarryArgs)
def LoopYieldOp : Op<MyDialect, "loop_yield", [
  Pure,
  Terminator,
  ParentOneOf<["LoopOp"]>
]> {
  let arguments = (ins
    I1:$cond,
    Variadic<MyTensorType>:$carryArgs
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$cond `,` $carryArgs attr-dict `:` type($cond) `,` type($carryArgs)";
}


#endif
