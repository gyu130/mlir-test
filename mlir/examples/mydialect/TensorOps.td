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

//===================== 张量逐元素减法 sub_tensor =====================
// 结构同 AddTensorOp / MulTensorOp；LowerMyDialectPrint 里同构处理
// （元素类型为整数 → arith.subi；浮点 → arith.subf）。
def SubTensorOp : Op<MyDialect, "sub_tensor", [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins
    MyTensorType:$lhs,
    MyTensorType:$rhs
  );
  let results = (outs MyTensorType:$res);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

//===================== 张量逐元素除法 div_tensor =====================
// 结构同 AddTensorOp / MulTensorOp；LowerMyDialectPrint 里同构处理
// （元素类型为整数 → arith.divis；浮点 → arith.divf）。
// 不做静态除零 verifier（逐元素除零是运行时语义），由编译器/运行时处理。
def DivTensorOp : Op<MyDialect, "div_tensor", [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins
    MyTensorType:$lhs,
    MyTensorType:$rhs
  );
  let results = (outs MyTensorType:$res);
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
 // ConstantLike
]> {
  let arguments = (ins
    ElementsAttr:$value
  );
  let results = (outs MyTensorType:$res);
  let hasVerifier = 1;
  let assemblyFormat = "$value attr-dict `:` type($res)";
  
  let extraClassDeclaration = [{
    ::mlir::LogicalResult getConstantValue(::mlir::Attribute &value);
  }];
}

//===================== 融合逐元素算子 fused_elementwise =====================
// 由 --mydialect-fuse-elementwise 把 add_tensor/mul_tensor 的
// producer-consumer 链融合而成：变参输入张量 + 单块 region 描述逐元素
// 标量计算（第 i 个 block 参数对应第 i 个输入张量的当前元素），
// 由 FusedYieldOp 产出标量结果。
// 设计同 LoopOp：不能用 SingleBlockImplicitTerminator（FusedYieldOp 有
// 必需操作数，无法零操作数默认构造），终止符在 IR 中显式书写，
// 由 FusedElementwiseOp::verify() 校验。
def FusedElementwiseOp : Op<MyDialect, "fused_elementwise", [
  Pure,
  SingleBlock
]> {
  let arguments = (ins
    Variadic<MyTensorType>:$inputs
  );
  let results = (outs MyTensorType:$res);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($res) $body";
}

// fused_elementwise 的 region 终止符：产出当前元素的标量计算结果
// value 用 AnyType：标量元素类型（i32/f32/...），由 FusedElementwiseOp
// 的 verifier 校验其与结果张量元素类型一致
def FusedYieldOp : Op<MyDialect, "fused_yield", [
  Pure,
  Terminator,
  ParentOneOf<["FusedElementwiseOp"]>
]> {
  let arguments = (ins AnyType:$value);
  let results = (outs);
  let assemblyFormat = "$value attr-dict `:` type($value)";
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

//===================== 张量逐元素 Relu relu_tensor =====================
// 逐元素 max(0, x)。LowerMyDialectPrint 里展开为循环 nest + 标量：
//   整数: arith.cmpi sgt(x, 0) + arith.select(x, 0, flag)
//   浮点: arith.cmpi ogt(x, 0.0) + arith.select(x, 0.0, flag)
// 注：本方言的 Arith dialect 没有 arith.maxsi/maxf，所以用 cmpi+select 组合。
def ReluTensorOp : Op<MyDialect, "relu_tensor", [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins MyTensorType:$x);
  let results = (outs MyTensorType:$res);
  let assemblyFormat = "$x attr-dict `:` type($res)";
}

//===================== 张量矩阵乘法 matmul_tensor =====================
// 语义：Y[m,n] = Σ_k A[m,k] * B[k,n]，A: [M,K], B: [K,N] → Y: [M,N]。
// 简化实现：先要求 rank=2（不支持 batched matmul），静态 shape。
// 结果 shape 由 A.shape[0] + B.shape[1] 推导，需手写 inferReturnTypes。
def MatMulTensorOp : Op<MyDialect, "matmul_tensor", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let arguments = (ins
    MyTensorType:$A,
    MyTensorType:$B
  );
  let results = (outs MyTensorType:$Y);
  let hasVerifier = 1;
  let assemblyFormat = "$A `,` $B attr-dict `:` type($A) `,` type($B) `->` type($Y)";
}

//===================== 张量 reduce_max reduce_max_tensor =====================
// 与 reduce_sum_tensor 同构，但归约用 max 而非 sum。
def ReduceMaxTensorOp : Op<MyDialect, "reduce_max_tensor", [
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

//===================== 张量逐元素 exp exp_tensor =====================
// 逐元素 exp(x)。LowerMyDialectPrint 里展开成循环 nest + math.exp。
def ExpTensorOp : Op<MyDialect, "exp_tensor", [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins MyTensorType:$x);
  let results = (outs MyTensorType:$res);
  let assemblyFormat = "$x attr-dict `:` type($res)";
}

//===================== 张量沿 axis 拼接 concat_tensor =====================
// inputs 沿 axis 拼接成更大的张量。所有输入 shape 必须在除 axis 外的维度一致。
// 结果 shape 推导：axis 维 = Σ inputs[*].shape[axis]，其余维 = inputs[0].shape。
def ConcatTensorOp : Op<MyDialect, "concat_tensor", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let arguments = (ins
    Variadic<MyTensorType>:$inputs,
    I64Attr:$axis
  );
  let results = (outs MyTensorType:$res);
  let hasVerifier = 1;
  let assemblyFormat = "$inputs `,` $axis attr-dict `:` type($inputs) `->` type($res)";
}

//===================== 基础 2D 卷积 conv2d_tensor =====================
// 简化 NCHW 实现：
//   input:  [N, C_in, H, W]
//   weight: [C_out, C_in, kH, kW]  （group=1）
//   output: [N, C_out, OH, OW]
// OH = (H + pad_top + pad_bot - dil_h*(kH-1) - 1)/stride_h + 1
// OW = (W + pad_left + pad_right - dil_w*(kW-1) - 1)/stride_w + 1
// 仅支持 group=1；无 bias；属性 pads=[top,left,bottom,right], strides=[h,w], dilations=[h,w]
// 结果 shape 推导由手写 inferReturnTypes 实现。
def Conv2dTensorOp : Op<MyDialect, "conv2d_tensor", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let arguments = (ins
    MyTensorType:$input,
    MyTensorType:$weight,
    OptionalAttr<ArrayAttr>:$pads,
    DefaultValuedAttr<ArrayAttr, "{}">:$strides,
    DefaultValuedAttr<ArrayAttr, "{}">:$dilations
  );
  let results = (outs MyTensorType:$output);
  let hasVerifier = 1;
  let assemblyFormat = "$input `,` $weight attr-dict `:` type($input) `,` type($weight) `->` type($output)";
}

//===================== Softmax softmax_tensor =====================
// softmax(x, axis) = exp(x - max(x,axis)) / sum(exp(x - max(x,axis)), axis)
// 简化实现：axis 为最后一维或任意指定维。结果 shape = 输入 shape
// （apply softmax 到 axis 维，其他维保持不变）。
// Lowering: 3 次扫描——先找 max(沿 axis)，再 exp + sum，最后 div。
def SoftmaxTensorOp : Op<MyDialect, "softmax_tensor", [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins
    MyTensorType:$x,
    I64Attr:$axis
  );
  let results = (outs MyTensorType:$res);
  let assemblyFormat = "$x `,` $axis attr-dict `:` type($res)";
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
    Variadic<MyTensorType>:$initArgs
  );
  let results = (outs
    Variadic<MyTensorType>:$results
  );
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = "$maxTripCount `,` $initCond `,` $initArgs attr-dict `:` type($initArgs) `->` type($results) $body";
}

// 循环体终止算子 loop_yield，产出 (newCond, newCarryArgs)
// cond 可以是 i1 标量 或 0维 tensor<i1>（ONNX 转换时 body 里可能是 tensor<i1> 或 MyTensorType<[],i1>）
def LoopYieldOp : Op<MyDialect, "loop_yield", [
  Pure,
  Terminator,
  ParentOneOf<["LoopOp"]>
]> {
  let arguments = (ins
    AnyType:$cond,
    Variadic<MyTensorType>:$carryArgs
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$cond `,` $carryArgs attr-dict `:` type($cond) `,` type($carryArgs)";
}


#endif
