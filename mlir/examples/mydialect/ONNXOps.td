#ifndef ONNX_OPS_TD
#define ONNX_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//============================= ONNX stub Dialect =============================
// Minimal ONNX-like dialect stub for ONNX -> MyDialect conversion demo.
// All ops use MLIR builtin tensor types; no external onnx-mlir dependency.
def OnnxDialect : Dialect {
  let name = "onnx";
  let summary = "Stub ONNX dialect for ONNX->MyDialect conversion demo";
  let description = "Minimal ONNX-like dialect using builtin tensor types";
  let cppNamespace = "::mlir::onnx";
}

// ONNX Add: A, B -> C (element-wise add)
def ONNXAddOp : Op<OnnxDialect, "Add", [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins AnyTensor:$a, AnyTensor:$b);
  let results = (outs AnyTensor:$c);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($a)";
}

// ONNX Mul: A, B -> C (element-wise mul)
def ONNXMulOp : Op<OnnxDialect, "Mul", [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins AnyTensor:$a, AnyTensor:$b);
  let results = (outs AnyTensor:$c);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($a)";
}

// ONNX Equal: A, B -> C (element-wise equality compare, C is i1 tensor)
def ONNXEqualOp : Op<OnnxDialect, "Equal", [
  Pure,
  SameTypeOperands
]> {
  let arguments = (ins AnyTensor:$a, AnyTensor:$b);
  let results = (outs AnyTensor:$c);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($a) `->` type($c)";
}

// ONNX Constant: ElementsAttr -> output
// 用 hasCustomAssemblyFormat=1 自己写 parser，绕过 ODS assemblyFormat 强制 attr-dict 的限制。
// onnx-mlir 输出不带额外属性（没有 onnx_node_name 等），而 ODS attr-dict 指令在 op 没有
// 其他 attribute 时会强制期待 `{}` 或内联属性，导致 "expected ':'" 解析错误。
def ONNXConstantOp : Op<OnnxDialect, "Constant", [
  Pure
]> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$output);
  let hasCustomAssemblyFormat = 1;
}

// ONNX Identity: input -> output
def ONNXIdentityOp : Op<OnnxDialect, "Identity", [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

// ONNX Reshape: data, shape -> output
def ONNXReshapeOp : Op<OnnxDialect, "Reshape", [Pure]> {
  let arguments = (ins AnyTensor:$data, AnyTensor:$shape);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$data `,` $shape attr-dict `:` type($data) `,` type($shape) `->` type($output)";
}

// ONNX Transpose: data, ArrayAttr perm -> output
def ONNXTransposeOp : Op<OnnxDialect, "Transpose", [Pure]> {
  let arguments = (ins AnyTensor:$data, OptionalAttr<ArrayAttr>:$perm);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$data (`,` $perm^)? attr-dict `:` type($data) `->` type($output)";
}

// ONNX ReduceSum: data, axes, keepdims -> output
// Note: ONNX 13+ has axes as input tensor; this stub uses ArrayAttr
// to match mydialect::ReduceSumTensorOp.
def ONNXReduceSumOp : Op<OnnxDialect, "ReduceSum", [Pure]> {
  let arguments = (ins AnyTensor:$data, OptionalAttr<ArrayAttr>:$axes,
                   DefaultValuedAttr<BoolAttr, "false">:$keepdims);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$data (`,` $axes^)? (`,` $keepdims^)? attr-dict `:` type($data) `->` type($output)";
}

//============================= ONNX Loop =============================
// 对标 ONNX Loop：最大迭代次数 + 条件 + 循环携带张量，body 产出 (cond, carry)。
// 终止符 mnemonic 用 "Yield"（即 onnx.Yield），与真实 onnx-mlir 命名一致。
// ONNX Loop 的 body 采用块参数形式（iter, cond, carries），
// 测试可用通用形式 "onnx.Loop" ... ({ ^bb0(...): ... })。
// onnx.Yield：onns-mlir 的 ONNX Loop body 里所有值（cond + carries）
// 都是 tensor 类型，所以 $cond 也必须是 AnyTensor。
def ONNXLoopYieldOp : Op<OnnxDialect, "Yield", [
  Pure,
  Terminator
]> {
  let arguments = (ins AnyTensor:$cond, Variadic<AnyTensor>:$carryArgs);
  let results = (outs);
  let assemblyFormat = "attr-dict $cond `,` $carryArgs `:` type($cond) `,` type($carryArgs)";
}

// ONNX Loop：max_trip_count + cond + carries 输入都是 tensor。
// onnx-mlir 对 ONNX Loop 用泛型格式（有 region），所以 assemblyFormat 不影响解析。
def ONNXLoopOp : Op<OnnxDialect, "Loop", [
  Pure,
  SingleBlock
]> {
  let arguments = (ins
    AnyTensor:$max_trip_count,
    AnyTensor:$cond,
    Variadic<AnyTensor>:$carried_inputs
  );
  let results = (outs Variadic<AnyTensor>:$outputs);
  let regions = (region SizedRegion<1>:$body);
}

// onnx-mlir 生成的入口标记：generic 语法以 properties 形式携带 func 符号引用
// （"onnx.EntryPoint"() <{func = @main_graph}> : () -> ()），
// 因此必须声明 func 属性，否则报 "this operation does not support properties"。
def ONNXEntryPointOp : Op<OnnxDialect, "EntryPoint", []> {
  let arguments = (ins FlatSymbolRefAttr:$func);
  let results = (outs);
}

#endif // ONNX_OPS_TD
