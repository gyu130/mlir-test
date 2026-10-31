#ifndef MYDIALECT_HWOPS_TD
#define MYDIALECT_HWOPS_TD

include "mlir/IR/OpBase.td"
include "mydialect.td"

def HwMulOp : Op<MyDialect, "hw_mul", [Pure]> {
  let summary = "Hardware accelerator i32 multiplication primitive";
  let description = [{
    Hardware multiplication operator for target accelerator.
    纯标量 op（i32 -> i32），硬件指令级别的乘法操作。
    两条 lowering 路径：
      1. Software path: lower to arith.muli for CPU JIT execution
      2. Hardware path: lower to custom llvm intrinsic for RISC-V custom instruction / ISS simulation
  }];

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$res);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

// ============ 异构部署：host/device 执行边界 ============
// device_exec 把分配到设备（如硬件加速器）的算子包成一个 region，
// 构成 host ↔ device 的流水线边界；region 内产生的、region 外仍要用的值
// 通过 device_yield 提升为该 op 的结果。
def DeviceExecOp : Op<MyDialect, "device_exec", [SingleBlock]> {
  let summary = "Host/device execution boundary for heterogeneous placement";
  let description = [{
    Groups ops placed on a target device into a single-block region.
    Values defined inside the region and used outside are yielded by
    mydialect.device_yield and become results of this op.
    下游 pass 可按 target 选择：
      1. cpu: 内联 region（设备算子退回 host 执行）
      2. device: 保留边界，交给设备后端/kernel 生成
  }];

  let arguments = (ins StrAttr:$device);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let assemblyFormat = "$device attr-dict `:` type($results) $body";
}

def DeviceYieldOp : Op<MyDialect, "device_yield", [
  Pure,
  Terminator,
  ParentOneOf<["DeviceExecOp"]>
]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
  let assemblyFormat = "$values attr-dict `:` type($values)";
}

#endif // MYDIALECT_HWOPS_TD

