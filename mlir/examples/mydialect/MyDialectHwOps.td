#ifndef MYDIALECT_HWOPS_TD
#define MYDIALECT_HWOPS_TD

include "mlir/IR/OpBase.td"
include "mydialect.td"

def HwMulOp : Op<MyDialect, "hw_mul"> {
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

#endif // MYDIALECT_HWOPS_TD

