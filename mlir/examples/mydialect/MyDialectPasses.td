#ifndef MYDIALECT_PASSES_TD
#define MYDIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerMyDialectPrint : Pass<"lower-mydialect-print", "mlir::ModuleOp"> {
  let summary = "Lower mydialect.print to llvm.call";
  let description = "Lower mydialect.print op into LLVM call to runtime print function";
  let options = [
    Option<"disableRuntimeDecl", "disable-runtime-decl", "bool",
           /*default=*/"false", "Do not emit runtime function declaration">
  ];
}

// ============ LowerHwMulPass：mydialect.hw_mul → arith.muli (SW) 或 llvm.call_intrinsic (HW) ============
def LowerHwMulPass : Pass<"lower-hw-mul", "mlir::ModuleOp"> {
  let summary = "Lower mydialect.hw_mul to arith.muli or LLVM intrinsic";
  let description = [{
    Lower mydialect.hw_mul hardware accelerator op into one of two paths:
      1. Software path (default): hw_mul → arith.muli for CPU JIT execution
      2. Hardware path (use-hardware-path=true): hw_mul → llvm.call_intrinsic
         for RISC-V custom instruction / ISS simulation
  }];
  let options = [
    Option<"useHardwarePath", "use-hardware-path", "bool",
           /*default=*/"false", "Use LLVM intrinsic path instead of arith.muli">
  ];
}

#endif

