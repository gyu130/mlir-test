#ifndef MYOPT_PASSES_TD
#define MYOPT_PASSES_TD
include "mlir/Pass/PassBase.td"

def MyOptPass : Pass<"my-opt", "mlir::ModuleOp"> {
  let summary = "Simple high level opt: redundant print remove + simple addi CSE";
  let description = "Simple high level opt: redundant print remove + simple addi CSE";
}
#endif

