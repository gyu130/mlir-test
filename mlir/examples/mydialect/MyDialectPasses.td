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

#endif

