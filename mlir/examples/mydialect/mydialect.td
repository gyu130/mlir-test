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



#endif

