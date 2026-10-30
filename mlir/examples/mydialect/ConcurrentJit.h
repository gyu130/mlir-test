#ifndef MYDIALECT_CONCURRENTJIT_H
#define MYDIALECT_CONCURRENTJIT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace mydialect {

llvm::Expected<int64_t> runSingleJitModule(mlir::OwningOpRef<mlir::ModuleOp> module, int optLevel = 2);

mlir::LogicalResult runConcurrentJit(llvm::ArrayRef<mlir::OwningOpRef<mlir::ModuleOp>> modules,
                                     unsigned maxThreadNum,
                                     int jitOptLevel = 2);

} // namespace mydialect

#endif

