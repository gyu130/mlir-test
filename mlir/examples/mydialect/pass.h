#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>

namespace mlir {
class ModuleOp;

}

namespace mydialect {

/// 创建LowerMyDialectPrintPass，降低 mydialect.print 算子
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerMyDialectPrintPass();
void populatePrintOpToLLVMPatterns(mlir::RewritePatternSet &patterns);

} // namespace mydialect
