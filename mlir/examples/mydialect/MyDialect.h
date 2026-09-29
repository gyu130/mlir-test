#ifndef MYDIALECT_H
#define MYDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"


// TableGen 生成的 Dialect 声明（class MyDialect 等）
#include "mydialect.h.inc"

// TableGen 生成的 Op 完整类声明（-gen-op-decls 产物；
// 不定义 GET_OP_CLASSES 时只有前向声明）
#define GET_OP_CLASSES
#include "mydialect-opdefs.h.inc"



namespace mydialect {
void populatePrintOpToLLVMPatterns(mlir::RewritePatternSet &patterns);
#if 0
std::unique_ptr<mlir::Pass> createLowerMyDialectPrintPass();
#endif
}
#endif
