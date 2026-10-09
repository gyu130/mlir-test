#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>

namespace mlir {
class ModuleOp;
class RewritePatternSet;
}

namespace mydialect {
/// 创建 LowerMyDialectPrintPass（TableGen 生成）
std::unique_ptr<mlir::Pass> createLowerMyDialectPrint();

/// 统一注册所有 MyDialect → 下层IR 的 patterns
void populateMyDialectToLowerPatterns(mlir::RewritePatternSet &patterns);

// 保留旧接口兼容旧调用
inline void populatePrintOpToLLVMPatterns(mlir::RewritePatternSet &patterns) {
    populateMyDialectToLowerPatterns(patterns);
}

void registerConstantFoldMyDialectPass();
std::unique_ptr<mlir::Pass> createConstantFoldMyDialectPass();
} // namespace mydialect

