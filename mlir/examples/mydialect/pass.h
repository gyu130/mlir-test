#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>

namespace mlir {
class ModuleOp;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace mydialect {
/// 创建 LowerMyDialectPrintPass（TableGen 生成）
std::unique_ptr<mlir::Pass> createLowerMyDialectPrint();

/// 注册所有 MyDialect → 下层 IR(scf/memref/arith/llvm) 的 patterns。
/// converter：DialectConversion 使用的类型转换器，
/// 每个 OpConversionPattern 构造时都绑定它，MLIR 22 的转换驱动
/// 直接从 pattern 获取 converter（不再传给 applyPartialConversion）。
void populateMyDialectToLowerPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::TypeConverter &converter);

void registerConstantFoldMyDialectPass();
std::unique_ptr<mlir::Pass> createConstantFoldMyDialectPass();
} // namespace mydialect
