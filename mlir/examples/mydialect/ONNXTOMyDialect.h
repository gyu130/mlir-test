#ifndef ONNX_TO_MYDIALECT_H
#define ONNX_TO_MYDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

// TableGen 生成的 OnnxDialect 类声明（含构造函数、initialize() 声明等）
#include "onnx.h.inc"

// TableGen 生成的 ONNX op 类声明
// 注意：onnx-opdefs.h.inc 使用 GET_OP_CLASSES 作为 guard
#define GET_OP_CLASSES
#include "onnx-opdefs.h.inc"

namespace mlir {
// ONNX → MyDialect 转换 pass 工厂函数
std::unique_ptr<mlir::Pass> createConvertONNXToMyDialectPass();
} // namespace mlir

#endif // ONNX_TO_MYDIALECT_H
