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

// 覆盖 MyDialect 声明：补充自定义类型 parse/print 钩子
// TableGen 生成的 mydialect.h.inc 已定义 MyDialect；此处不可重定义。
// 改用 Dialect 的注册模板：给 MyDialect 加 NonDefault 的 parse/print 声明
namespace mydialect {
// WrappedI32Type 的前向声明（TableGen 类型类在下方 typedefs 中定义）
class WrappedI32Type;
} // namespace mydialect

// TableGen 生成的自定义类型声明（WrappedI32Type）
#define GET_TYPEDEF_CLASSES
#include "mydialect.typedefs.h.inc"

// TableGen 生成的 Op 完整类声明（-gen-op-decls 产物；
// 不定义 GET_OP_CLASSES 时只有前向声明）
#define GET_OP_CLASSES
#include "mydialect-opdefs.h.inc"



namespace mydialect {


  // 类型解析：parseType
//  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  // 类型打印：printType
//  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;


	
std::unique_ptr<mlir::Pass> createLowerMyDialectPrint();
void registerMyDialectPasses();
}
#endif
