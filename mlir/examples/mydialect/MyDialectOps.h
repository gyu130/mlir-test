#pragma once
#include "MyDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Op 类声明已由 MyDialect.h 中的 `#define GET_OP_CLASSES` +
// `#include "mydialect-opdefs.h.inc"` 提供，此处不再重复 include，
// 避免重复定义（mydialect-opdefs.h.inc 无 include guard）。
