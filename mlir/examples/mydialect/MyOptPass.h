#ifndef MYOPT_PASS_H
#define MYOPT_PASS_H
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mydialect {
std::unique_ptr<mlir::Pass> createMyOptPass();
void registerMyOptPass();
} // namespace mydialect
#endif
