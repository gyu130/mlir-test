#ifndef MYOPT_PASS_H
#define MYOPT_PASS_H
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mydialect {
std::unique_ptr<mlir::Pass> createMyOptPass();
void registerMyOptPass();
// --mydialect-fuse-elementwise：add/mul 张量链融合为 fused_elementwise
std::unique_ptr<mlir::Pass> createFusionElementwisePass();
// --mydialect-tile：对 elementwise scf.for 做 strip-mining 分块
std::unique_ptr<mlir::Pass> createTilePass();
} // namespace mydialect
#endif
