#ifndef MYOPT_PASS_H
#define MYOPT_PASS_H
#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mydialect {
std::unique_ptr<mlir::Pass> createMyOptPass();
void registerMyOptPass();
// --mydialect-fuse-elementwise：add/mul 张量链融合为 fused_elementwise
std::unique_ptr<mlir::Pass> createFusionElementwisePass();
// --mydialect-tile：对 elementwise scf.for 做 strip-mining 分块
std::unique_ptr<mlir::Pass> createTilePass();
// --mydialect-parallelize：把可安全并行的 elementwise scf.for 提升为 scf.forall
std::unique_ptr<mlir::Pass> createParallelPass();
// --mydialect-schedule：基本块级 list scheduling（FU 数量 + 延迟模型）
std::unique_ptr<mlir::Pass> createSchedulePass();
// --mydialect-place-device：以 hw_mul 为种子做 device placement，
// 把设备子图包进 mydialect.device_exec region
std::unique_ptr<mlir::Pass> createPlaceDevicePass();
// --mydialect-lower-device：按 target 落地 device_exec 边界
// （target=cpu 内联 region，其它值保留边界）
std::unique_ptr<mlir::Pass> createLowerDevicePass(std::string target = "cpu",
                                                  bool verbose = false);
} // namespace mydialect
#endif
