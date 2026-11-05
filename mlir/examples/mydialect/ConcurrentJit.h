#ifndef MYDIALECT_CONCURRENTJIT_H
#define MYDIALECT_CONCURRENTJIT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace mydialect {

llvm::Expected<int64_t> runSingleJitModule(mlir::OwningOpRef<mlir::ModuleOp> module, int optLevel = 2);

mlir::LogicalResult runConcurrentJit(llvm::ArrayRef<mlir::OwningOpRef<mlir::ModuleOp>> modules,
                                     unsigned maxThreadNum,
                                     int jitOptLevel = 2);

// 带 JitCache 的 JIT 入口：
//   - 首次：lower module → LLVM bitcode → 缓存 → ExecutionEngine
//   - 命中：直接从 bitcode 重建 ExecutionEngine（跳过 MLIR lowering）
//   - verbose=true 打印 "jit-cache HIT / MISS" 与耗时
llvm::Expected<int64_t> runJitWithCache(mlir::OwningOpRef<mlir::ModuleOp> module,
                                         int optLevel = 2,
                                         bool verbose = false);

// 设置 JIT 缓存的磁盘持久化目录（空串 = 纯内存缓存）。
// 设置后 bitcode 会落盘为 <dir>/<key>.bc，下次进程启动可直接命中。
void setJitCachePersistDir(llvm::StringRef dir);

} // namespace mydialect

#endif

