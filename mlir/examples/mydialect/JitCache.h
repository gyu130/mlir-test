#ifndef MYDIALECT_JIT_CACHE_H
#define MYDIALECT_JIT_CACHE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mydialect {

// JitCache：模块级 JIT 编译结果缓存
//
// 缓存粒度：一个 mlir::ModuleOp → 一份 lowering 后的 LLVM bitcode
// key = SHA256(module 文本 canonicalized + 代码生成选项)
//
// 用法：
//   JitCache cache;
//   auto key = cache.computeKey(module);
//   if (auto *bc = cache.lookup(key)) {
//     // 命中：直接用 *bc 建 ExecutionEngine
//   } else {
//     // 未命中：lower module → LLVM bitcode，然后 cache.insert(key, bitcode)
//   }
//
// 线程安全：所有 public 方法都内部加锁；外部不需要自己锁。
class JitCache {
public:
  using Bitcode = std::vector<uint8_t>;

  // 按 ModuleOp 文本 + optLevel + triple 算 key（SHA256，取前 32 hex 字符）
  static std::string computeKey(mlir::ModuleOp module, int optLevel,
                                llvm::StringRef triple = "");

  // 设置磁盘持久化目录（空字符串 = 纯内存缓存）。
  // 设置后 lookup 先在内存找，再尝试读 <dir>/<key>.bc；
  // insert 会同时写内存和 <dir>/<key>.bc，跨进程复用。
  void setPersistDir(std::string dir);
  std::string getPersistDir() const;

  Bitcode *lookup(llvm::StringRef key);
  void insert(llvm::StringRef key, Bitcode bitcode);
  void erase(llvm::StringRef key);
  size_t size() const;
  void clear();

private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, Bitcode> cache_;
  std::string persistDir_;
};

// AotCompiler：把 MLIR module 编译为 .so 共享库
//
// 步骤：MLIR → (PassManager lowering) → LLVM bitcode
//       → llvm::TargetMachine::emitFile(".o") → 外部 clang 链接 .so
//       → 返回生成路径
struct AotCompilerOptions {
  int optLevel = 2;                       // 0..3
  std::string triple;                     // ""=宿主默认；"x86_64-unknown-linux-gnu" 等
  std::string outputDir = "/tmp/mydialect_aot";
  std::string soName;                     // ""=自动按 module 里第一个 func 命名
  std::string clangPath = "clang";        // 系统 clang 路径
  std::vector<std::string> linkLibs;      // 额外 -lxxx（如 "pthread"）
};

llvm::Expected<std::string> compileModuleToSharedLib(
    mlir::ModuleOp module, const AotCompilerOptions &opts);

// 加载并执行 .so 里的 "main" 符号（可选）；只给需要脱离 JIT 场景的调用方用
llvm::Expected<int64_t> loadAndRunSharedLib(llvm::StringRef soPath,
                                            llvm::StringRef entry = "main");

// === 增量编译 ===
// 模块内函数级缓存：func.func 单独 lowering+compile，只重编变更函数
// 当前实现粒度保守：直接用 JitCache 做 module 级 cache + AOT 为 .so，
// 增量机制留给 func.func 独立 lowering（后续在 LowerMyDialectPrint
// 里按函数拆分 module，再分别 compile）。

// 从 MLIR module lowering 到 LLVM bitcode（公开 helper）
llvm::Expected<std::vector<uint8_t>> mlirModuleToBitcode(mlir::ModuleOp module);
// 从 bitcode 重建 llvm::Module（新 LLVMContext）
std::unique_ptr<llvm::Module> bitcodeToLLVMModule(llvm::StringRef bitcodeData,
                                                  llvm::LLVMContext &ctx);

} // namespace mydialect

#endif // MYDIALECT_JIT_CACHE_H
