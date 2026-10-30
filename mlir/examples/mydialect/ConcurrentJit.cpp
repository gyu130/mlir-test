#include "ConcurrentJit.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include <mutex>
#include <thread>
#include <future>
#include <vector>

using namespace mlir;
using namespace llvm;

namespace mydialect {

static std::mutex gRuntimeIoMutex;

static void jitPrintI32(int32_t val) {
  std::lock_guard<std::mutex> lock(gRuntimeIoMutex);
  llvm::outs() << "[JIT-PRINT] " << val << "\n";
}

// 全局符号表，给每个EE注册
static llvm::StringMap<void*> gExternSymbols = {
  {"print", (void*)jitPrintI32}
};

llvm::Expected<int64_t> runSingleJitModule(OwningOpRef<ModuleOp> module, int optLevel) {
  ExecutionEngineOptions opts;
  // LLVM 22 中 CodeGenOptLevel 是 enum class，用 CodeGenOpt::getLevel 转换
  // （范围外返回 nullopt：0=None 1=Less 2=Default 3=Aggressive）
  if (auto level = llvm::CodeGenOpt::getLevel(optLevel)) {
    opts.jitCodeGenOptLevel = *level;
  }

  // 【重点】这个版本 create 只接受 op + options，符号要后续注册
  auto engineOrErr = mlir::ExecutionEngine::create(module.get(), opts);
  if (!engineOrErr) {
    return engineOrErr.takeError();
  }
  auto engine = std::move(*engineOrErr);

  // 注册外部符号：registerSymbols 接收一个构建 SymbolMap 的回调
  engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    for (auto &entry : gExternSymbols) {
      symbolMap[interner(entry.getKey())] = {
          llvm::orc::ExecutorAddr::fromPtr(entry.getValue()),
          llvm::JITSymbolFlags::Exported};
    }
    // 项目 lowering 生成/引用的运行时打印函数
    symbolMap[interner("my_runtime_print_i32")] = {
        llvm::orc::ExecutorAddr::fromPtr(jitPrintI32),
        llvm::JITSymbolFlags::Exported};
    return symbolMap;
  });

  // === invokePacked 坑：必须传指针数组 MutableArrayRef<void*> ===
  int64_t retVal = 0;
  void *args[] = {&retVal};
  MutableArrayRef<void*> argRef(args);

  Error invokeErr = engine->invokePacked("main", argRef);
  if (invokeErr) {
    return std::move(invokeErr);
  }
  return retVal;
}

LogicalResult runConcurrentJit(ArrayRef<OwningOpRef<ModuleOp>> modules,
                               unsigned maxThreadNum,
                               int jitOptLevel) {
  std::vector<std::thread> workers;
  std::vector<std::promise<llvm::Expected<int64_t>>> promises;
  std::vector<std::future<llvm::Expected<int64_t>>> futures;

  size_t total = modules.size();
  size_t idx = 0;
  bool anyFailed = false;

  while (idx < total) {
    size_t batchEnd = std::min(idx + maxThreadNum, total);
    size_t batchSize = batchEnd - idx;
    promises.clear();
    futures.clear();
    workers.clear();
    // 关键：先 reserve 再取 promise 的地址，否则后续 emplace_back 触发
    // vector 重分配会搬走 promise，使 lambda 捕获的指针悬空，
    // p.set_value() 写入已释放内存 → 堆损坏（表现为 clone/region 断言或段错误）
    promises.reserve(batchSize);
    futures.reserve(batchSize);
    workers.reserve(batchSize);

    for (size_t i = idx; i < batchEnd; ++i) {
      OwningOpRef<ModuleOp> clonedMod = modules[i].get().clone();
      promises.emplace_back();
      futures.push_back(promises.back().get_future());

      auto *p = &promises.back();
      workers.emplace_back([m = std::move(clonedMod), opt = jitOptLevel, p]() mutable {
        auto res = runSingleJitModule(std::move(m), opt);
        p->set_value(std::move(res));
      });
    }

    for (auto &t : workers) {
      t.join();
    }

    for (auto &fut : futures) {
      auto res = fut.get();
      if (!res) {
        llvm::errs() << "Concurrent JIT task error: " << toString(res.takeError()) << "\n";
        anyFailed = true;
      } else {
        llvm::outs() << "Concurrent JIT task success, return value = " << *res << "\n";
      }
    }
    idx = batchEnd;
  }

  return anyFailed ? failure() : success();
}

} // namespace mydialect

