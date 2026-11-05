// JitCache.cpp：模块级 JIT 编译结果缓存 + AOT 共享库导出
//
// 架构：
//   JitCache         — 进程内内存级缓存，key = SHA256(module text)
//                      缓存 LLVM bitcode，命中时跳过 MLIR lowering
//   AotCompiler      — MLIR → LLVM IR (.ll) → llc -o .o → clang -shared
//                      （委托系统工具链，避开 LLVM TargetMachine API 差异）
//
// 缓存命中入口：ConcurrentJit 里用 ExecutionEngineOptions.llvmModuleBuilder
// 把已缓存的 bitcode 喂给 ExecutionEngine，完全跳过 MLIR lowering。

#include "JitCache.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SHA256.h"
#include "llvm/TargetParser/Host.h"
#include <cstdlib>
#include <mutex>

using namespace mlir;
using namespace llvm;

namespace mydialect {

// ============ SHA256 ============
static std::string sha256Hex(StringRef data) {
  // 算 key 时用 StringRef (data + size)；SHA256::hash 需要 ArrayRef<uint8_t>
  // StringRef::data() 返回 const char*，ArrayRef<uint8_t> 从它构造需要 cast
  SmallVector<uint8_t, 64> buf;
  buf.append(data.begin(), data.end());
  auto hash = SHA256::hash(ArrayRef<uint8_t>(buf.data(), buf.size()));
  std::string out;
  for (unsigned b : hash)
    out += llvm::utohexstr(b, /*LowerCase=*/true);
  return out;
}

// ============ JitCache ============
std::string JitCache::computeKey(mlir::ModuleOp module, int optLevel,
                                 StringRef triple) {
  std::string text;
  {
    raw_string_ostream os(text);
    module.print(os);
  }
  SmallString<256> keyBuf;
  keyBuf.append(text);
  keyBuf.push_back('\0');
  raw_svector_ostream os(keyBuf);
  os << "opt=" << optLevel << "\n";
  if (!triple.empty()) os << "triple=" << triple << "\n";
  return sha256Hex(keyBuf);
}

// 磁盘缓存文件名：<persistDir>/<key>.bc
static void cacheFilePath(StringRef dir, StringRef key, SmallString<256> &out) {
  out = dir;
  SmallString<256> name(key);
  name += ".bc";
  sys::path::append(out, name);
}

void JitCache::setPersistDir(std::string dir) {
  std::lock_guard<std::mutex> lock(mu_);
  if (dir.empty()) {
    persistDir_.clear();
    return;
  }
  sys::fs::create_directories(dir);
  persistDir_ = std::move(dir);
}

std::string JitCache::getPersistDir() const {
  std::lock_guard<std::mutex> lock(mu_);
  return persistDir_;
}

JitCache::Bitcode *JitCache::lookup(StringRef key) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = cache_.find(key.str());
  if (it != cache_.end()) return &it->second;
  // 内存未命中：尝试从磁盘持久化缓存重建（跨进程复用）
  if (persistDir_.empty()) return nullptr;
  SmallString<256> path;
  cacheFilePath(persistDir_, key, path);
  auto bufOrErr = MemoryBuffer::getFile(path);
  if (!bufOrErr) return nullptr;
  StringRef data = (*bufOrErr)->getBuffer();
  Bitcode bc(data.begin(), data.end());
  auto insIt = cache_.emplace(key.str(), std::move(bc)).first;
  return &insIt->second;
}

void JitCache::insert(StringRef key, Bitcode bitcode) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!persistDir_.empty()) {
    SmallString<256> path;
    cacheFilePath(persistDir_, key, path);
    std::error_code ec;
    raw_fd_ostream os(path, ec, sys::fs::OF_None);
    if (!ec) {
      os.write(reinterpret_cast<const char *>(bitcode.data()), bitcode.size());
      os.close();
    } else {
      llvm::errs() << "[JitCache] write " << path << " failed: "
                   << ec.message() << "\n";
    }
  }
  cache_[key.str()] = std::move(bitcode);
}

void JitCache::erase(StringRef key) {
  std::lock_guard<std::mutex> lock(mu_);
  cache_.erase(key.str());
  if (persistDir_.empty()) return;
  SmallString<256> path;
  cacheFilePath(persistDir_, key, path);
  sys::fs::remove(path);
}

size_t JitCache::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return cache_.size();
}

void JitCache::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  cache_.clear();
}

// ============ 公开 helper ============
llvm::Expected<std::vector<uint8_t>>
mlirModuleToBitcode(mlir::ModuleOp module) {
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto llvmModOrErr = mlir::translateModuleToLLVMIR(module, *ctx);
  if (!llvmModOrErr)
    return createStringError(inconvertibleErrorCode(),
                             "translateModuleToLLVMIR failed");
  llvm::Module *llvmMod = llvmModOrErr.get();
  SmallVector<char, 0> buf;
  raw_svector_ostream os(buf);
  WriteBitcodeToFile(*llvmMod, os);
  return std::vector<uint8_t>(buf.begin(), buf.end());
}

std::unique_ptr<llvm::Module>
bitcodeToLLVMModule(StringRef bitcodeData, llvm::LLVMContext &ctx) {
  auto modOrErr = parseBitcodeFile(
      MemoryBuffer::getMemBufferCopy(bitcodeData, "cached-bc")->getMemBufferRef(),
      ctx);
  if (!modOrErr) {
    llvm::errs() << "[JitCache] bitcode parse failed: "
                 << toString(modOrErr.takeError()) << "\n";
    return nullptr;
  }
  return std::move(*modOrErr);
}

// ============ AOT 编译 ============
// MLIR → LLVM IR (.ll) → clang/llc 编译 .o → clang -shared .so
llvm::Expected<std::string>
compileModuleToSharedLib(mlir::ModuleOp module, const AotCompilerOptions &opts) {
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto llvmModOrErr = mlir::translateModuleToLLVMIR(module, *ctx);
  if (!llvmModOrErr)
    return createStringError(inconvertibleErrorCode(),
                             "translateModuleToLLVMIR failed");
  llvm::Module *llvmMod = llvmModOrErr.get();

  // 输出目录
  if (sys::fs::create_directories(opts.outputDir))
    return createStringError(inconvertibleErrorCode(),
                             "mkdir " + opts.outputDir + " failed");

  // 自动生成 .so 名
  std::string soName = opts.soName;
  if (soName.empty()) {
    soName = "mydialect.so";
    for (auto &fn : llvmMod->functions()) {
      if (!fn.isDeclaration() && !fn.hasLocalLinkage()) {
        soName = fn.getName().str() + ".so";
        break;
      }
    }
  }
  SmallString<256> soPath(opts.outputDir);
  soPath += "/" + soName;
  SmallString<256> llPath(soPath);
  sys::path::replace_extension(llPath, "ll");
  SmallString<256> objPath(soPath);
  sys::path::replace_extension(objPath, "o");

  // 写 .ll
  std::error_code ec;
  raw_fd_ostream llOS(llPath, ec, sys::fs::OF_None);
  if (ec) return createStringError(inconvertibleErrorCode(),
                                    "open " + llPath + ": " + ec.message());
  llvmMod->print(llOS, /*AAW=*/nullptr);
  llOS.close();

  // 找 llc / clang：ExecuteAndWait 不做 PATH 解析，必须先解析成绝对路径
  // 工具解析顺序：同目录（与 mydriver 同构建树，LLVM 版本一致）→ PATH。
  // 这点很关键：本仓库 LLVM 22 生成的 IR 含 GEP nuw 等新语法，
  // 系统老版本 clang（如 17）无法解析，必须用构建树里同版本的 clang/llc。
  auto resolveProgram = [](StringRef name) -> StringRef {
    static std::vector<std::string> owned; // 保持 StringRef 生命周期
    // 显式路径直接使用
    if (sys::path::is_absolute(name) && sys::fs::exists(name)) {
      owned.push_back(name.str());
      return StringRef(owned.back());
    }
    SmallString<256> exePath;
    std::error_code rpEC = sys::fs::real_path("/proc/self/exe", exePath);
    if (!rpEC) {
      sys::path::remove_filename(exePath);           // .../build_mlir/bin/mydriver → .../bin
      sys::path::append(exePath, sys::path::filename(name)); // → .../bin/clang
      if (sys::fs::exists(exePath)) {
        owned.emplace_back(exePath.str().str());
        return StringRef(owned.back());
      }
    }
    auto found = sys::findProgramByName(name);
    if (!found) return StringRef();
    owned.push_back(std::move(*found));
    return StringRef(owned.back());
  };

  StringRef clangPath = resolveProgram(opts.clangPath);
  if (clangPath.empty())
    return createStringError(inconvertibleErrorCode(),
                             "cannot find clang: " + opts.clangPath);
  StringRef llcPath = resolveProgram("llc");

  std::string optStr = "-O" + std::to_string(std::clamp(opts.optLevel, 0, 3));
  SmallVector<StringRef, 8> compileArgs;
  if (!llcPath.empty()) {
    // llc 路径
    compileArgs = {llcPath, optStr, "-filetype=obj", "-o", objPath.c_str(),
                   llPath.c_str()};
  } else {
    // clang 当 llc 用：clang -c -O2 -o xxx.o -x ir xxx.ll
    compileArgs = {clangPath, "-c", optStr, "-o", objPath.c_str(),
                   "-x", "ir", llPath.c_str()};
  }

  std::string errMsg;
  int code = sys::ExecuteAndWait(compileArgs[0], compileArgs,
                                 /*Env=*/std::nullopt,
                                 /*Redirects=*/{},
                                 /*SecondsToWait=*/0, /*MemoryLimit=*/0,
                                 &errMsg);
  if (code != 0)
    return createStringError(inconvertibleErrorCode(),
                             "compile failed (exit=" + std::to_string(code) +
                                 "): " + errMsg);

  // clang -shared
  SmallVector<StringRef, 8> linkArgs = {clangPath, "-shared",
                                        "-o", soPath.c_str(), objPath.c_str()};
  for (auto &l : opts.linkLibs) {
    linkArgs.push_back("-l");
    linkArgs.push_back(l);
  }
  linkArgs.push_back("-lm");

  errMsg.clear();
  code = sys::ExecuteAndWait(linkArgs[0], linkArgs,
                             /*Env=*/std::nullopt,
                             /*Redirects=*/{},
                             /*SecondsToWait=*/0, /*MemoryLimit=*/0,
                             &errMsg);
  if (code != 0)
    return createStringError(inconvertibleErrorCode(),
                             "link failed (exit=" + std::to_string(code) +
                                 "): " + errMsg);

  return soPath.str().str();
}

llvm::Expected<int64_t> loadAndRunSharedLib(StringRef soPath, StringRef entry) {
  std::string err;
  auto lib = sys::DynamicLibrary::getPermanentLibrary(
      soPath.str().c_str(), &err);
  if (!lib.isValid())
    return createStringError(inconvertibleErrorCode(),
                             "dlopen " + soPath.str() + ": " + err);
  using MainFn = int64_t (*)();
  auto fn = reinterpret_cast<MainFn>(lib.getAddressOfSymbol(entry.str().c_str()));
  if (!fn)
    return createStringError(inconvertibleErrorCode(),
                             "symbol not found: " + entry.str());
  return fn();
}
} // namespace mydialect
