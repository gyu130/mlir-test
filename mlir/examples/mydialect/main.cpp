#include "MyDialect.h"
#include "pass.h"
#include "MyOptPass.h"
#include "SimpleCParser.h"
#include "ONNXToMyDialect.h"
#include "LowerHwMulPass.h"
#include "OpCountCycleEstimate.h"
#include "ConcurrentJit.h"
#include "JitCache.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

// ============ JIT 任意签名支持：参数描述与打包 ============
// 在 lowering 前从原始 func 签名捕获参数/结果信息；
// lowering 后按 memref descriptor 展开约定（ptr, ptr, offset, sizes..., strides...）
// 构造 invokePacked 的 void* 参数数组。
namespace {
struct ArgDesc {
  enum class Kind { Int, F32, F64, MemRef } kind;
  unsigned intWidth = 0;             // Int 位宽（index 按 64 处理）
  llvm::SmallVector<int64_t> shape;  // MemRef 静态 shape
  char elemKind = 'i';               // MemRef 元素类别：'i' 整型 / 'f' 浮点
  unsigned elemWidth = 32;           // MemRef 元素位宽
};

// 把单个 MLIR 类型翻译为 ArgDesc；不支持返回 false
static bool describeType(mlir::Type ty, ArgDesc &d) {
  if (auto it = llvm::dyn_cast<mlir::IntegerType>(ty)) {
    if (it.getWidth() > 64) return false;
    d.kind = ArgDesc::Kind::Int; d.intWidth = it.getWidth(); return true;
  }
  if (ty.isIndex()) { d.kind = ArgDesc::Kind::Int; d.intWidth = 64; return true; }
  if (ty.isF32()) { d.kind = ArgDesc::Kind::F32; return true; }
  if (ty.isF64()) { d.kind = ArgDesc::Kind::F64; return true; }

  mlir::Type elemTy;
  if (auto mt = llvm::dyn_cast<mlir::MemRefType>(ty)) {
    if (!mt.hasStaticShape()) return false;
    elemTy = mt.getElementType();
    d.shape.assign(mt.getShape().begin(), mt.getShape().end());
  } else if (auto tt = llvm::dyn_cast<mydialect::MyTensorType>(ty)) {
    elemTy = tt.getElementType();
    for (auto a : tt.getShape())
      d.shape.push_back(llvm::cast<mlir::IntegerAttr>(a).getInt());
  } else {
    return false;
  }
  d.kind = ArgDesc::Kind::MemRef;
  if (auto ie = llvm::dyn_cast<mlir::IntegerType>(elemTy)) {
    d.elemKind = 'i'; d.elemWidth = ie.getWidth();
  } else if (elemTy.isIndex()) {
    d.elemKind = 'i'; d.elemWidth = 64;
  } else if (elemTy.isF32()) {
    d.elemKind = 'f'; d.elemWidth = 32;
  } else if (elemTy.isF64()) {
    d.elemKind = 'f'; d.elemWidth = 64;
  } else {
    return false;
  }
  return true;
}

static bool captureSignature(mlir::func::FuncOp func,
                             llvm::SmallVectorImpl<ArgDesc> &args,
                             llvm::SmallVectorImpl<ArgDesc> &results) {
  for (auto ty : func.getArgumentTypes()) {
    ArgDesc d;
    if (!describeType(ty, d)) return false;
    args.push_back(std::move(d));
  }
  for (auto ty : func.getResultTypes()) {
    ArgDesc d;
    if (!describeType(ty, d)) return false;
    results.push_back(std::move(d));
  }
  return true;
}

// 标量参数/结果的稳定存储（地址在 invoke 前不得移动）
struct ScalarSlot { alignas(8) char data[8]; };
// 输入 memref 的 descriptor 槽（指针会被取地址放入 invokeArgs）
struct MemRefSlot {
  void *allocated = nullptr;
  void *aligned = nullptr;
  int64_t offset = 0;
  llvm::SmallVector<int64_t, 4> sizes;
  llvm::SmallVector<int64_t, 4> strides;
};

static int64_t numElementsOf(llvm::ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// 按元素类别/位宽向 buf 第 i 个元素写入 iota 值 (i+1)
static void fillIota(char *buf, int64_t i, char kind, unsigned width) {
  if (kind == 'f') {
    if (width == 32) { float v = (float)(i + 1); std::memcpy(buf + i * 4, &v, 4); }
    else { double v = (double)(i + 1); std::memcpy(buf + i * 8, &v, 8); }
    return;
  }
  if (width <= 32) { int32_t v = (int32_t)(i + 1); std::memcpy(buf + i * 4, &v, 4); }
  else { int64_t v = i + 1; std::memcpy(buf + i * 8, &v, 8); }
}

// 按元素类别/位宽读取第 i 个元素并打印
static void printElem(const char *buf, int64_t i, char kind, unsigned width) {
  if (kind == 'f') {
    if (width == 32) { float v; std::memcpy(&v, buf + i * 4, 4); llvm::outs() << v; }
    else { double v; std::memcpy(&v, buf + i * 8, 8); llvm::outs() << v; }
    return;
  }
  if (width == 1) { int8_t v; std::memcpy(&v, buf + i, 1); llvm::outs() << (int)v; }
  else if (width <= 32) { int32_t v; std::memcpy(&v, buf + i * 4, 4); llvm::outs() << v; }
  else { int64_t v; std::memcpy(&v, buf + i * 8, 8); llvm::outs() << v; }
}
} // anonymous namespace
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Conversion/Passes.h"

// ============ GPU AOT：GPU IR → fatbin(cubin+PTX) → 可部署 ELF ============
// 本机不需要 NVIDIA GPU：mydriver 只负责编译产出；产物拷到带 GPU 与驱动的
// 机器上直接运行。GPU kernel 由 NVVM 序列化为 fatbin（同时含 cubin 与 PTX，
// 驱动可自动挑选），host 侧通过内置 mgpu* shim 调 CUDA driver API。
// 链接只加 -L<toolkit>/lib64/stubs -lcuda，禁止写 RUNPATH：目标机若装了
// toolkit，rpath 会优先命中 stubs 里的空实现，导致 cuModuleLoadData 无效。
namespace {

struct GpuAotOptions {
  std::string outputDir;
  std::string entryName;
  std::string arch = "sm_80";
  std::string features; // 空 = 按 arch 自动推导
  std::string toolkit;  // 空 = CUDA_HOME/CUDA_ROOT/CUDA_PATH 或 /usr/local/cuda
  int optLevel = 2;
};

// arch → 默认 PTX ISA 版本（取该 arch 支持、ptxas 可接受的版本）
static std::string defaultPtxFeatures(llvm::StringRef arch) {
  int n = 0;
  if (arch.consume_front("sm_"))
    n = atoi(arch.str().c_str());
  if (n >= 100) return "+ptx86";
  if (n >= 80) return "+ptx78";
  if (n >= 75) return "+ptx63";
  return "+ptx60";
}

// 工具解析：优先与 mydriver 同目录（构建树内，LLVM 版本一致），其次 PATH
static std::string resolveTool(llvm::StringRef name) {
  if (llvm::sys::path::is_absolute(name) && llvm::sys::fs::exists(name))
    return name.str();
  llvm::SmallString<256> exePath;
  if (!llvm::sys::fs::real_path("/proc/self/exe", exePath)) {
    llvm::sys::path::remove_filename(exePath);
    llvm::sys::path::append(exePath, llvm::sys::path::filename(name));
    if (llvm::sys::fs::exists(exePath))
      return exePath.str().str();
  }
  auto found = llvm::sys::findProgramByName(name);
  return found ? *found : std::string();
}

static bool runTool(llvm::StringRef tool, llvm::ArrayRef<std::string> args,
                    llvm::StringRef what) {
  llvm::SmallVector<llvm::StringRef, 16> argv;
  argv.push_back(tool);
  for (auto &a : args) argv.push_back(a);
  std::string errMsg;
  int code = llvm::sys::ExecuteAndWait(tool, argv, /*Env=*/std::nullopt,
                                       /*Redirects=*/{},
                                       /*SecondsToWait=*/0, /*MemoryLimit=*/0,
                                       &errMsg);
  if (code != 0) {
    llvm::errs() << "[gpu-aot] " << what << " failed (exit=" << code << ")"
                 << (errMsg.empty() ? "" : ": " + errMsg) << "\n";
    return false;
  }
  return true;
}

static std::string cScalarType(const ArgDesc &d) {
  if (d.kind == ArgDesc::Kind::F32) return "float";
  if (d.kind == ArgDesc::Kind::F64) return "double";
  unsigned w = d.intWidth;
  if (w <= 8) return "int8_t";
  if (w <= 16) return "int16_t";
  if (w <= 32) return "int32_t";
  return "int64_t";
}

static std::string cElemType(const ArgDesc &d) {
  if (d.elemKind == 'f') return d.elemWidth == 32 ? "float" : "double";
  if (d.elemWidth <= 8) return "int8_t";
  if (d.elemWidth <= 16) return "int16_t";
  if (d.elemWidth <= 32) return "int32_t";
  return "int64_t";
}

static std::string cFormatFor(char kind, unsigned width) {
  if (kind == 'f') return "%f";
  return width <= 32 ? "%d" : "%lld";
}

static std::string cPrintCastFor(char kind, unsigned width) {
  if (kind == 'f') return "(double)";
  return width <= 32 ? "(int)" : "(long long)";
}

// 生成宿主 main：按 memref descriptor ABI 展开参数，输入填 iota，打印结果首尾
static std::string generateGpuHostSource(const GpuAotOptions &opts,
                                         llvm::ArrayRef<ArgDesc> args,
                                         llvm::ArrayRef<ArgDesc> results) {
  std::string src;
  llvm::raw_string_ostream os(src);
  os << "// generated by mydriver --gpu-aot\n"
        "// memref ABI: (allocated, aligned, offset, sizes[rank]..., "
        "strides[rank]...)\n"
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n\n";

  std::set<size_t> ranks;
  for (auto &d : args)
    if (d.kind == ArgDesc::Kind::MemRef) ranks.insert(d.shape.size());
  for (auto &d : results)
    if (d.kind == ArgDesc::Kind::MemRef) ranks.insert(d.shape.size());
  for (size_t r : ranks) {
    os << "typedef struct { void *allocated; void *aligned; int64_t offset;";
    if (r > 0)
      os << " int64_t sizes[" << r << "]; int64_t strides[" << r << "];";
    os << " } MemRefDesc" << r << ";\n";
  }
  os << "\n";

  os << "extern ";
  if (results.empty())
    os << "void";
  else if (results[0].kind == ArgDesc::Kind::MemRef)
    os << "MemRefDesc" << results[0].shape.size();
  else
    os << cScalarType(results[0]);
  os << " " << opts.entryName << "(";
  bool firstParam = true;
  for (auto &d : args) {
    if (d.kind != ArgDesc::Kind::MemRef) {
      if (!firstParam) os << ", ";
      firstParam = false;
      os << cScalarType(d);
      continue;
    }
    for (size_t i = 0; i < 3 + 2 * d.shape.size(); ++i) {
      if (!firstParam) os << ", ";
      firstParam = false;
      os << (i < 2 ? "void *" : "int64_t");
    }
  }
  os << ");\n\nint main(void) {\n";

  for (size_t i = 0; i < args.size(); ++i) {
    const ArgDesc &d = args[i];
    if (d.kind != ArgDesc::Kind::MemRef) {
      os << "  " << cScalarType(d) << " a" << i << " = (" << cScalarType(d)
         << ")1;\n";
      continue;
    }
    int64_t n = 1;
    for (int64_t s : d.shape) n *= s;
    std::string elemTy = cElemType(d);
    os << "  int64_t n" << i << " = " << n << ";\n"
       << "  " << elemTy << " *a" << i << " = (" << elemTy
       << "*)malloc((size_t)n" << i << " * sizeof(" << elemTy << "));\n"
       << "  for (int64_t k = 0; k < n" << i << "; ++k) a" << i << "[k] = ("
       << elemTy << ")(k + 1);\n";

    size_t r = d.shape.size();
    os << "  int64_t sz" << i << "[" << (r ? r : 1) << "] = {";
    if (r == 0) os << "0";
    for (size_t k = 0; k < r; ++k) os << (k ? ", " : "") << d.shape[k];
    os << "};\n  int64_t st" << i << "[" << (r ? r : 1) << "] = {";
    if (r == 0) os << "0";
    for (size_t k = 0; k < r; ++k) {
      int64_t acc = 1;
      for (size_t t = k + 1; t < r; ++t) acc *= d.shape[t];
      os << (k ? ", " : "") << acc;
    }
    os << "};\n";
  }

  std::string callArgs;
  {
    llvm::raw_string_ostream ca(callArgs);
    bool firstArg = true;
    for (size_t i = 0; i < args.size(); ++i) {
      const ArgDesc &d = args[i];
      if (!firstArg) ca << ", ";
      firstArg = false;
      if (d.kind != ArgDesc::Kind::MemRef) {
        ca << "a" << i;
        continue;
      }
      ca << "(void*)a" << i << ", (void*)a" << i << ", (int64_t)0";
      for (size_t k = 0; k < d.shape.size(); ++k) ca << ", sz" << i << "[" << k << "]";
      for (size_t k = 0; k < d.shape.size(); ++k) ca << ", st" << i << "[" << k << "]";
    }
  }

  if (results.empty()) {
    os << "  " << opts.entryName << "(" << callArgs << ");\n";
  } else if (results[0].kind == ArgDesc::Kind::MemRef) {
    const ArgDesc &rd = results[0];
    int64_t n = 1;
    for (int64_t s : rd.shape) n *= s;
    std::string elemTy = cElemType(rd);
    std::string fmt = cFormatFor(rd.elemKind, rd.elemWidth);
    std::string cast = cPrintCastFor(rd.elemKind, rd.elemWidth);
    os << "  MemRefDesc" << rd.shape.size() << " res = " << opts.entryName
       << "(" << callArgs << ");\n";
    os << "  " << elemTy << " *rp = (" << elemTy << "*)res.aligned;\n";
    os << "  printf(\"[gpu-aot] " << opts.entryName << "[0] = " << fmt
       << "\\n\", " << cast << "rp[0]);\n";
    if (n > 1)
      os << "  printf(\"[gpu-aot] " << opts.entryName << "[" << (n - 1)
         << "] = " << fmt << "\\n\", " << cast << "rp[" << (n - 1) << "]);\n";
  } else {
    const ArgDesc &rd = results[0];
    char kind = (rd.kind == ArgDesc::Kind::F32 || rd.kind == ArgDesc::Kind::F64)
                    ? 'f' : 'i';
    unsigned w = rd.kind == ArgDesc::Kind::F32 ? 32
                 : rd.kind == ArgDesc::Kind::F64 ? 64
                                                 : rd.intWidth;
    os << "  " << cScalarType(rd) << " res = " << opts.entryName << "("
       << callArgs << ");\n";
    os << "  printf(\"[gpu-aot] " << opts.entryName << " = "
       << cFormatFor(kind, w) << "\\n\", " << cPrintCastFor(kind, w)
       << "res);\n";
  }
  os << "  return 0;\n}\n";
  return src;
}

// 最小 mgpu* 运行时：把 MLIR gpu-to-llvm 生成的 host 调用映射到 CUDA driver API
static const char *kMgpuShimSource = R"SHIM(
#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

CUmodule mgpuModuleLoad(void *data, size_t size) {
  CUmodule m = 0;
  cuModuleLoadData(&m, data);
  return m;
}
void mgpuModuleUnload(CUmodule m) { cuModuleUnload(m); }
CUfunction mgpuModuleGetFunction(CUmodule m, const char *name) {
  CUfunction f = 0;
  cuModuleGetFunction(&f, m, name);
  return f;
}
CUstream mgpuStreamCreate(void) {
  CUstream s = 0;
  cuStreamCreate(&s, CU_STREAM_NON_BLOCKING);
  return s;
}
void mgpuStreamDestroy(CUstream s) { cuStreamDestroy(s); }
void mgpuStreamSynchronize(CUstream s) { cuStreamSynchronize(s); }
void mgpuLaunchKernel(CUfunction f, intptr_t gx, intptr_t gy, intptr_t gz,
                      intptr_t bx, intptr_t by, intptr_t bz, unsigned smem,
                      CUstream s, void **params, void **extra,
                      size_t gridConst) {
  cuLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, s, params, extra);
}
)SHIM";

static llvm::Expected<std::string>
runGpuAot(mlir::ModuleOp module, const GpuAotOptions &opts,
          llvm::ArrayRef<ArgDesc> args, llvm::ArrayRef<ArgDesc> results) {
  using namespace llvm;
  for (auto &d : args)
    for (int64_t s : d.shape)
      if (s < 0)
        return createStringError(inconvertibleErrorCode(),
                                 "GPU AOT requires static shapes");
  for (auto &d : results)
    for (int64_t s : d.shape)
      if (s < 0)
        return createStringError(inconvertibleErrorCode(),
                                 "GPU AOT requires static shapes");

  if (sys::fs::create_directories(opts.outputDir))
    return createStringError(inconvertibleErrorCode(),
                             "mkdir " + opts.outputDir + " failed");

  std::string base = opts.outputDir + "/" + opts.entryName;
  std::string gpuMlirPath = base + ".gpu.mlir";
  std::string nvvmMlirPath = base + ".nvvm.mlir";
  std::string llPath = base + ".ll";
  std::string hostCPath = base + "_host.c";
  std::string shimPath = base + "_mgpu_shim.c";
  std::string shimObj = base + "_mgpu_shim.o";
  std::string exePath = base + "_gpu";

  // 1) 写 GPU 高层 IR（gpu.launch + func/memref/arith/scf）
  {
    std::error_code ec;
    raw_fd_ostream os(gpuMlirPath, ec, sys::fs::OF_None);
    if (ec)
      return createStringError(inconvertibleErrorCode(),
                               "open " + gpuMlirPath + ": " + ec.message());
    module.print(os);
    os << "\n";
  }

  // 2) 定位 CUDA toolkit（NVVM 通过 CUDA_HOME/CUDA_ROOT/CUDA_PATH 找 libdevice）
  std::string toolkit = opts.toolkit;
  if (toolkit.empty()) {
    for (const char *v : {"CUDA_HOME", "CUDA_ROOT", "CUDA_PATH"})
      if (const char *p = std::getenv(v)) { toolkit = p; break; }
  }
  if (toolkit.empty()) toolkit = "/usr/local/cuda";
  if (!sys::fs::exists(toolkit + "/include/cuda.h"))
    return createStringError(inconvertibleErrorCode(),
                             "CUDA toolkit not found at " + toolkit +
                                 " (use --gpu-toolkit)");
  setenv("CUDA_HOME", toolkit.c_str(), 1);
  setenv("CUDA_ROOT", toolkit.c_str(), 1);

  // 3) 构建树工具链（必须与 mydriver 同版本，LLVM 22 IR 语法）
  std::string mlirOpt = resolveTool("mlir-opt");
  std::string mlirTranslate = resolveTool("mlir-translate");
  std::string clangPath = resolveTool("clang");
  if (mlirOpt.empty() || mlirTranslate.empty())
    return createStringError(inconvertibleErrorCode(),
                             "mlir-opt/mlir-translate not found (expected "
                             "next to mydriver)");
  if (clangPath.empty())
    return createStringError(inconvertibleErrorCode(), "clang not found");

  std::string features =
      opts.features.empty() ? defaultPtxFeatures(opts.arch) : opts.features;
  std::string pipeline = "--gpu-lower-to-nvvm-pipeline=cubin-chip=" + opts.arch +
                         " cubin-features=" + features +
                         " cubin-format=fatbin opt-level=" +
                         std::to_string(opts.optLevel);
  llvm::errs() << "[gpu-aot] arch=" << opts.arch << " features=" << features
               << " format=fatbin(cubin+PTX)\n";

  if (!runTool(mlirOpt, {gpuMlirPath, pipeline, "-o", nvvmMlirPath},
               "gpu-lower-to-nvvm-pipeline"))
    return createStringError(inconvertibleErrorCode(),
                             "NVVM lowering failed");
  if (!runTool(mlirTranslate, {"--mlir-to-llvmir", nvvmMlirPath, "-o", llPath},
               "mlir-to-llvmir"))
    return createStringError(inconvertibleErrorCode(),
                             "mlir-translate failed");

  // 4) host main + mgpu shim
  {
    std::error_code ec;
    raw_fd_ostream os(hostCPath, ec, sys::fs::OF_None);
    if (ec)
      return createStringError(inconvertibleErrorCode(),
                               "open " + hostCPath + ": " + ec.message());
    os << generateGpuHostSource(opts, args, results);
  }
  {
    std::error_code ec;
    raw_fd_ostream os(shimPath, ec, sys::fs::OF_None);
    if (ec)
      return createStringError(inconvertibleErrorCode(),
                               "open " + shimPath + ": " + ec.message());
    os << kMgpuShimSource;
  }

  if (!runTool(clangPath,
               {"-c", "-O2", "-I" + toolkit + "/include", shimPath, "-o",
                shimObj},
               "compile mgpu shim"))
    return createStringError(inconvertibleErrorCode(),
                             "mgpu shim compile failed");

  // 只加 -L<stubs> -lcuda，不写 RUNPATH（避免目标机误加载空实现 stub）
  if (!runTool(clangPath,
               {"-O2", llPath, hostCPath, shimObj, "-o", exePath,
                "-L" + toolkit + "/lib64/stubs", "-lcuda"},
               "link GPU executable"))
    return createStringError(inconvertibleErrorCode(),
                             "GPU executable link failed");
  return exePath;
}

} // namespace

extern "C" void my_runtime_print_i32(int32_t val) {
	fprintf(stdout, "[my-runtime-print] value = %d\n", val);
}

// runtime helper
extern "C" void my_runtime_print_string(const char* s) {
  fprintf(stdout, "[my-runtime-print-str] %s\n", s);
  fflush(stdout);
}

static llvm::cl::opt<std::string> parseCInput(
    "parse-c",
    llvm::cl::desc("Parse simple C source file and generate MyDialect MLIR"),
    llvm::cl::value_desc("filename")
);

static llvm::cl::opt<std::string> parseMLIRInput(
    "parse-mlir",
    llvm::cl::desc("Load MyDialect MLIR file"),
    llvm::cl::value_desc("filename")
);

static llvm::cl::opt<bool> hwLowerMode(
    "hw-lower",
    llvm::cl::desc("Select hw_mul lowering path: true=llvm intrinsic, false=arith.muli"),
    llvm::cl::init(false));

// 并行化开关：无 iter_args 且无外部调用的 elementwise 循环提升为 scf.forall。
// 默认开启；`--parallelize=false` 可关闭以便 A/B 对比。
static llvm::cl::opt<bool> parallelizeMode(
    "parallelize",
    llvm::cl::desc("Promote legal elementwise scf.for loops to scf.forall"),
    llvm::cl::init(true));

// 指令调度开关：基本块级 list scheduling（FU 数量 + 延迟模型）
static llvm::cl::opt<bool> scheduleMode(
    "schedule",
    llvm::cl::desc("Enable block-level list scheduling before LLVM lowering"),
    llvm::cl::init(true));

// 异构部署开关：device placement（把 hw_mul 闭包包进 device_exec）
static llvm::cl::opt<bool> devicePlacementMode(
    "device-placement",
    llvm::cl::desc("Place hw_mul closure ops into mydialect.device_exec regions"),
    llvm::cl::init(true));

// 部署目标：cpu 内联 device_exec（JIT 可执行）；其它值保留 host/device 边界
static llvm::cl::opt<std::string> deviceTarget(
    "device-target",
    llvm::cl::desc("Deployment target: cpu inlines device_exec, "
                   "otherwise keep the host/device boundary"),
    llvm::cl::init("cpu"));

// layout 传播开关：把布局转换提升出 elementwise 算子并复合相邻转换
static llvm::cl::opt<bool> layoutPropagationMode(
    "layout-propagation",
    llvm::cl::desc("Propagate tensor layouts before operator fusion"),
    llvm::cl::init(true));

// 优化档位：统一控制可选优化 pass 的开启程度
//   0 = 仅做必要的方言降级（不做任何可选优化）
//   1 = 局部/图清理优化（MyOpt、常量折叠、CSE、SymbolDCE、layout 传播）
//   2 = 全量图优化（在 1 的基础上加算子融合、tiling、并行化、指令调度）—— 默认
// 显式给出的 granular 开关（--parallelize/--schedule/--layout-propagation/
// --device-placement）优先于档位；未显式给出时由档位决定。
static llvm::cl::opt<int> optLevel(
    "opt-level",
    llvm::cl::desc("Optimization level: 0=lowering only, 1=local opts, "
                   "2=full graph opts (default)"),
    llvm::cl::init(2));

// 解析单个可选 pass 是否启用：命令行显式给出的 granular 开关优先，
// 否则按 opt-level 是否达到 requiredLevel 判定
static bool resolveToggle(const llvm::cl::opt<bool> &flag, int level,
                          int requiredLevel) {
  if (flag.getNumOccurrences() > 0)
    return flag.getValue();
  return level >= requiredLevel;
}

// 位置参数：<mlir-file> [intval|funcname] [funcname]
// 必须注册位置选项：cl 在没有 positional 选项时遇到任何位置参数
// 都会报 "Too many positional arguments" 并退出，位置参数模式会失效
static llvm::cl::list<std::string> positionalArgs(
    llvm::cl::Positional,
    llvm::cl::desc("<mlir-file> [intval|funcname] [funcname]"),
    llvm::cl::ZeroOrMore
);

static llvm::cl::opt<bool> concurrentJit(
  "concurrent-jit",
  llvm::cl::desc("Enable multi-thread concurrent JIT, run multiple mlir modules"),
  llvm::cl::init(false));

static llvm::cl::opt<unsigned> jitThreadCount(
  "jit-threads",
  llvm::cl::desc("Max thread count for concurrent JIT"),
  llvm::cl::init(4));

static llvm::cl::opt<int> jitOptLevel(
  "jit-opt",
  llvm::cl::desc("JIT LLVM optimization level"),
  llvm::cl::init(2));

static llvm::cl::opt<bool> jitCache(
  "jit-cache",
  llvm::cl::desc("Enable JIT bitcode cache: HIT skips MLIR lowering"),
  llvm::cl::init(false));

static llvm::cl::opt<bool> jitVerbose(
  "jit-verbose",
  llvm::cl::desc("Print JIT cache HIT/MISS + time"),
  llvm::cl::init(false));

static llvm::cl::opt<std::string> jitCacheDir(
  "jit-cache-dir",
  llvm::cl::desc("Persist JIT bitcode cache to this dir (<dir>/<key>.bc), "
                 "cross-process reuse"),
  llvm::cl::init(""));

static llvm::cl::opt<std::string> aotCompile(
  "aot-compile",
  llvm::cl::desc("AOT compile to .so (output path); empty=JIT"),
  llvm::cl::init(""));

// 输出已降级到 LLVM dialect 的 .mlir（供外部 mlir-translate/llc 工具链使用）
static llvm::cl::opt<std::string> emitMlir(
    "emit-mlir",
    llvm::cl::desc("Write the lowered (LLVM dialect) .mlir to this path and exit"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init(""));

// GPU AOT：本机编译，产物拿到带 NVIDIA GPU 的机器上运行。
// 产出 <dir>/<entry>_gpu（ELF，fatbin 内同时含 cubin 与 PTX）。
static llvm::cl::opt<std::string> gpuAot(
    "gpu-aot",
    llvm::cl::desc("Compile GPU IR to a deployable ELF in this dir "
                   "(requires --device-target=gpu)"),
    llvm::cl::value_desc("dir"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string> gpuArch(
    "gpu-arch",
    llvm::cl::desc("GPU arch for --gpu-aot (sm_70/sm_80/sm_90/...)"),
    llvm::cl::init("sm_80"));

static llvm::cl::opt<std::string> gpuFeatures(
    "gpu-features",
    llvm::cl::desc("PTX features for --gpu-aot (empty = derive from --gpu-arch)"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string> gpuToolkit(
    "gpu-toolkit",
    llvm::cl::desc("CUDA toolkit path for --gpu-aot "
                   "(empty = CUDA_HOME/CUDA_ROOT/CUDA_PATH or /usr/local/cuda)"),
    llvm::cl::init(""));

static llvm::cl::list<std::string> jitMlirFiles(
  "jit-mlir-files",
  llvm::cl::desc("MLIR files to run concurrently via JIT"),
  llvm::cl::value_desc("xxx.mlir"),
  llvm::cl::CommaSeparated);


int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  if (!parseCInput.empty() && !parseMLIRInput.empty() && !jitMlirFiles.empty()) {
    llvm::errs() << "Error: cannot use --parse-c and --parse-mlir together\n";
    return 1;
  }

  if (!parseCInput.empty()) {
    if (!positionalArgs.empty()) {
      llvm::errs() << "Error: --parse-c does not take positional arguments\n";
      return 1;
    }
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufOrErr =
        llvm::MemoryBuffer::getFile(parseCInput);
    if (!bufOrErr) {
      llvm::errs() << "Cannot open C file: " << parseCInput << "\n";
      return 1;
    }
    std::string src = bufOrErr.get()->getBuffer().str();

    mlir::DialectRegistry registry;
    registry.insert<mydialect::MyDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::tensor::TensorDialect>();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mydialect::MyDialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();
    ctx.loadDialect<mlir::tensor::TensorDialect>();

    mlir::OpBuilder builder(&ctx);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    mydialect::SimpleCParser parser(builder, module);
    if (mlir::failed(parser.parse(src))) {
      llvm::errs() << "Simple C parse failed!\n";
      return 1;
    }

    // 横幅走 stderr，IR 走 stdout：`--parse-c x.c > x.mlir` 重定向得到
    // 的文件只含可回读的 MLIR 文本（dump() 固定打印到 stderr，不能用）
    llvm::errs() << "===== Generated MyDialect IR from C =====\n";
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }

  if (!parseMLIRInput.empty()) {
  // ========= MLIR 输入模式：--parse-mlir <file> 或 位置参数 <file> =========
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::DialectRegistry registry;
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  // tensor 方言构造时会级联加载 complex 方言（upstream 行为），
  // 而 complex 声明了 ConvertToLLVMPatternInterface 的 promise；
  // 不注册其实现扩展的话，convert-to-llvm pass 遍历已加载方言时
  // 会触发 "promised by dialect 'complex' but never implemented" 致命错误
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  // lower-affine pass（AffineToStandard）在依赖分析时会级联加载 vector 方言，
  // vector 同样声明了 ConvertToLLVMPatternInterface 的 promise，
  // 因此必须注册其实现扩展，否则 convert-to-llvm 会触发同类致命错误
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);

  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  // onnx 输入文件（如 loop_test.mlir）解析需要 onnx 方言；
  // onnx.LoopYield 转换会生成 tensor.extract，需要 tensor 方言
  registry.insert<mlir::onnx::OnnxDialect>();
  registry.insert<mlir::tensor::TensorDialect>();

  mlir::MLIRContext ctx(registry);
  ctx.loadDialect<mydialect::MyDialect>();
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<mlir::LLVM::LLVMDialect>();
  ctx.loadDialect<mlir::memref::MemRefDialect>();
  ctx.loadDialect<mlir::affine::AffineDialect>();

  // ========= 确定输入文件与可选附加参数（intval / funcname）=========
  std::string mlirFilePath;
  size_t extraIdx = 0; // positionalArgs 中第一个附加参数的下标
  if (!parseMLIRInput.empty()) {
    mlirFilePath = parseMLIRInput;
  } else if (!positionalArgs.empty()) {
    mlirFilePath = positionalArgs[0];
    extraIdx = 1;
  } else {
    llvm::errs() << "Usage:\n"
                 << argv[0] << " <mlir-file> [intval] [funcname]\n"
                 << argv[0] << " --parse-mlir <mlir-file> [intval|funcname] [funcname]\n"
                 << argv[0] << " --parse-c <c-file>\n";
    return 1;
  }

  std::optional<int64_t> cliArg;
  std::string entryName = "main";
  bool entryNameExplicit = false;
  for (size_t i = extraIdx; i < positionalArgs.size(); ++i) {
    llvm::StringRef a(positionalArgs[i]);
    bool negative = a.starts_with("-");
    bool isNum = !a.empty() &&
                 llvm::all_of(a.drop_front(negative ? 1 : 0),
                              [](char c) { return c >= '0' && c <= '9'; }) &&
                 (negative ? a.size() > 1 : true);
    if (isNum && !cliArg.has_value()) {
      cliArg = atoll(positionalArgs[i].c_str());
    } else if (!isNum && !entryNameExplicit) {
      entryName = positionalArgs[i];
      entryNameExplicit = true;
    } else {
      llvm::errs() << "Error: too many or conflicting extra arguments\n";
      return 1;
    }
  }

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(mlirFilePath, &ctx);
  if (!module) {
    llvm::errs() << "parse mlir file failed\n";
    return 1;
  }

  llvm::errs() << "===== parsed IR dump =====\n";
  module->dump();

  // ========= 含 ONNX op 的输入先转 mydialect（无 onnx op 时为 no-op）=========
  // 必须在捕获入口签名之前完成：ONNX 转换会把入口签名的 builtin tensor
  // 改写成 MyTensorType，captureSignature 只识别 MyTensorType/memref/标量
  {
    mlir::PassManager onnxPM(&ctx);
    onnxPM.addPass(mlir::createConvertONNXToMyDialectPass());
    if (mlir::failed(onnxPM.run(module.get()))) {
      llvm::errs() << "ONNX to MyDialect conversion failed\n";
      return 1;
    }
  }

  // ========= 确定入口函数并在 lowering 前捕获原始签名 =========
  // 兼容 func.func（高层）和 llvm.func（已 lowering 到 LLVM dialect）两种入口
  bool entryIsLLVMFunc = false;
  mlir::func::FuncOp entryFunc;
  mlir::LLVM::LLVMFuncOp entryLLVMFunc;
  if (!entryName.empty()) {
    entryFunc = module->lookupSymbol<mlir::func::FuncOp>(entryName);
    if (!entryFunc) {
      entryLLVMFunc = module->lookupSymbol<mlir::LLVM::LLVMFuncOp>(entryName);
      if (entryLLVMFunc) entryIsLLVMFunc = true;
    }
  }
  if (!entryFunc && !entryLLVMFunc && !entryNameExplicit) {
    // 未显式指定入口：优先取第一个 func.func，否则取第一个 llvm.func
    module->walk([&](mlir::func::FuncOp f) {
      if (!entryFunc && !f.isDeclaration()) {
        entryFunc = f;
        entryName = f.getName().str();
      }
    });
    if (!entryFunc) {
      module->walk([&](mlir::LLVM::LLVMFuncOp f) {
        if (!entryLLVMFunc && !f.isDeclaration()) {
          entryLLVMFunc = f;
          entryIsLLVMFunc = true;
          entryName = f.getName().str();
        }
      });
    }
  }
  if (!entryFunc && !entryLLVMFunc) {
    llvm::errs() << "Error: cannot find entry func @" << entryName << "\n";
    return 1;
  }

  // ========= 检测 module 是否已 lowering 到 LLVM dialect =========
  // 如果是，跳过所有 mydialect/arith/scf → LLVM 的 lowering pass，
  // 直接进入 AOT 或 JIT 阶段（减少不必要的 pass 开销，也避免 mydialect pass
  // 对纯 LLVM dialect module 做出错误假设）。
  bool alreadyLLVM = true;
  module->walk([&](mlir::Operation *op) {
    if (!alreadyLLVM) return;
    if (auto *dialect = op->getDialect()) {
      mlir::StringRef ns = dialect->getNamespace();
      if (!ns.empty() && ns != "llvm" && ns != "builtin") {
        alreadyLLVM = false;
      }
    }
  });
  llvm::SmallVector<ArgDesc> argDescs, resultDescs;
  if (!alreadyLLVM) {
    if (!captureSignature(entryFunc, argDescs, resultDescs)) {
      llvm::errs() << "Error: unsupported argument/result type in entry func @"
                   << entryName << " (only int/index/f32/f64/static memref/tensor)\n";
      return 1;
    }
    if (resultDescs.size() > 1) {
      llvm::errs() << "Error: entry func with multiple results not supported\n";
      return 1;
    }
  }
  // 纯 LLVM dialect 时 argDescs/resultDescs 不填：compileModuleToSharedLib 不依赖签名，
  // runJitWithCache 也直接 invokePacked，暂不需要预先描述。
  // 后续如果要支持 JIT 路径的 llvm.func 参数/结果，再补 LLVM dialect 类型 → builtin 类型转换。

  // Pass流水线（仅在还未 lowering 到 LLVM dialect 时运行）
  // opt-level 档位派生：
  //   level>=1 开启局部/图清理优化；level>=2 开启全量图优化；
  //   granular 开关显式给出时优先。
  int level = optLevel.getValue();
  if (level < 0) level = 0;
  bool doLocalOpts = level >= 1;
  bool doGraphOpts = level >= 2;
  bool doLayout = resolveToggle(layoutPropagationMode, level, 1);
  bool doPlaceDevice = resolveToggle(devicePlacementMode, level, 1);
  bool doParallel = resolveToggle(parallelizeMode, level, 2);
  bool doSchedule = resolveToggle(scheduleMode, level, 2);
  // GPU 目标：kernel 生成必须在 LowerMyDialectPrint 之前（张量算子还在），
  // 且对外输出的是降级前的高层 IR（供 MLIR GPU 后端继续处理）
  bool gpuDeviceTarget = doPlaceDevice && deviceTarget.getValue() == "gpu";
  bool emitPreLLVMIR =
      gpuDeviceTarget && (!emitMlir.empty() || !gpuAot.empty());

  if (!alreadyLLVM) {
  mlir::PassManager optPM(&ctx);
  if (doLocalOpts) {
    optPM.addPass(mydialect::createMyOptPass()); // 高层优化Pass
    optPM.addPass(mydialect::createConstantFoldMyDialectPass());
    // 全局公共子表达式消除 + 死符号消除（折叠/优化后残留的重复计算与无用
    // memref.global/func 声明在此清理）
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createSymbolDCEPass());
    // layout 传播：把 transpose 类布局转换提升到 elementwise 消费端并复合，
    // 暴露可融合的 elementwise 链（必须在算子融合之前）
    if (doLayout)
      optPM.addPass(mydialect::createLayoutPropagationPass());
  }
  // 算子融合：add/mul 张量链 → fused_elementwise，须在 LowerMyDialectPrint
  // 之前（lowering 时融合体只生成一个循环 nest，中间结果不落地 memref）
  if (doGraphOpts)
    optPM.addPass(mydialect::createFusionElementwisePass());
  // 异构部署：hw_mul 仍存在时先做 device placement，把设备子图包进
  // mydialect.device_exec region（形成 host ↔ device 流水线边界）
  if (doPlaceDevice)
    optPM.addPass(mydialect::createPlaceDevicePass());
  // GPU 目标：在张量算子被 LowerMyDialectPrint 拆成 scf.for 之前，
  // 先把 device_exec region 内的算子替换成手写 gpu.launch kernel
  if (gpuDeviceTarget)
    optPM.addPass(mydialect::createLowerDevicePass("gpu"));
  // LowerHwMulPass 必须在 LowerMyDialectPrint 之前：hw_mul 先 lower 成
  // arith.muli / llvm.intrinsic，然后 LowerMyDialectPrint 处理剩下的
  // tensor_to_scalar / scalar_to_tensor（SimpleCParser 生成的桥接 op）
  optPM.addPass(mydialect::createLowerHwMulPass(hwLowerMode.getValue()));
  optPM.addPass(mydialect::createOpCountCycleEstimatePass());
  optPM.addPass(mydialect::createLowerMyDialectPrint());
  // Linalg 优化：LowerMyDialectPrint 把逐元素算子与 matmul 发射成
  // linalg.generic / linalg.matmul。先做 linalg 分块，再统一降回 scf.for，
  // 后续 schedule / SCF→CF / LLVM 链路保持原样。
  if (doGraphOpts)
    optPM.addPass(mydialect::createLinalgOptimizePass());
  optPM.addPass(mlir::createConvertLinalgToLoopsPass());
  // 边界落地：cpu 目标内联 device_exec（设备算子退回 host 执行）；
  // gpu 目标在 kernel 生成后同样内联，让 gpu.launch 直接位于 func 内，
  // 便于后续 --gpu-lower-to-nvvm-pipeline 之类的标准 GPU 后端处理
  if (doPlaceDevice)
    optPM.addPass(mydialect::createLowerDevicePass(
        gpuDeviceTarget ? "cpu" : deviceTarget.getValue()));
  // 分块：兜底处理其余 lowering 产生的 elementwise scf.for 做 strip-mining
  // （linalg 生成的分块循环 step≠1，会被该 pass 跳过）
  if (doGraphOpts)
    optPM.addPass(mydialect::createTilePass());
  // 并行化：tile 之后把最外层分块循环提升为 scf.forall
  // （含外部调用的循环如 print 保持串行；scf.forall 由 SCFToControlFlow 串行展开）
  if (doParallel)
    optPM.addPass(mydialect::createParallelPass());
  // affine → std：affine.min（分块边界）在 convert-to-llvm 前必须降级，
  // 否则会残留 index 类型的 unrealized_conversion_cast
  optPM.addPass(mlir::createLowerAffinePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  // 指令调度：在 LLVM lowering 前对基本块内无副作用算子做 list scheduling
  if (doSchedule)
    optPM.addPass(mydialect::createSchedulePass());
  // GPU 目标 + --emit-mlir：停在高层 IR（func/memref/arith/scf/gpu），
  // 交给 --gpu-lower-to-nvvm-pipeline 之类的标准 GPU 后端继续处理；
  // scf→cf / convert-to-llvm 会破坏该后端期望的输入形态
  if (!emitPreLLVMIR) {
    // scf -> cf（此版本 --convert-to-llvm 不直接降低 scf）
    optPM.addPass(mlir::createSCFToControlFlowPass());
    // cf -> llvm 等方言到 LLVM 的转换
    optPM.addPass(mlir::createConvertToLLVMPass());
  }

  if (mlir::failed(optPM.run(module.get()))) {
    llvm::errs() << "Lowering to LLVM dialect failed\n";
    return 1;
  }
  } // !alreadyLLVM

  llvm::errs() << "===== lowered IR dump =====\n";
  module->dump();

  // ===== GPU AOT（--gpu-aot <dir>）：本机编译，产物拿到 GPU 机器执行 =====
  if (!gpuAot.empty()) {
    if (!gpuDeviceTarget) {
      llvm::errs() << "Error: --gpu-aot requires --device-target=gpu\n";
      return 1;
    }
    if (alreadyLLVM) {
      llvm::errs() << "Error: --gpu-aot expects high-level GPU IR "
                      "(input is already LLVM dialect)\n";
      return 1;
    }
    GpuAotOptions gpuOpts;
    gpuOpts.outputDir = gpuAot;
    gpuOpts.entryName = entryName;
    gpuOpts.arch = gpuArch;
    gpuOpts.features = gpuFeatures;
    gpuOpts.toolkit = gpuToolkit;
    gpuOpts.optLevel = std::clamp(jitOptLevel.getValue(), 0, 3);
    auto exeOrErr = runGpuAot(module.get(), gpuOpts, argDescs, resultDescs);
    if (!exeOrErr) {
      llvm::handleAllErrors(exeOrErr.takeError(),
                            [](const llvm::ErrorInfoBase &e) {
        llvm::errs() << "[gpu-aot] failed: " << e.message() << "\n";
      });
      return 1;
    }
    llvm::outs() << "[gpu-aot] executable → " << *exeOrErr << "\n"
                 << "[gpu-aot] copy it to a machine with an NVIDIA GPU + "
                    "driver (libcuda.so.1) and run it directly\n";
    return 0;
  }

  // ===== 输出 .mlir =====
  // 普通目标：已降到 LLVM dialect（AOT 工具链：mlir-translate → llc）
  // GPU 目标：停在 gpu.launch 所在的高层 IR（交给 MLIR GPU 后端）
  if (!emitMlir.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream os(emitMlir, ec);
    if (ec) {
      llvm::errs() << "cannot open " << emitMlir << ": " << ec.message() << "\n";
      return 1;
    }
    module->print(os);
    os << "\n";
    llvm::errs() << "[emit-mlir] wrote " << emitMlir << "\n";
    return 0;
  }

  // GPU 目标产物含 gpu.launch，无法由进程内 JIT 直接执行
  if (gpuDeviceTarget) {
    llvm::errs() << "[device-target=gpu] generated gpu.launch kernels; GPU IR "
                    "cannot be JIT-executed on host, use --emit-mlir and feed "
                    "it to a GPU backend\n";
    return 0;
  }

  // ===== AOT 编译（--aot-compile <path>）=====
  if (!aotCompile.empty()) {
    mydialect::AotCompilerOptions aotOpts;
    aotOpts.outputDir = aotCompile;
    aotOpts.optLevel = jitOptLevel;
    auto soOrErr = mydialect::compileModuleToSharedLib(module.get(), aotOpts);
    if (!soOrErr) {
      llvm::handleAllErrors(soOrErr.takeError(), [](const llvm::ErrorInfoBase &e) {
        llvm::errs() << "AOT failed: " << e.message() << "\n";
      });
      return 1;
    }
    llvm::outs() << "[aot] compiled → " << *soOrErr << "\n";
    if (!jitMlirFiles.empty()) {
      auto runOrErr = mydialect::loadAndRunSharedLib(*soOrErr);
      if (runOrErr) {
        llvm::outs() << "[aot] main() → " << *runOrErr << "\n";
      } else {
        llvm::handleAllErrors(runOrErr.takeError(), [](const llvm::ErrorInfoBase &e) {
          llvm::errs() << "dlopen/run failed: " << e.message() << "\n";
        });
      }
    }
    return 0;
  }

  // ===== 带缓存的 JIT（--jit-cache）=====
  if (jitCache) {
    if (!jitCacheDir.empty())
      mydialect::setJitCachePersistDir(jitCacheDir);
    auto ownedModule = mlir::OwningOpRef<mlir::ModuleOp>(module.get().clone());
    auto result = mydialect::runJitWithCache(
        std::move(ownedModule), jitOptLevel, jitVerbose);
    if (!result) {
      llvm::handleAllErrors(result.takeError(), [](const llvm::ErrorInfoBase &e) {
        llvm::errs() << "runJitWithCache failed: " << e.message() << "\n";
      });
      return 1;
    }
    llvm::outs() << "[jit-cache] main() → " << *result << "\n";
    return 0;
  }

  // 创建ExecutionEngine（默认路径，无缓存）
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = mlir::makeOptimizingTransformer(0,0,nullptr);
  llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> maybeEngine =
      mlir::ExecutionEngine::create(module.get(), engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "create ExecutionEngine failed\n";
    llvm::handleAllErrors(maybeEngine.takeError(), [](const llvm::ErrorInfoBase &e){
      llvm::errs() << e.message() << "\n";
    });
    return 1;
  }
  auto engine = std::move(*maybeEngine);

  // 注册外部C函数 my_runtime_print_i32
  engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    auto symName = interner("my_runtime_print_i32");
    symbolMap[symName] = {
        llvm::orc::ExecutorAddr::fromPtr(my_runtime_print_i32),
        llvm::JITSymbolFlags::Exported
    };
    return symbolMap;
  });


  engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    auto symName = interner("my_runtime_print_string");
    symbolMap[symName] = {
        llvm::orc::ExecutorAddr::fromPtr(my_runtime_print_string),
        llvm::JITSymbolFlags::Exported
    };
    return symbolMap;
  });

  // ========= 读取入口函数（lowering 后为 llvm.func）=========
  auto llvmFuncOp = module->lookupSymbol<mlir::LLVM::LLVMFuncOp>(entryName);
  if (!llvmFuncOp) {
    llvm::errs() << "Error: cannot find llvm.func @" << entryName << "\n";
    return 1;
  }
  // 校验 lowering 后参数数与按原始签名展开的数量一致
  unsigned expectedArgs = 0;
  for (auto &d : argDescs)
    expectedArgs += (d.kind == ArgDesc::Kind::MemRef)
                        ? 3 + 2 * (unsigned)d.shape.size()
                        : 1;
  if (llvmFuncOp.getNumArguments() != expectedArgs) {
    llvm::errs() << "Error: lowered func arg count mismatch: expected "
                 << expectedArgs << ", got " << llvmFuncOp.getNumArguments()
                 << "\n";
    return 1;
  }

  // ========= 按签名构造 invokePacked 参数列表 =========
  // 布局：args[i] 指向第 i 个展开参数的值；最后一个元素指向返回值存储。
  // memref 参数展开为 (allocated, aligned, offset, sizes..., strides...)。
  llvm::SmallVector<void *> invokeArgs;
  std::vector<std::unique_ptr<ScalarSlot>> scalarSlots; // 保持地址稳定
  std::vector<std::unique_ptr<MemRefSlot>> memSlots;
  bool hasIntScalarArg = false;

  for (auto &d : argDescs) {
    if (d.kind == ArgDesc::Kind::Int) {
      auto s = std::make_unique<ScalarSlot>();
      hasIntScalarArg = true;
      int64_t v = cliArg.value_or(0);
      if (d.intWidth <= 32) { int32_t t = (int32_t)v; std::memcpy(s->data, &t, 4); }
      else std::memcpy(s->data, &v, 8);
      invokeArgs.push_back(s->data);
      scalarSlots.push_back(std::move(s));
    } else if (d.kind == ArgDesc::Kind::F32) {
      auto s = std::make_unique<ScalarSlot>();
      float v = (float)cliArg.value_or(0);
      std::memcpy(s->data, &v, 4);
      invokeArgs.push_back(s->data);
      scalarSlots.push_back(std::move(s));
    } else if (d.kind == ArgDesc::Kind::F64) {
      auto s = std::make_unique<ScalarSlot>();
      double v = (double)cliArg.value_or(0);
      std::memcpy(s->data, &v, 8);
      invokeArgs.push_back(s->data);
      scalarSlots.push_back(std::move(s));
    } else { // MemRef
      auto m = std::make_unique<MemRefSlot>();
      int64_t numElems = numElementsOf(d.shape);
      size_t elemBytes = d.elemWidth / 8;
      m->allocated = std::malloc((size_t)numElems * elemBytes);
      m->aligned = m->allocated;
      m->offset = 0;
      for (int64_t i = 0; i < numElems; ++i)
        fillIota((char *)m->allocated, i, d.elemKind, d.elemWidth);
      m->sizes.assign(d.shape.begin(), d.shape.end());
      // 行主序连续 strides
      int64_t acc = 1;
      m->strides.resize(d.shape.size());
      for (int i = (int)d.shape.size() - 1; i >= 0; --i) {
        m->strides[i] = acc;
        acc *= d.shape[i];
      }
      invokeArgs.push_back(&m->allocated);
      invokeArgs.push_back(&m->aligned);
      invokeArgs.push_back(&m->offset);
      for (auto &sz : m->sizes) invokeArgs.push_back(&sz);
      for (auto &st : m->strides) invokeArgs.push_back(&st);
      memSlots.push_back(std::move(m));
    }
  }
  if (cliArg.has_value() && !hasIntScalarArg && !argDescs.empty())
    llvm::errs() << "Warning: no integer scalar argument, ignoring intval\n";

  // 返回值存储（invokePacked 约定：最后一个 void* 指向结果存储）
  ScalarSlot retScalar;
  // memref 结果 descriptor 原始存储：(3 + 2*rank) 个 8 字节槽，
  // 布局 = allocated, aligned, offset, sizes..., strides...
  std::vector<int64_t> retMemBuf;
  if (!resultDescs.empty()) {
    auto &rd = resultDescs.front();
    if (rd.kind == ArgDesc::Kind::MemRef) {
      retMemBuf.resize(3 + 2 * rd.shape.size(), 0);
      invokeArgs.push_back(retMemBuf.data());
    } else {
      invokeArgs.push_back(retScalar.data);
    }
  }

  // ========= JIT invoke =========
  auto invokeRes = engine->invokePacked(entryName, invokeArgs);
  if (static_cast<bool>(invokeRes)) {
    llvm::errs() << "JIT invoke @" << entryName << " failed\n";
    return 1;
  }

  // ========= 打印结果 =========
  if (resultDescs.empty()) {
    llvm::outs() << entryName << "() returned void\n";
  } else {
    auto &rd = resultDescs.front();
    if (rd.kind == ArgDesc::Kind::Int) {
      int64_t v;
      if (rd.intWidth <= 32) { int32_t t; std::memcpy(&t, retScalar.data, 4); v = t; }
      else std::memcpy(&v, retScalar.data, 8);
      llvm::outs() << entryName << "() return: " << v << "\n";
    } else if (rd.kind == ArgDesc::Kind::F32) {
      float v; std::memcpy(&v, retScalar.data, 4);
      llvm::outs() << entryName << "() return: " << v << "\n";
    } else if (rd.kind == ArgDesc::Kind::F64) {
      double v; std::memcpy(&v, retScalar.data, 8);
      llvm::outs() << entryName << "() return: " << v << "\n";
    } else { // MemRef 结果：解析 descriptor 并按 strides 打印元素
      void *allocated = (void *)(intptr_t)retMemBuf[0];
      char *aligned = (char *)(intptr_t)retMemBuf[1];
      int64_t offset = retMemBuf[2];
      int rank = (int)rd.shape.size();
      llvm::ArrayRef<int64_t> sizes(retMemBuf.data() + 3, rank);
      llvm::ArrayRef<int64_t> strides(retMemBuf.data() + 3 + rank, rank);

      llvm::outs() << entryName << "() returned memref<";
      for (int i = 0; i < rank; ++i) llvm::outs() << rd.shape[i] << "x";
      llvm::outs() << (rd.elemKind == 'f' ? "f" : "i") << rd.elemWidth << "> = [";

      int64_t total = numElementsOf(rd.shape);
      llvm::SmallVector<int64_t> idx(rank, 0);
      for (int64_t lin = 0; lin < total; ++lin) {
        int64_t off = offset;
        for (int dd = 0; dd < rank; ++dd) off += idx[dd] * strides[dd];
        if (lin) llvm::outs() << ", ";
        printElem(aligned, off, rd.elemKind, rd.elemWidth);
        // 进位式递增多维下标
        for (int dd = rank - 1; dd >= 0; --dd) {
          if (++idx[dd] < sizes[dd]) break;
          idx[dd] = 0;
        }
      }
      llvm::outs() << "]\n";
      std::free(allocated);
    }
  }

  // 释放输入 memref 数据
  for (auto &m : memSlots)
    std::free(m->allocated);
  }

  if (concurrentJit) {
  if (jitMlirFiles.empty()) {
    llvm::errs() << "Error: --concurrent-jit requires --jit-mlir-files\n";
    return 1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // 多线程 JIT 模式：输入应为已 lowering 到 LLVM 方言（或可直接翻译）的模块。
  // 必须注册 LLVM/builtin 方言翻译接口，否则 ExecutionEngine::create 内的
  // translateModuleToLLVMIR 会因缺 interface 返回 null 报 "could not convert
  // to LLVM IR"
  mlir::DialectRegistry registry;
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();

  // 开启MLIRContext线程安全（只读IR安全）
  mlir::MLIRContext ctx(registry);
  ctx.enableMultithreading();
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> moduleList;

  for (auto &filePath : jitMlirFiles) {
    auto modOrErr = mlir::parseSourceFile<mlir::ModuleOp>(filePath, &ctx);
    if (!modOrErr) {
      llvm::errs() << "Parse mlir file failed: " << filePath << "\n";
      return 1;
    }
    // 注意：parseSourceFile 返回的 OwningOpRef 才持有所有权。
    // push_back(std::move(*modOrErr)) 只是拷贝 ModuleOp 句柄（operator*
    // 按值返回），modOrErr 在迭代末尾析构时会 erase 掉 op，导致
    // moduleList 存悬空指针，后续 clone 复用已释放内存触发断言/段错误。
    // 必须移动整个 OwningOpRef 完成所有权转移。
    moduleList.push_back(std::move(modOrErr));
  }

  // 【重要】如果你需要先做 lowering（hw_mul -> arith），在这里串行跑pass
  // PassManager pm(&ctx);
  // pm.addPass(mydialect::createLowerHwMulToArithPass());
  // pm.addPass(mydialect::createOpCountCycleEstimatePass());
  // for (auto &m : moduleList) if (failed(pm.run(m.get()))) { return 1; }

  if (mlir::failed(mydialect::runConcurrentJit(moduleList, jitThreadCount, jitOptLevel))) {
    llvm::errs() << "Concurrent JIT execution failed!\n";
    return 1;
  }
  }
  return 0;
}

