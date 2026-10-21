#include "MyDialect.h"
#include "pass.h"
#include "MyOptPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include <cstdlib>
#include <cstring>
#include <memory>
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

extern "C" void my_runtime_print_i32(int32_t val) {
	fprintf(stdout, "[my-runtime-print] value = %d\n", val);
}

// runtime helper
extern "C" void my_runtime_print_string(const char* s) {
  fprintf(stdout, "[my-runtime-print-str] %s\n", s);
  fflush(stdout);
}


int main(int argc, char** argv) {
  // 用法：
  //   ./mydriver file.mlir                    # 入口默认 main（缺省取第一个 func）
  //   ./mydriver file.mlir <intval>           # 整型标量参数取 intval
  //   ./mydriver file.mlir <funcname>         # 指定入口函数名
  //   ./mydriver file.mlir <intval> <funcname>
  if (argc < 2 || argc > 4) {
    llvm::errs() << "Usage:\n"
                 << argv[0] << " <mlir-file> [intval] [funcname]\n";
    return 1;
  }
  std::string mlirFilePath = argv[1];
  std::optional<int64_t> cliArg;
  std::string entryName = "main";
  bool entryNameExplicit = false;
  if (argc >= 3) {
    llvm::StringRef a2(argv[2]);
    bool negative = a2.starts_with("-");
    bool isNum = !a2.empty() &&
                 llvm::all_of(a2.drop_front(negative ? 1 : 0),
                              [](char c) { return c >= '0' && c <= '9'; }) &&
                 (negative ? a2.size() > 1 : true);
    if (isNum) cliArg = atoll(argv[2]);
    else { entryName = argv[2]; entryNameExplicit = true; }
  }
  if (argc >= 4) {
    entryName = argv[3];
    entryNameExplicit = true;
  }

  
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::DialectRegistry registry;
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);

  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::memref::MemRefDialect>();

  mlir::MLIRContext ctx(registry);
  ctx.loadDialect<mydialect::MyDialect>();
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<mlir::LLVM::LLVMDialect>();
  ctx.loadDialect<mlir::memref::MemRefDialect>();
  ctx.loadDialect<mlir::affine::AffineDialect>();


  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &ctx);
  if (!module) {
    llvm::errs() << "parse mlir file failed\n";
    return 1;
  }

  llvm::outs() << "===== parsed IR dump =====\n";
  module->dump();

  // ========= 确定入口函数并在 lowering 前捕获原始签名 =========
  auto entryFunc = module->lookupSymbol<mlir::func::FuncOp>(entryName);
  if (!entryFunc && !entryNameExplicit) {
    // 未显式指定且没有 @main：取第一个非声明 func.func
    module->walk([&](mlir::func::FuncOp f) {
      if (!entryFunc && !f.isDeclaration()) {
        entryFunc = f;
        entryName = f.getName().str();
      }
    });
  }
  if (!entryFunc) {
    llvm::errs() << "Error: cannot find entry func @" << entryName << "\n";
    return 1;
  }
  llvm::SmallVector<ArgDesc> argDescs, resultDescs;
  if (!captureSignature(entryFunc, argDescs, resultDescs)) {
    llvm::errs() << "Error: unsupported argument/result type in entry func @"
                 << entryName << " (only int/index/f32/f64/static memref/tensor)\n";
    return 1;
  }
  if (resultDescs.size() > 1) {
    llvm::errs() << "Error: entry func with multiple results not supported\n";
    return 1;
  }

  // Pass流水线
  mlir::PassManager optPM(&ctx);
  optPM.addPass(mydialect::createMyOptPass()); // 高层优化Pass
  optPM.addPass(mydialect::createConstantFoldMyDialectPass());
  optPM.addPass(mydialect::createLowerMyDialectPrint());
  optPM.addPass(mlir::createCanonicalizerPass());
  // scf -> cf（此版本 --convert-to-llvm 不直接降低 scf）
  optPM.addPass(mlir::createSCFToControlFlowPass());
  // cf -> llvm 等方言到 LLVM 的转换
  optPM.addPass(mlir::createConvertToLLVMPass());

  if (mlir::failed(optPM.run(module.get()))) {
    llvm::errs() << "Lowering to LLVM dialect failed\n";
    return 1;
  }

  llvm::outs() << "===== lowered IR dump =====\n";
  module->dump();

  // 创建ExecutionEngine
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
  return 0;
}

