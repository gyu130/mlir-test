#include "MyDialect.h"
#include "pass.h"
#include "MyOptPass.h"
#include "SimpleCParser.h"
#include "ONNXToMyDialect.h"
#include "LowerHwMulPass.h"
#include "OpCountCycleEstimate.h"
#include "ConcurrentJit.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
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
#include "mlir/Conversion/Passes.h"

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
  // LowerHwMulPass 必须在 LowerMyDialectPrint 之前：hw_mul 先 lower 成
  // arith.muli / llvm.intrinsic，然后 LowerMyDialectPrint 处理剩下的
  // tensor_to_scalar / scalar_to_tensor（SimpleCParser 生成的桥接 op）
  optPM.addPass(mydialect::createLowerHwMulPass(hwLowerMode.getValue()));
  optPM.addPass(mydialect::createOpCountCycleEstimatePass());
  optPM.addPass(mydialect::createLowerMyDialectPrint());
  // 边界落地：cpu 目标内联 device_exec（设备算子退回 host 执行）
  if (doPlaceDevice)
    optPM.addPass(
        mydialect::createLowerDevicePass(deviceTarget.getValue()));
  // 分块：对 lowering 生成的 elementwise scf.for 做 strip-mining
  // （必须在 LowerMyDialectPrint 之后，loop 才存在）
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
  // scf -> cf（此版本 --convert-to-llvm 不直接降低 scf）
  optPM.addPass(mlir::createSCFToControlFlowPass());
  // cf -> llvm 等方言到 LLVM 的转换
  optPM.addPass(mlir::createConvertToLLVMPass());

  if (mlir::failed(optPM.run(module.get()))) {
    llvm::errs() << "Lowering to LLVM dialect failed\n";
    return 1;
  }

  llvm::errs() << "===== lowered IR dump =====\n";
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

