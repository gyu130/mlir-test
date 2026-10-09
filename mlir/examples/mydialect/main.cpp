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


extern "C" void my_runtime_print_i32(int32_t val) {
	fprintf(stdout, "[my-runtime-print] value = %d\n", val);
}

// runtime helper
extern "C" void my_runtime_print_string(const char* s) {
  fprintf(stdout, "[my-runtime-print-str] %s\n", s);
  fflush(stdout);
}


int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs() << "usage: ./mydriver input.mlir\n";
    return 1;
  }
  
  // 允许两种调用形式
  // ./mydriver file.mlir          --> main()
  // ./mydriver file.mlir  <int>   --> main(i32) -> i32
  if (argc < 2 || argc >3) {
    llvm::errs() << "Usage:\n";
    llvm::errs() << argv[0] << " <mlir-file>          # run main() no args\n";
    llvm::errs() << argv[0] << " <mlir-file> <intval>  # run main(i32) -> i32\n";
    return 1;
  }
  std::string mlirFilePath = argv[1];
  std::optional<int> cliArg;
  if(argc ==3) {
    cliArg = atoi(argv[2]);
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


  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &ctx);
  if (!module) {
    llvm::errs() << "parse mlir file failed\n";
    return 1;
  }

  llvm::outs() << "===== parsed IR dump =====\n";
  module->dump();

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

  // ========= 读取main函数签名（lowering后为 llvm.func）=========
  auto mainFuncOp = module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("main");
  if (!mainFuncOp) {
    llvm::errs() << "Error: cannot find func @main in module\n";
    return 1;
  }
  // 返回类型必须是 i32
  if (mainFuncOp.getNumResults() != 1 ||
      !llvm::isa<mlir::IntegerType>(mainFuncOp.getResultTypes()[0])) {
    llvm::errs() << "Error: func @main must return i32\n";
    return 1;
  }
  // 参数数量校验（只支持 main() 或 main(i32)）
  unsigned numInputs = mainFuncOp.getNumArguments();
  if (numInputs == 0 && cliArg.has_value()) {
    llvm::errs() << "Error: IR main() has NO argument, but you supplied command line argument "
                 << cliArg.value() << "\n";
    return 1;
  }
  if (numInputs == 1 && !cliArg.has_value()) {
    llvm::errs() << "Error: IR main(i32) expects one i32 argument, please provide integer in command line\n";
    return 1;
  }
  if (numInputs > 1) {
    llvm::errs() << "Error: only support main() or main(i32) -> i32, got "
                 << numInputs << " args\n";
    return 1;
  }

  // ========= 根据签名自动构造 invokePacked 参数列表 =========
  int retVal = 0;
  int input = cliArg.value_or(0);
  llvm::SmallVector<void*, 2> invokeArgs;
  if (numInputs == 1) {
    // main(i32) -> i32
    invokeArgs.push_back(&input);
  }
  // 无论有无参数，最后一个元素：返回值指针
  invokeArgs.push_back(&retVal);

  // ========= JIT invoke =========
  auto invokeRes = engine->invokePacked("main", invokeArgs);
  if (static_cast<bool>(invokeRes)) {
    llvm::errs() << "JIT invoke @main failed\n";
    return 1;
  }
  llvm::outs() << "main() return: " << retVal << "\n";
  return 0;
}

