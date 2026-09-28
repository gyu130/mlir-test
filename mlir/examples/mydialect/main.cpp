#include "MyDialect.h"
#include "pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"

extern "C" void my_runtime_print_i32(int32_t val) {
	fprintf(stdout, "[my-runtime-print] value = %d\n", val);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs() << "usage: ./mydriver input.mlir\n";
    return 1;
  }
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::DialectRegistry registry;
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);

  mlir::MLIRContext ctx(registry);
  ctx.loadDialect<mydialect::MyDialect>();
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<mlir::LLVM::LLVMDialect>();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &ctx);
  if (!module) {
    llvm::errs() << "parse mlir file failed\n";
    return 1;
  }

  llvm::outs() << "===== parsed IR dump =====\n";
  module->dump();

  // Pass流水线
  mlir::PassManager optPM(&ctx);
  optPM.addPass(mydialect::createLowerMyDialectPrintPass());
  optPM.addPass(mlir::createCanonicalizerPass());
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

  // ========== 核心修改：使用 lookup() 查找原生main ==========
  auto mainSymOrErr = engine->lookup("main");
  if (!mainSymOrErr) {
    llvm::errs() << "ERROR: cannot find symbol main\n";
    llvm::handleAllErrors(mainSymOrErr.takeError(), [](const llvm::ErrorInfoBase &e){
      llvm::errs() << e.message() << "\n";
    });
    return 1;
  }
  void* rawPtr = *mainSymOrErr;
  using MainFn = int (*)();
  MainFn mainFunc = reinterpret_cast<MainFn>(rawPtr);
  int ret = mainFunc();
  llvm::outs() << "main() return: " << ret << "\n";

  return 0;
}

