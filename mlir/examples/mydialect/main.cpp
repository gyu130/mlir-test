#include "MyDialect.h"
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

  // 先声明外部 runtime 函数
 {
    mlir::OpBuilder moduleBuilder(module->getBody(), module->getBody()->begin());
    mlir::Type i32Ty = moduleBuilder.getI32Type();
    auto funcType = mlir::FunctionType::get(moduleBuilder.getContext(), {i32Ty}, {});
    auto funcOp = moduleBuilder.create<mlir::func::FuncOp>(
        moduleBuilder.getUnknownLoc(),
        "my_runtime_print_i32",
        funcType);
    funcOp.setPrivate();
  } 


  // 遍历IR，手动替换 mydialect.print → LLVM.call
  module->walk([&](mlir::Operation *op) {
    auto printOp = mlir::dyn_cast<mydialect::PrintOp>(op);
    if (!printOp) return;

    mlir::OpBuilder builder(printOp);
    mlir::Location loc = printOp.getLoc();
    auto symRef = mlir::SymbolRefAttr::get(builder.getContext(), "my_runtime_print_i32");
    builder.create<mlir::func::CallOp>(loc, mlir::TypeRange{}, symRef, printOp.getOperand());
    printOp->erase();
  });

  auto mainFunc = module->lookupSymbol<mlir::func::FuncOp>("main");
  if (mainFunc)
    mainFunc->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(&ctx));

    mlir::PassManager optPM(&ctx);
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createConvertToLLVMPass());

    if (mlir::failed(optPM.run(module.get()))) {
      llvm::errs() << "Lowering to LLVM dialect failed\n";
      return 1;
    }
  
  llvm::outs() << "===== lowered IR dump =====\n";
  module->dump();
  
  // ExecutionEngine
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

  // ========== 这里改成你示例的聚合初始化写法，不再写ExecutorSymbolDef ==========
  engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    auto symName = interner("my_runtime_print_i32");
    symbolMap[symName] = {
        llvm::orc::ExecutorAddr::fromPtr(my_runtime_print_i32),
        llvm::JITSymbolFlags::Exported
    };
    return symbolMap;
  });

  int32_t result = 0;
  auto err = engine->invoke("main", result);
  if (err) {
    llvm::errs() << "invoke main failed\n";
    llvm::handleAllErrors(std::move(err), [](const llvm::ErrorInfoBase &e){
      llvm::errs() << e.message() << "\n";
    });
    return 1;
  }


  return 0;
}

