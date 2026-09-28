#include "pass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace {
struct LowerMyDialectPrintPass
    : public mlir::PassWrapper<LowerMyDialectPrintPass, mlir::OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMyDialectPrintPass)

  // ========== 关键：给Pass提供命令行名字和描述 ==========
  llvm::StringRef getArgument() const final {
    return "lower-mydialect-print";
  }
  llvm::StringRef getDescription() const final {
    return "Lower mydialect.print op to LLVM dialect";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    ctx->loadDialect<mydialect::MyDialect>();
    ctx->loadDialect<mlir::LLVM::LLVMDialect>();

    mlir::OpBuilder moduleBuilder(module.getBody(), module.getBody()->begin());
    mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
    auto llvmFuncTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {i32Ty}, false);
    moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
        moduleBuilder.getUnknownLoc(),
        "my_runtime_print_i32",
        llvmFuncTy,
        mlir::LLVM::Linkage::External);

    mlir::RewritePatternSet patterns(&getContext());
    mydialect::populatePrintOpToLLVMPatterns(patterns);

    mlir::GreedyRewriteConfig config;
    (void)applyPatternsGreedily(module, std::move(patterns), config);
  }
};
} // anonymous namespace

namespace mydialect {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerMyDialectPrintPass() {
  return std::make_unique<LowerMyDialectPrintPass>();
}
} // namespace mydialect

static mlir::PassRegistration<LowerMyDialectPrintPass> reg;

#include "mlir/IR/DialectRegistry.h"
extern "C" void mlirRegisterAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mydialect::MyDialect>();
}

