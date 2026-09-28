#include "pass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"   // 新增
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace {

// ---------------- PrintOp Lowering Pattern（原有保留） ----------------
struct PrintOpLowering : public mlir::OpRewritePattern<mydialect::PrintOp> {
  using OpRewritePattern<mydialect::PrintOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mydialect::PrintOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sym = mlir::SymbolRefAttr::get(rewriter.getContext(), "my_runtime_print_i32");
    //rewriter.create<mlir::LLVM::LLVMCallOp>(loc, mlir::TypeRange{}, sym, op.getInput());
    mlir::LLVM::CallOp::create(rewriter, loc, mlir::TypeRange{}, sym, op.getValue());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ---------------- 新增：AddI32Op → arith.addi ----------------
struct AddI32OpLowering : public mlir::OpRewritePattern<mydialect::AddI32Op> {
  using OpRewritePattern<mydialect::AddI32Op>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mydialect::AddI32Op op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // mydialect.addi 直接替换为标准 arith.addi
    //auto newAdd = rewriter.create<mlir::arith::AddIOp>(loc, op.getLhs(), op.getRhs());
    auto newAdd = mlir::arith::AddIOp::create(rewriter, loc, op.getLhs(), op.getRhs());
    rewriter.replaceOp(op, newAdd.getResult());
    return mlir::success();
  }
};

struct LowerMyDialectPrintPass
    : public mlir::PassWrapper<LowerMyDialectPrintPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMyDialectPrintPass)

  llvm::StringRef getArgument() const final {
    return "lower-mydialect-print";
  }
  llvm::StringRef getDescription() const final {
    return "Lower mydialect.print and mydialect.addi op to LLVM/Arith dialect";
  }

  // 声明依赖方言，greedy模式最好加上
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<
        mydialect::MyDialect,
        mlir::LLVM::LLVMDialect,
        mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();
    ctx->loadDialect<mydialect::MyDialect>();
    ctx->loadDialect<mlir::LLVM::LLVMDialect>();
    ctx->loadDialect<mlir::arith::ArithDialect>(); // 新增加载arith

    mlir::OpBuilder moduleBuilder(module.getBody(), module.getBody()->begin());
    mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
    auto llvmFuncTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {i32Ty}, false);
    //moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
    mlir::LLVM::LLVMFuncOp::create(moduleBuilder,  
        moduleBuilder.getUnknownLoc(),
        "my_runtime_print_i32",
        llvmFuncTy,
        mlir::LLVM::Linkage::External);

    mlir::RewritePatternSet patterns(&getContext());
    // 使用新统一populate函数
    mydialect::populateMyDialectToLowerPatterns(patterns);

    mlir::GreedyRewriteConfig config;
    (void)applyPatternsGreedily(module, std::move(patterns), config);
  }
};
} // anonymous namespace

namespace mydialect {

// 统一注册全部pattern
void populateMyDialectToLowerPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<PrintOpLowering, AddI32OpLowering>(patterns.getContext());
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerMyDialectPrintPass() {
  return std::make_unique<LowerMyDialectPrintPass>();
}

} // namespace mydialect

static mlir::PassRegistration<LowerMyDialectPrintPass> reg;

#include "mlir/IR/DialectRegistry.h"
extern "C" void mlirRegisterAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mydialect::MyDialect>();
}

