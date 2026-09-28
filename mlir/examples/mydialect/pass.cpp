#include "pass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace {

// ============ PrintOp: mydialect.print → llvm.call ============
struct PrintOpLowering : public mlir::OpConversionPattern<mydialect::PrintOp> {
  using OpConversionPattern<mydialect::PrintOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mydialect::PrintOp op,
      typename mydialect::PrintOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sym = mlir::SymbolRefAttr::get(rewriter.getContext(), "my_runtime_print_i32");
    mlir::LLVM::CallOp::create(rewriter, loc, mlir::TypeRange{}, sym, adaptor.getValue());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ============ AddI32Op: mydialect.addi → arith.addi ============
struct AddI32OpLowering : public mlir::OpConversionPattern<mydialect::AddI32Op> {
  using OpConversionPattern<mydialect::AddI32Op>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mydialect::AddI32Op op,
      typename mydialect::AddI32Op::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newAdd = mlir::arith::AddIOp::create(
        rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newAdd.getResult());
    return mlir::success();
  }
};

// ============ MuliOp: mydialect.muli → arith.muli ============
struct MuliOpLowering : public mlir::OpConversionPattern<mydialect::MuliOp> {
  using OpConversionPattern<mydialect::MuliOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mydialect::MuliOp op,
      typename mydialect::MuliOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newMul = mlir::arith::MulIOp::create(
        rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newMul.getResult());
    return mlir::success();
  }
};

// ============ SubI32Op: mydialect.subi → arith.subi ============
struct SubI32OpLowering : public mlir::OpConversionPattern<mydialect::SubI32Op> {
  using OpConversionPattern<mydialect::SubI32Op>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mydialect::SubI32Op op,
      typename mydialect::SubI32Op::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newSub = mlir::arith::SubIOp::create(
        rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newSub.getResult());
    return mlir::success();
  }
};

// ============ DivSIOp: mydialect.divi_s → arith.divis ============
struct DivSIOpLowering : public mlir::OpConversionPattern<mydialect::DivSIOp> {
  using OpConversionPattern<mydialect::DivSIOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mydialect::DivSIOp op,
      typename mydialect::DivSIOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newDiv = mlir::arith::DivSIOp::create(
        rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newDiv.getResult());
    return mlir::success();
  }
};



// ============ Pass 定义 ============
struct LowerMyDialectPrintPass
    : public mlir::PassWrapper<LowerMyDialectPrintPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMyDialectPrintPass)

  llvm::StringRef getArgument() const final {
    return "lower-mydialect-print";
  }
  llvm::StringRef getDescription() const final {
    return "Lower mydialect.print / addi / muli to Arith/LLVM dialect";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<
        mydialect::MyDialect,
        mlir::func::FuncDialect,
        mlir::LLVM::LLVMDialect,
        mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    // --- 1. 插入外部 runtime 函数声明 ---
    mlir::OpBuilder moduleBuilder(module.getBody(), module.getBody()->begin());
    mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
    auto llvmFuncTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {i32Ty}, false);
    mlir::LLVM::LLVMFuncOp::create(moduleBuilder,
        moduleBuilder.getUnknownLoc(),
        "my_runtime_print_i32",
        llvmFuncTy,
        mlir::LLVM::Linkage::External);

    // --- 2. ConversionTarget：标记 mydialect op 为非法 ---
    mlir::ConversionTarget target(*ctx);
    target.addIllegalOp<mydialect::PrintOp>();
    target.addIllegalOp<mydialect::AddI32Op>();
    target.addIllegalOp<mydialect::MuliOp>();
    target.addIllegalOp<mydialect::SubI32Op>();
    target.addIllegalOp<mydialect::DivSIOp>();
    // 允许的目标方言
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();

    // --- 3. 注册 patterns ---
    mlir::RewritePatternSet patterns(ctx);
    mydialect::populateMyDialectToLowerPatterns(patterns);

    // --- 4. applyPartialConversion（替代 applyPatternsGreedily）---
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

namespace mydialect {

void populateMyDialectToLowerPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<PrintOpLowering, AddI32OpLowering, MuliOpLowering, SubI32OpLowering, DivSIOpLowering>(patterns.getContext());
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

