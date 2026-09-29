#include "pass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

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

// ============ Lower Pass ============
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
    // --- 4. applyPartialConversion ---
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// 折叠 mydialect.muli (MuliOp)
struct FoldMuliPattern : public mlir::RewritePattern {
  FoldMuliPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(mydialect::MuliOp::getOperationName(), 1, ctx) {}

  llvm::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto mulOp = llvm::cast<mydialect::MuliOp>(op);
    auto lhsConst = mulOp.getLhs().getDefiningOp<mlir::arith::ConstantOp>();
    auto rhsConst = mulOp.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
    if (!lhsConst || !rhsConst)
      return mlir::failure();

    auto lhsAttr = llvm::dyn_cast<mlir::IntegerAttr>(lhsConst.getValue());
    auto rhsAttr = llvm::dyn_cast<mlir::IntegerAttr>(rhsConst.getValue());
    if (!lhsAttr || !rhsAttr)
      return mlir::failure();

    int64_t lhs = lhsAttr.getInt();
    int64_t rhs = rhsAttr.getInt();
    int64_t res = lhs * rhs;

    // ========== 修复这一行 ==========
    auto newConst = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI32IntegerAttr(res));
    rewriter.replaceOp(op, newConst);
    return mlir::success();
  }
};

// 折叠 mydialect.addi (AddI32Op)
struct FoldAddiPattern : public mlir::RewritePattern {
  FoldAddiPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(mydialect::AddI32Op::getOperationName(), 1, ctx) {}

  llvm::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto addOp = llvm::cast<mydialect::AddI32Op>(op);
    auto lhsConst = addOp.getLhs().getDefiningOp<mlir::arith::ConstantOp>();
    auto rhsConst = addOp.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
    if (!lhsConst || !rhsConst)
      return mlir::failure();

    auto lhsAttr = llvm::dyn_cast<mlir::IntegerAttr>(lhsConst.getValue());
    auto rhsAttr = llvm::dyn_cast<mlir::IntegerAttr>(rhsConst.getValue());
    if (!lhsAttr || !rhsAttr)
      return mlir::failure();

    int64_t lhs = lhsAttr.getInt();
    int64_t rhs = rhsAttr.getInt();
    int64_t res = lhs + rhs;

    // ========== 修复这一行 ==========
    auto newConst = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI32IntegerAttr(res));
    rewriter.replaceOp(op, newConst);
    return mlir::success();
  }
};

// Print算子常量折叠Pattern
struct ConstantFoldMyDialectPattern : public mlir::RewritePattern {
  ConstantFoldMyDialectPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(mydialect::PrintOp::getOperationName(), 1, ctx) {}
  llvm::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    llvm::errs() << "Hit mydialect print op: " << op->getName() << "\n";
    auto constOp = op->getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
    if (!constOp)
      return mlir::failure();
    mlir::Attribute attr = constOp.getValue();
    auto typedAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr);
    if (!typedAttr)
      return mlir::failure();
    //auto loc = op->getLoc();
    // ========== 修复这一行 ==========
    //auto cstOp = rewriter.create<mlir::arith::ConstantOp>(loc, typedAttr);
    //rewriter.replaceOp(op, cstOp);
    llvm::errs() << "[DEBUG] PrintOp sees constant value: " << typedAttr.getInt() << "\n";
    return mlir::failure(); //没有改动IR，返回success会触发无限迭代
  }
};

struct ConstantFoldMyDialectPass
    : public mlir::PassWrapper<ConstantFoldMyDialectPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldMyDialectPass)
  llvm::StringRef getArgument() const final {
    return "constant-fold-mydialect";
  }
  llvm::StringRef getDescription() const final {
    return "Constant fold mydialect ops";
  }
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FoldMuliPattern, FoldAddiPattern, ConstantFoldMyDialectPattern>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
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

void registerConstantFoldMyDialectPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ConstantFoldMyDialectPass>();
  });
}

std::unique_ptr<mlir::Pass> createConstantFoldMyDialectPass() {
  return std::make_unique<ConstantFoldMyDialectPass>();
}
} // namespace mydialect

static mlir::PassRegistration<LowerMyDialectPrintPass> reg;
static mlir::PassRegistration<ConstantFoldMyDialectPass> regFold;

#include "mlir/IR/DialectRegistry.h"
extern "C" void mlirRegisterAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mydialect::MyDialect>();
}

