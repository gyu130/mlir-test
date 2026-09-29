#include "pass.h"
#include "MyDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
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
        mlir::arith::ArithDialect,
        mlir::scf::SCFDialect>();
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
    target.addIllegalOp<mydialect::CmpiOp>();
    target.addIllegalOp<mydialect::YieldOp>();
    // 允许的目标方言
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
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
    mlir::ImplicitLocOpBuilder ib(op->getLoc(), rewriter);
    auto newConst = mlir::arith::ConstantOp::create(ib, rewriter.getI32IntegerAttr(res));
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
    mlir::ImplicitLocOpBuilder ib(op->getLoc(), rewriter);
    auto newConst = mlir::arith::ConstantOp::create(ib, rewriter.getI32IntegerAttr(res));
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
    // ========== 修复这一行 ==========
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

// ============ CmpiOp: mydialect.cmpi → arith.cmpi ============
struct CmpiOpLowering : public mlir::OpConversionPattern<mydialect::CmpiOp> {
  using OpConversionPattern<mydialect::CmpiOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mydialect::CmpiOp op,
      typename mydialect::CmpiOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto predStr = op.getPredicate();

    mlir::arith::CmpIPredicate pred;
    if (predStr == "slt") pred = mlir::arith::CmpIPredicate::slt;
    else if (predStr == "sgt") pred = mlir::arith::CmpIPredicate::sgt;
    else if (predStr == "sle") pred = mlir::arith::CmpIPredicate::sle;
    else if (predStr == "sge") pred = mlir::arith::CmpIPredicate::sge;
    else if (predStr == "eq")  pred = mlir::arith::CmpIPredicate::eq;
    else if (predStr == "ne")  pred = mlir::arith::CmpIPredicate::ne;
    else return rewriter.notifyMatchFailure(op, "unsupported predicate: " + predStr);

    auto newCmp = mlir::arith::CmpIOp::create(
        rewriter, loc, pred, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newCmp.getResult());
    return mlir::success();
  }
};

struct ForOpLowering : public mlir::OpConversionPattern<mydialect::ForOp> {
  using mlir::OpConversionPattern<mydialect::ForOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mydialect::ForOp op,
                                      mydialect::ForOp::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // scf.for 的边界/归纳变量为 index 类型，先把 i32 边界转成 index
    auto castToIndex = [&](mlir::Value v) {
      return mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), v);
    };
    mlir::Value lower = castToIndex(adaptor.getLower());
    mlir::Value upper = castToIndex(adaptor.getUpper());
    mlir::Value step  = castToIndex(adaptor.getStep());

    // 创建 scf.for（无迭代参数时 build 会自动生成空块和 scf.yield）
    auto scfFor = mlir::scf::ForOp::create(rewriter, loc, lower, upper, step);
    mlir::Block *dstBlock = scfFor.getBody();

    // 移除自动生成的 scf.yield，稍后用 mydialect.yield 转换后的代替
    rewriter.eraseOp(dstBlock->getTerminator());

    // 循环体入口：把 index 归纳变量转回 i32，供原 mydialect 循环体使用
    rewriter.setInsertionPointToStart(dstBlock);
    mlir::Value ivI32 = mlir::arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(), scfFor.getInductionVar());

    // 把原 body 块拼接到 scf.for body，i32 块参数 %i 替换为 ivI32
    mlir::Block *srcBlock = &op.getBody().front();
    rewriter.inlineBlockBefore(srcBlock, dstBlock, dstBlock->end(), {ivI32});

    // mydialect.yield -> scf.yield
    auto oldYield = llvm::cast<mydialect::YieldOp>(dstBlock->getTerminator());
    rewriter.setInsertionPoint(oldYield);
    mlir::scf::YieldOp::create(rewriter, loc);
    rewriter.eraseOp(oldYield);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};


} // anonymous namespace

namespace mydialect {
void populateMyDialectToLowerPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<PrintOpLowering, 
	AddI32OpLowering, 
	MuliOpLowering, 
	SubI32OpLowering, 
	DivSIOpLowering, 
	CmpiOpLowering,
	ForOpLowering>(patterns.getContext());
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

