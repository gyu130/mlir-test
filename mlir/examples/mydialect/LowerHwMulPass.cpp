#include "LowerHwMulPass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mydialect {

// ============ Software lowering: hw_mul → arith.muli ============
struct HwMulToArithMuli : public mlir::OpRewritePattern<HwMulOp> {
  using OpRewritePattern<HwMulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(HwMulOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto muli = mlir::arith::MulIOp::create(
        rewriter, op.getLoc(), op.getLhs(), op.getRhs());
    rewriter.replaceOp(op, muli.getResult());
    return mlir::success();
  }
};

// ============ Hardware lowering: hw_mul → llvm.call_intrinsic ============
struct HwMulToLLVMIntrinsic : public mlir::OpRewritePattern<HwMulOp> {
  using OpRewritePattern<HwMulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(HwMulOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Type i32Ty = rewriter.getI32Type();
    auto intrinName = mlir::StringAttr::get(
        rewriter.getContext(), "llvm.mydialect.hw.mul.i32");

    auto callIntrinsic = mlir::LLVM::CallIntrinsicOp::create(
        rewriter, loc, i32Ty, intrinName,
        mlir::ValueRange{op.getLhs(), op.getRhs()});
    rewriter.replaceOp(op, callIntrinsic.getResults());
    return mlir::success();
  }
};

// ============ TableGen 生成的 Pass 基类声明/定义 ============
#define GEN_PASS_DECL_LOWERHWMULPASS
#define GEN_PASS_DEF_LOWERHWMULPASS
#include "MyDialectPasses.h.inc"

struct LowerHwMulPass : public impl::LowerHwMulPassBase<LowerHwMulPass> {
  using Base::Base;
  using Base::useHardwarePath;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mydialect::MyDialect,
                    mlir::arith::ArithDialect,
                    mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    if (useHardwarePath) {
      patterns.add<HwMulToLLVMIntrinsic>(&getContext());
    } else {
      patterns.add<HwMulToArithMuli>(&getContext());
    }
    (void)mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

// ============ Pass 注册由 MyDialect.cpp 的 GEN_PASS_REGISTRATION 统一处理 ============

// ============ 手写工厂函数（带 bool 参数，无默认值，不和 TableGen 冲突）========
std::unique_ptr<mlir::Pass> createLowerHwMulPass(bool useHardwarePath) {
  auto pass = std::make_unique<LowerHwMulPass>();
  pass->useHardwarePath = useHardwarePath;
  return pass;
}

// 静态注册：mlir-opt 启动时自动注册 --lower-hw-mul
#define GEN_PASS_REGISTRATION_LOWERHWMULPASS
#include "MyDialectPasses.h.inc"

static bool lowerHwMulPassRegistered = []() {
  registerLowerHwMulPass();
  return true;
}();

} // namespace mydialect
