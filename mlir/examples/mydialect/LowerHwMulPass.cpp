#include "LowerHwMulPass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mydialect {

// ============ 标量算子族 SW 路径：hw_xxx → 对应 arith 二元 op ============
// hw_mul→muli, hw_add→addi, hw_div→divsi, hw_and→andi, hw_or→ori, hw_xor→xori
template <typename OpTy, typename ArithOpTy>
struct HwScalarToArith : public mlir::OpRewritePattern<OpTy> {
  // 模板基类是依赖类型，不能用 using 继承构造器，显式转发
  HwScalarToArith(mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<OpTy>(ctx) {}

  mlir::LogicalResult matchAndRewrite(OpTy op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto res = ArithOpTy::create(rewriter, op.getLoc(), op.getLhs(),
                                 op.getRhs());
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ 标量算子族 HW 路径：hw_xxx → llvm.call_intrinsic ============
// 内在函数名由构造时传入，如 "llvm.mydialect.hw.mul.i32"，
// 对应 RISC-V 自定义指令 / ISS 仿真。
template <typename OpTy>
struct HwScalarToLLVMIntrinsic : public mlir::OpRewritePattern<OpTy> {
  HwScalarToLLVMIntrinsic(mlir::MLIRContext *ctx, llvm::StringRef name)
      : mlir::OpRewritePattern<OpTy>(ctx),
        intrinsicName(mlir::StringAttr::get(ctx, name)) {}

  mlir::LogicalResult matchAndRewrite(OpTy op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto callIntrinsic = mlir::LLVM::CallIntrinsicOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), intrinsicName,
        mlir::ValueRange{op.getLhs(), op.getRhs()});
    rewriter.replaceOp(op, callIntrinsic.getResults());
    return mlir::success();
  }

  mlir::StringAttr intrinsicName;
};

// ============ HW 路径公共工具：查找或创建模块级 private 声明 ============
// matmul/DMA 的硬件原语带 memref 操作数，不适合 llvm.call_intrinsic
// （LLVM intrinsic 无法承载 memref），统一降级为 func.call 到外部符号，
// 由设备后端/JIT 运行时提供实现（等价于设备 kernel 启动边界）。
// 返回 nullptr 表示已有同名声明但签名不匹配（调用方 notifyMatchFailure）。
static mlir::func::FuncOp
getOrCreateDecl(mlir::PatternRewriter &rewriter, mlir::Location loc,
                mlir::ModuleOp module, llvm::StringRef sym,
                mlir::FunctionType fnTy) {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(sym)) {
    if (existing.getFunctionType() != fnTy)
      return nullptr;
    return existing;
  }
  // 用 rewriter.create 插入模块开头（静态 FuncOp::create 不会插入符号表）
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto fn = rewriter.create<mlir::func::FuncOp>(loc, sym, fnTy);
  fn.setPrivate();
  return fn;
}

// ============ hw_matmul SW 路径：scf.for 三层循环软件模拟 ============
// 先把 C 清零，再 C[i,j] += A[i,k] * B[k,j]（K 循环在最内层）。
// 循环上界用 memref.dim 取值：静态维度会被 canonicalize 折叠成常量，
// 动态维度则运行时取值，天然支持两种 shape。
struct HwMatmulToScfLoops : public mlir::OpRewritePattern<HwMatmulOp> {
  using OpRewritePattern<HwMatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      HwMatmulOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value A = op.getA(), B = op.getB(), C = op.getC();

    auto c0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto c1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    mlir::Value M = mlir::memref::DimOp::create(rewriter, loc, A, c0);
    mlir::Value K = mlir::memref::DimOp::create(rewriter, loc, A, c1);
    mlir::Value N = mlir::memref::DimOp::create(rewriter, loc, B, c1);

    // 1) C[i,j] = 0
    auto zeroI = mlir::scf::ForOp::create(rewriter, loc, c0, M, c1);
    rewriter.setInsertionPointToStart(zeroI.getBody());
    auto zeroJ = mlir::scf::ForOp::create(rewriter, loc, c0, N, c1);
    rewriter.setInsertionPointToStart(zeroJ.getBody());
    mlir::Value zero = mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    mlir::memref::StoreOp::create(
        rewriter, loc, zero, C,
        mlir::ValueRange{zeroI.getInductionVar(), zeroJ.getInductionVar()});

    // 2) C[i,j] += A[i,k] * B[k,j]
    rewriter.setInsertionPointAfter(zeroI);
    auto accI = mlir::scf::ForOp::create(rewriter, loc, c0, M, c1);
    rewriter.setInsertionPointToStart(accI.getBody());
    auto accJ = mlir::scf::ForOp::create(rewriter, loc, c0, N, c1);
    rewriter.setInsertionPointToStart(accJ.getBody());
    auto accK = mlir::scf::ForOp::create(rewriter, loc, c0, K, c1);
    rewriter.setInsertionPointToStart(accK.getBody());
    mlir::Value i = accI.getInductionVar();
    mlir::Value j = accJ.getInductionVar();
    mlir::Value k = accK.getInductionVar();
    mlir::Value a = mlir::memref::LoadOp::create(rewriter, loc, A,
                                                 mlir::ValueRange{i, k});
    mlir::Value b = mlir::memref::LoadOp::create(rewriter, loc, B,
                                                 mlir::ValueRange{k, j});
    mlir::Value prod = mlir::arith::MulIOp::create(rewriter, loc, a, b);
    mlir::Value c = mlir::memref::LoadOp::create(rewriter, loc, C,
                                                 mlir::ValueRange{i, j});
    mlir::Value sum = mlir::arith::AddIOp::create(rewriter, loc, c, prod);
    mlir::memref::StoreOp::create(rewriter, loc, sum, C,
                                  mlir::ValueRange{i, j});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ============ hw_matmul HW 路径：func.call 外部符号 ============
struct HwMatmulToExternCall : public mlir::OpRewritePattern<HwMatmulOp> {
  using OpRewritePattern<HwMatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      HwMatmulOp op, mlir::PatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto fnTy = mlir::FunctionType::get(
        op.getContext(),
        {op.getA().getType(), op.getB().getType(), op.getC().getType()}, {});
    auto fn = getOrCreateDecl(rewriter, op.getLoc(), module,
                              "mydialect.hw.matmul", fnTy);
    if (!fn)
      return op.emitError()
             << "existing declaration of mydialect.hw.matmul has "
                "mismatched signature";
    mlir::func::CallOp::create(rewriter, op.getLoc(), fn,
                               mlir::ValueRange{op.getA(), op.getB(), op.getC()});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ============ hw_load/hw_store SW 路径：memref.load/store ============
// 索引逐个经 arith.index_cast i32→index，与 mydialect.load/store 降级一致。
struct HwLoadToMemRef : public mlir::OpRewritePattern<HwLoadOp> {
  using OpRewritePattern<HwLoadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      HwLoadOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::Value, 4> indexVars;
    for (auto idx : op.getIndices())
      indexVars.push_back(mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), idx));
    auto load = mlir::memref::LoadOp::create(
        rewriter, loc, op.getMemref(), mlir::ValueRange{indexVars});
    rewriter.replaceOp(op, load.getResult());
    return mlir::success();
  }
};

struct HwStoreToMemRef : public mlir::OpRewritePattern<HwStoreOp> {
  using OpRewritePattern<HwStoreOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      HwStoreOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::Value, 4> indexVars;
    for (auto idx : op.getIndices())
      indexVars.push_back(mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), idx));
    mlir::memref::StoreOp::create(rewriter, loc, op.getValue(),
                                  op.getMemref(),
                                  mlir::ValueRange{indexVars});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ============ hw_load/hw_store HW 路径：func.call DMA 外部符号 ============
struct HwLoadToExternCall : public mlir::OpRewritePattern<HwLoadOp> {
  using OpRewritePattern<HwLoadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      HwLoadOp op, mlir::PatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    llvm::SmallVector<mlir::Type> argTys{op.getMemref().getType()};
    for (auto idx : op.getIndices())
      argTys.push_back(idx.getType());
    auto fnTy = mlir::FunctionType::get(op.getContext(), argTys,
                                        {rewriter.getI32Type()});
    auto fn = getOrCreateDecl(rewriter, op.getLoc(), module,
                              "mydialect.hw.dma.load", fnTy);
    if (!fn)
      return op.emitError()
             << "existing declaration of mydialect.hw.dma.load has "
                "mismatched signature";
    auto call = mlir::func::CallOp::create(rewriter, op.getLoc(), fn,
                                           op.getOperands());
    rewriter.replaceOp(op, call->getResult(0));
    return mlir::success();
  }
};

struct HwStoreToExternCall : public mlir::OpRewritePattern<HwStoreOp> {
  using OpRewritePattern<HwStoreOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      HwStoreOp op, mlir::PatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto fnTy = mlir::FunctionType::get(op.getContext(), op.getOperandTypes(),
                                        {});
    auto fn = getOrCreateDecl(rewriter, op.getLoc(), module,
                              "mydialect.hw.dma.store", fnTy);
    if (!fn)
      return op.emitError()
             << "existing declaration of mydialect.hw.dma.store has "
                "mismatched signature";
    mlir::func::CallOp::create(rewriter, op.getLoc(), fn, op.getOperands());
    rewriter.eraseOp(op);
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
                    mlir::LLVM::LLVMDialect,
                    mlir::scf::SCFDialect,
                    mlir::memref::MemRefDialect,
                    mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    if (useHardwarePath) {
      // 标量族 → llvm.call_intrinsic（硬件自定义指令 / ISS 仿真）
      patterns.add<HwScalarToLLVMIntrinsic<HwMulOp>>(&getContext(),
          "llvm.mydialect.hw.mul.i32");
      patterns.add<HwScalarToLLVMIntrinsic<HwAddOp>>(&getContext(),
          "llvm.mydialect.hw.add.i32");
      patterns.add<HwScalarToLLVMIntrinsic<HwDivOp>>(&getContext(),
          "llvm.mydialect.hw.div.i32");
      patterns.add<HwScalarToLLVMIntrinsic<HwAndOp>>(&getContext(),
          "llvm.mydialect.hw.and.i32");
      patterns.add<HwScalarToLLVMIntrinsic<HwOrOp>>(&getContext(),
          "llvm.mydialect.hw.or.i32");
      patterns.add<HwScalarToLLVMIntrinsic<HwXorOp>>(&getContext(),
          "llvm.mydialect.hw.xor.i32");
      // 矩阵单元 / DMA → func.call 外部符号（设备 kernel 启动边界）
      patterns.add<HwMatmulToExternCall>(&getContext());
      patterns.add<HwLoadToExternCall>(&getContext());
      patterns.add<HwStoreToExternCall>(&getContext());
    } else {
      // 标量族 → arith（CPU JIT 可直接执行）
      patterns.add<HwScalarToArith<HwMulOp, mlir::arith::MulIOp>>(&getContext());
      patterns.add<HwScalarToArith<HwAddOp, mlir::arith::AddIOp>>(&getContext());
      patterns.add<HwScalarToArith<HwDivOp, mlir::arith::DivSIOp>>(&getContext());
      patterns.add<HwScalarToArith<HwAndOp, mlir::arith::AndIOp>>(&getContext());
      patterns.add<HwScalarToArith<HwOrOp, mlir::arith::OrIOp>>(&getContext());
      patterns.add<HwScalarToArith<HwXorOp, mlir::arith::XOrIOp>>(&getContext());
      // 矩阵单元 → scf 循环软件模拟；DMA → memref.load/store
      patterns.add<HwMatmulToScfLoops>(&getContext());
      patterns.add<HwLoadToMemRef>(&getContext());
      patterns.add<HwStoreToMemRef>(&getContext());
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
