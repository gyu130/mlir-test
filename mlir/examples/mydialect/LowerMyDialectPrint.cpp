#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "MyDialect.h"
#include "pass.h"
#include <memory>

namespace mydialect {

// 必须先定义 DECL 再定义 DEF，否则 LowerMyDialectPrintOptions 未声明
#define GEN_PASS_DECL_LOWERMYDIALECTPRINT
#define GEN_PASS_DEF_LOWERMYDIALECTPRINT
#include "MyDialectPasses.h.inc"

struct LowerMyDialectPrintPass
    : public impl::LowerMyDialectPrintBase<LowerMyDialectPrintPass> {
  using Base::Base;
  using Base::disableRuntimeDecl;
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mydialect::MyDialect,
                    mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect,
                    mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect,
                    mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    // --- 1. 插入外部 runtime 函数声明（可通过 --disable-runtime-decl 跳过）---
    if (!disableRuntimeDecl) {
      mlir::OpBuilder moduleBuilder(module.getBody(),
                                    module.getBody()->begin());
      mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
      mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
      auto llvmFuncTy =
          mlir::LLVM::LLVMFunctionType::get(voidTy, {i32Ty}, false);
      mlir::LLVM::LLVMFuncOp::create(moduleBuilder,
                                     moduleBuilder.getUnknownLoc(),
                                     "my_runtime_print_i32",
                                     llvmFuncTy,
                                     mlir::LLVM::Linkage::External);
    }

    // --- 2. ConversionTarget：标记 mydialect op 为非法 ---
    mlir::ConversionTarget target(*ctx);
    target.addIllegalOp<mydialect::PrintOp>();
    target.addIllegalOp<mydialect::AddI32Op>();
    target.addIllegalOp<mydialect::MuliOp>();
    target.addIllegalOp<mydialect::SubI32Op>();
    target.addIllegalOp<mydialect::DivSIOp>();
    target.addIllegalOp<mydialect::CmpiOp>();
    target.addIllegalOp<mydialect::YieldOp>();
    target.addIllegalOp<mydialect::ForOp>();
    target.addIllegalOp<mydialect::AllocOp>();
    target.addIllegalOp<mydialect::LoadOp>();
    target.addIllegalOp<mydialect::StoreOp>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();

    // --- 3. 注册 patterns ---
    mlir::RewritePatternSet patterns(ctx);
    mydialect::populateMyDialectToLowerPatterns(patterns);

    // --- 4. applyPartialConversion ---
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mydialect

namespace mydialect {
// 引入注册函数声明（inline，可多 TU 安全）
#define GEN_PASS_REGISTRATION_LOWERMYDIALECTPRINT
#include "MyDialectPasses.h.inc"
} // namespace mydialect

// 静态注册：mlir-opt 启动时自动注册 --lower-mydialect-print pass
static bool _registered = []() {
  mydialect::registerLowerMyDialectPrint();
  return true;
}();
