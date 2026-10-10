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

// ========== 自定义 TypeConverter：mydialect 层的类型降级入口 ==========
// 注意：本 pass 的降级目标是 scf/memref/arith（print 直接到 llvm.call），
// 不是直接降到 LLVM，因此继承基类 TypeConverter 而不是 LLVMTypeConverter
// （后者会把 memref 转成 LLVM struct，与本 pass 产生的 memref op 冲突；
//  memref→LLVM 由后续 --convert-to-llvm pass 用它自己的 converter 完成）。
struct MyDialectTypeConverter : public mlir::TypeConverter {
  MyDialectTypeConverter() {
    // 自定义类型转换规则写在这里（addConversion 后注册的规则优先匹配）：
    //
    // 示例：将来给 mydialect 增加自定义类型 !mydialect.wrapped_i32 后，
    // addConversion([](mydialect::WrappedI32Type) -> mlir::Type {
    //   return mlir::IntegerType::get(..., 32);  // 降级成 i32
    // });

    // 兜底：其余类型（i32/index/memref/...）保持原样。
    // 裸 TypeConverter 没有任何规则时 convertType 对所有类型都失败、
    // isLegal 恒为 false，所以 identity 兜底必须存在。
    addConversion([](mlir::Type ty) { return ty; });
  }
};

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

    // ========== 1. 构造自定义 TypeConverter ==========
    MyDialectTypeConverter typeConverter;

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
      // void my_runtime_print_string(ptr)
      auto strPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
      auto strFuncTy =
          mlir::LLVM::LLVMFunctionType::get(voidTy, {strPtrTy}, false);
      mlir::LLVM::LLVMFuncOp::create(moduleBuilder,
                                     moduleBuilder.getUnknownLoc(),
                                     "my_runtime_print_string",
                                     strFuncTy,
                                     mlir::LLVM::Linkage::External);
    }

    // --- 2. ConversionTarget：标记 mydialect op 为非法 ---
    mlir::ConversionTarget target(*ctx);
    target.addIllegalOp<mydialect::PrintOp>();
    target.addIllegalOp<mydialect::PrintStrOp>();
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

    // 说明：MLIR 22 的 ConversionTarget 没有 addDynamicallyLegalType。
    // 类型合法性由绑定在 OpConversionPattern 上的 TypeConverter 驱动：
    // 转换驱动在应用 pattern 时取 pattern 的 converter，对操作数/结果
    // 自动做类型重映射，转换前后类型不一致处插入 materialization
    // （默认 unrealized_conversion_cast）。当前 converter 为 identity，
    // 所有现有类型合法；将来加入自定义类型时，除了 addConversion，
    // 还需对承载该类型的 op（如 func.func）addDynamicallyLegalOp，
    // 在回调里用 typeConverter.isLegal(...) / isSignatureLegal(...) 判定。

    // --- 3. 注册 patterns（converter 在构造时绑定到每个 pattern）---
    mlir::RewritePatternSet patterns(ctx);
    mydialect::populateMyDialectToLowerPatterns(patterns, typeConverter);

    // --- 4. applyPartialConversion ---
    // MLIR 22 不再接受 TypeConverter* 作第 4 个参数（改为 ConversionConfig），
    // converter 经 patterns 传入。
    if (mlir::failed(mlir::applyPartialConversion(
            module, target, std::move(patterns)))) {
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
