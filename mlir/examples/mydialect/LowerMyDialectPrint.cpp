#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "MyDialect.h"
#include "pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

namespace mydialect {

// 消除「可通过 TypeConverter 折叠」的 unrealized_conversion_cast：
//   source=memref, target=MyTensorType → convert(source)=memref, convert(target)=memref → identity，消掉
// 对 LowerDevicePass GPU 分支插的反向 cast（memref→MyTensorType）特别有用
struct CastEliminationPattern
    : public mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rw) const override {
    if (op.getInputs().size() != 1 || op.getResults().size() != 1)
      return mlir::failure();
    mlir::Value src = op.getInputs()[0];
    mlir::Type srcTy = src.getType();
    mlir::Type dstTy = op.getResult(0).getType();
    mlir::Type srcConverted = getTypeConverter()->convertType(srcTy);
    mlir::Type dstConverted = getTypeConverter()->convertType(dstTy);
    if (!srcConverted || !dstConverted || srcConverted != dstConverted)
      return mlir::failure();
    rw.replaceOp(op, src);
    return mlir::success();
  }
};

// ========== 自定义 TypeConverter ==========
struct MyDialectTypeConverter : public mlir::TypeConverter {
  MyDialectTypeConverter() {
    // 兜底：其余类型（i32/index/memref/...）保持原样。
    // 裸 TypeConverter 没有任何规则时 convertType 对所有类型都失败、
    // isLegal 恒为 false，所以 identity 兜底必须存在。
    // 注意：addConversion 是 LIFO——后注册的先尝试，
    // 因此 identity 兜底必须先注册，具体规则后注册才生效。
    addConversion([](mlir::Type ty) { return ty; });

    // wrapped_i32 降级为普通 i32（之后 i32 走正常 arith/llvm 链路）
    addConversion([](mydialect::WrappedI32Type type) -> mlir::Type {
      return mlir::IntegerType::get(type.getContext(), 32);
    });

    // wrapped_i32 ↔ i32 materialization：用 unrealized_conversion_cast 占位
    addSourceMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1) return nullptr;
          return builder.create<mlir::UnrealizedConversionCastOp>(
              loc, resultType, inputs[0]).getResult(0);
        });
    addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1) return nullptr;
          return builder.create<mlir::UnrealizedConversionCastOp>(
              loc, resultType, inputs[0]).getResult(0);
        });

    // 在 TypeConverter 构造/setup 函数内添加：
    addConversion([](mydialect::MyTensorType tensorTy) -> mlir::Type {
      // 取出 attr 参数
      mlir::ArrayAttr shapeAttr = tensorTy.getShape();
      mlir::Type elemTy = tensorTy.getElementType();

      mlir::SmallVector<int64_t,4> dims;
      for(auto attr : shapeAttr) {
        auto intAttr = mlir::cast<mlir::IntegerAttr>(attr);
        dims.push_back(intAttr.getInt());
      }
      // 降级目标：memref<dims, elemTy>
      return mlir::MemRefType::get(dims, elemTy);
   });


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
                    mlir::memref::MemRefDialect,
                    mlir::math::MathDialect,
                    mlir::linalg::LinalgDialect>();
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
    target.addIllegalOp<mydialect::WrapOp>();
    target.addIllegalOp<mydialect::UnwrapOp>();
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
    target.addIllegalOp<mydialect::LoopOp>();
    target.addIllegalOp<mydialect::LoopYieldOp>();
    target.addIllegalOp<mydialect::ConstantTensorOp>();
    target.addIllegalOp<mydialect::AddTensorOp>();
    target.addIllegalOp<mydialect::MulTensorOp>();
    target.addIllegalOp<mydialect::SubTensorOp>();
    target.addIllegalOp<mydialect::DivTensorOp>();
    target.addIllegalOp<mydialect::FusedElementwiseOp>();
    target.addIllegalOp<mydialect::CmpITensorOp>();
    target.addIllegalOp<mydialect::IdentityTensorOp>();
    target.addIllegalOp<mydialect::ReshapeTensorOp>();
    target.addIllegalOp<mydialect::TransposeTensorOp>();
    target.addIllegalOp<mydialect::ReduceSumTensorOp>();
    target.addIllegalOp<mydialect::ReluTensorOp>();
    target.addIllegalOp<mydialect::MatMulTensorOp>();
    target.addIllegalOp<mydialect::ReduceMaxTensorOp>();
    target.addIllegalOp<mydialect::ExpTensorOp>();
    target.addIllegalOp<mydialect::ConcatTensorOp>();
    target.addIllegalOp<mydialect::Conv2dTensorOp>();
    target.addIllegalOp<mydialect::SoftmaxTensorOp>();
    // SimpleCParser 桥接 op：由 pass.cpp 里的 TensorToScalarOpLowering /
    // ScalarToTensorOpLowering 转成 memref.load / memref.alloc+store
    target.addIllegalOp<mydialect::TensorToScalarOp>();
    target.addIllegalOp<mydialect::ScalarToTensorOp>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    // 张量算子的 lowering 现在发射 linalg.generic / linalg.matmul，
    // 由后续 --mydialect-linalg-opt / --convert-linalg-to-loops 处理
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    // UnrealizedConversionCastOp 标记为 illegal，让 CastEliminationPattern
    // 折叠 identity cast（memref↔MyTensorType 等经 converter 后同类型的 cast）
    target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

    // 自定义类型（wrapped_i32/mydialect.tensor）出现时 func.func 签名必须转换：
    // dialect 级 addLegalDialect 不感知操作数类型，需按签名动态判定合法性，
    // 配合 populateFunctionOpInterfaceTypeConversionPattern 完成重写
    // （MyTensorType→memref<...>、WrappedI32Type→i32 由 converter 完成）
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    // func.return 操作数类型须随签名一起转换（tensor→memref）
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return typeConverter.isLegal(op.getOperandTypes());
        });

    // --- 3. 注册 patterns（converter 在构造时绑定到每个 pattern）---
    mlir::RewritePatternSet patterns(ctx);
    mydialect::populateMyDialectToLowerPatterns(patterns, typeConverter);
    // identity cast 消除（memref↔MyTensorType 等经 converter 后同类型的 cast）
    patterns.add<mydialect::CastEliminationPattern>(typeConverter, ctx);
    // func.func 签名重写（承接上面的 addDynamicallyLegalOp）
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    // func.return 操作数类型重写
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // --- 4. applyPartialConversion ---
    // MLIR 22 不再接受 TypeConverter* 作第 4 个参数（改为 ConversionConfig），
    // converter 经 patterns 传入。
    if (mlir::failed(mlir::applyPartialConversion(
            module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

// ========= 类型转换说明 =========
// UnwrapOp/WrapOp 已在 TD 中定义，供后续扩展；
// 当前 test 用例中 print 直接接受 wrapped_i32，由 MyDialectTypeConverter 转 i32。

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
