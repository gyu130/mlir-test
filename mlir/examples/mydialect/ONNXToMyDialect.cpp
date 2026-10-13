//===-- ONNXToMyDialect.cpp - ONNX stub dialect + ONNX->MyDialect pass -----===//
//
// 本文件包含：
//   1. ONNX stub dialect 的实现（OnnxDialect::initialize()）
//   2. builtin tensor <-> mydialect::MyTensorType 的 TypeConverter
//   3. 8 个 ONNX -> mydialect 的 ConversionPattern
//   4. convert-onnx-to-mydialect pass 的实现与注册
//
// stub ONNX dialect 在 ONNXOps.td 中定义，使用 MLIR builtin 的 tensor 类型，
// 不依赖外部 onnx-mlir 项目。
//
//===----------------------------------------------------------------------===//

#include "ONNXToMyDialect.h"
#include "MyDialectOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//============================= ONNX stub dialect 实现 =======================
// onnx.cpp.inc 提供 OnnxDialect 的构造函数/析构函数/TypeID 定义
#include "onnx.cpp.inc"

// onnx-opdefs.cpp.inc 提供 ONNX 各 op 的方法定义
#define GET_OP_CLASSES
#include "onnx-opdefs.cpp.inc"

namespace mlir::onnx {

// OnnxDialect::initialize() 由 onnx.h.inc 声明，这里实现：
// 注册所有 stub ONNX ops
void OnnxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "onnx-opdefs.cpp.inc"
      >();
}

} // namespace mlir::onnx

//============================= TypeConverter ================================
// builtin::TensorType <-> mydialect::MyTensorType
//
// 转换方向说明：ONNX 输入用 builtin tensor，MyDialect 输出用 mydialect tensor。
// TypeConverter 同时支持两个方向的转换（LIFO 兜底后注册具体规则）。
struct OnnxToMyDialectTypeConverter : public mlir::TypeConverter {
  OnnxToMyDialectTypeConverter() {
    // 1. identity 兜底（先注册，后尝试）—— 必须在具体规则前注册
    addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
      return type;
    });

    // 2. builtin RankedTensorType -> mydialect::MyTensorType
    // 仅支持静态 shape（动态 shape 会在 pattern 内 notifyMatchFailure）
    // 注意：不添加反向 MyTensorType -> RankedTensorType 转换，否则
    // 转换后的 mydialect 张量类型会被 isLegal() 判定为非法（因再次转换回
    // builtin tensor 后与原类型不同），导致 FuncOp 签名转换死循环。
    addConversion([](mlir::RankedTensorType ty) -> std::optional<mlir::Type> {
      if (!ty.hasStaticShape())
        return std::nullopt;
      auto ctx = ty.getContext();
      SmallVector<mlir::Attribute> shapeAttrVec;
      for (int64_t dim : ty.getShape()) {
        shapeAttrVec.push_back(
            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), dim));
      }
      auto shapeAttr = mlir::ArrayAttr::get(ctx, shapeAttrVec);
      return mydialect::MyTensorType::get(ctx, shapeAttr, ty.getElementType());
    });
  }
};

//============================= 8 个 ConversionPattern =======================

// ONNX.Add -> mydialect.add_tensor
struct ConvertONNXAdd : public OpConversionPattern<mlir::onnx::ONNXAddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::onnx::ONNXAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return failure();
    auto newOp = rewriter.create<mydialect::AddTensorOp>(
        op.getLoc(), outTy, adaptor.getA(), adaptor.getB());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// ONNX.Mul -> mydialect.mul_tensor
struct ConvertONNXMul : public OpConversionPattern<mlir::onnx::ONNXMulOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::onnx::ONNXMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return failure();
    auto newOp = rewriter.create<mydialect::MulTensorOp>(
        op.getLoc(), outTy, adaptor.getA(), adaptor.getB());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// ONNX.Equal -> mydialect.cmpi_tensor（predicate = 4 表示 "eq"）
// mydialect::CmpITensorOp 的 predicate 取值范围：0~5（slt/sgt/sle/sge/eq/ne）
// 这里固定映射到 eq（4）
struct ConvertONNXEqual : public OpConversionPattern<mlir::onnx::ONNXEqualOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::onnx::ONNXEqualOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // 输入张量转换为 mydialect 张量类型
    auto inTy = getTypeConverter()->convertType(op.getA().getType());
    if (!inTy)
      return failure();
    // 结果类型：i1 张量（mydialect::CmpITensorOp 的 inferReturnTypes 会推导）
    // 但 applyPartialConversion 需要预先指定结果类型
    auto lhsTensorTy = mlir::cast<mydialect::MyTensorType>(inTy);
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto outTy = mydialect::MyTensorType::get(
        rewriter.getContext(), lhsTensorTy.getShape(), i1Ty);
    // predicate=4 对应 eq
    auto predAttr = rewriter.getI64IntegerAttr(4);
    auto newOp = rewriter.create<mydialect::CmpITensorOp>(
        op.getLoc(), outTy, adaptor.getA(), adaptor.getB(), predAttr);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// ONNX.Constant -> mydialect.constant_tensor
struct ConvertONNXConstant
    : public OpConversionPattern<mlir::onnx::ONNXConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::onnx::ONNXConstantOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto valAttr = op.getValueAttr();
    // 从 ElementsAttr 推导 mydialect 张量类型
    auto st = valAttr.getShapedType();
    if (!st.hasStaticShape())
      return failure();
    auto ctx = rewriter.getContext();
    SmallVector<mlir::Attribute> shapeAttrVec;
    for (int64_t d : st.getShape()) {
      shapeAttrVec.push_back(mlir::IntegerAttr::get(mlir::IndexType::get(ctx), d));
    }
    auto shapeAttr = mlir::ArrayAttr::get(ctx, shapeAttrVec);
    auto outTy =
        mydialect::MyTensorType::get(ctx, shapeAttr, st.getElementType());
    auto newOp = rewriter.create<mydialect::ConstantTensorOp>(
        op.getLoc(), outTy, valAttr);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// ONNX.Identity -> mydialect.identity_tensor
struct ConvertONNXIdentity
    : public OpConversionPattern<mlir::onnx::ONNXIdentityOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::onnx::ONNXIdentityOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto outTy = getTypeConverter()->convertType(op.getOutput().getType());
    if (!outTy)
      return failure();
    auto newOp = rewriter.create<mydialect::IdentityTensorOp>(
        op.getLoc(), outTy, adaptor.getInput());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// ONNX.Reshape -> mydialect.reshape_tensor
// 注意：mydialect::ReshapeTensorOp::inferReturnTypes 返回 failure（shape 是运行时张量）
// 因此 pattern 需要从 op 的结果类型显式推导输出类型
struct ConvertONNXReshape
    : public OpConversionPattern<mlir::onnx::ONNXReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::onnx::ONNXReshapeOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto outTy = getTypeConverter()->convertType(op.getOutput().getType());
    if (!outTy)
      return failure();
    auto newOp = rewriter.create<mydialect::ReshapeTensorOp>(
        op.getLoc(), outTy, adaptor.getData(), adaptor.getShape());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// ONNX.Transpose -> mydialect.transpose_tensor
struct ConvertONNXTranspose
    : public OpConversionPattern<mlir::onnx::ONNXTransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::onnx::ONNXTransposeOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto outTy = getTypeConverter()->convertType(op.getOutput().getType());
    if (!outTy)
      return failure();
    // ONNX Transpose 的 perm 是 OptionalAttr；若未指定则按默认逆序
    ArrayAttr permAttr = op.getPermAttr();
    if (!permAttr) {
      // 默认：对 N-D 输入按维度逆序构造 perm
      auto inTensorTy =
          mlir::dyn_cast<RankedTensorType>(op.getData().getType());
      if (!inTensorTy)
        return failure();
      int64_t rank = inTensorTy.getRank();
      SmallVector<mlir::Attribute> defaultPerm;
      for (int64_t i = rank - 1; i >= 0; --i) {
        defaultPerm.push_back(mlir::IntegerAttr::get(
            mlir::IndexType::get(rewriter.getContext()), i));
      }
      permAttr = mlir::ArrayAttr::get(rewriter.getContext(), defaultPerm);
    }
    auto newOp = rewriter.create<mydialect::TransposeTensorOp>(
        op.getLoc(), outTy, adaptor.getData(), permAttr);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// ONNX.ReduceSum -> mydialect.reduce_sum_tensor
struct ConvertONNXReduceSum
    : public OpConversionPattern<mlir::onnx::ONNXReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::onnx::ONNXReduceSumOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto outTy = getTypeConverter()->convertType(op.getOutput().getType());
    if (!outTy)
      return failure();
    // axes 缺省时默认空数组（不归约任何维度）
    ArrayAttr axesAttr = op.getAxesAttr();
    if (!axesAttr) {
      axesAttr = rewriter.getArrayAttr({});
    }
    BoolAttr keepDimsAttr = op.getKeepdimsAttr();
    if (!keepDimsAttr) {
      keepDimsAttr = rewriter.getBoolAttr(false);
    }
    auto newOp = rewriter.create<mydialect::ReduceSumTensorOp>(
        op.getLoc(), outTy, adaptor.getData(), axesAttr, keepDimsAttr);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

//============================= Pattern 集合注册 =============================
namespace {
void populateOnnxToMyDialectConversionPatterns(RewritePatternSet &patterns,
    OnnxToMyDialectTypeConverter &typeConverter) {
  patterns.add<
    ConvertONNXAdd,
    ConvertONNXMul,
    ConvertONNXEqual,
    ConvertONNXConstant,
    ConvertONNXIdentity,
    ConvertONNXReshape,
    ConvertONNXTranspose,
    ConvertONNXReduceSum
  >(typeConverter, patterns.getContext());
}
} // namespace

//============================= Pass 定义 ===================================
namespace {
struct ConvertOnnxToMyDialectPass
    : public PassWrapper<ConvertOnnxToMyDialectPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOnnxToMyDialectPass)

  llvm::StringRef getArgument() const final {
    return "convert-onnx-to-mydialect";
  }
  llvm::StringRef getDescription() const final {
    return "Convert ONNX stub dialect to MyDialect";
  }

  // 声明依赖的方言：pass 运行前 MLIR 会自动加载它们，
  // 这样 TypeConverter 内创建 mydialect::MyTensorType 才不会因 dialect 未加载而崩溃。
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mydialect::MyDialect>();
    registry.insert<mlir::onnx::OnnxDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OnnxToMyDialectTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    populateOnnxToMyDialectConversionPatterns(patterns, typeConverter);

    ConversionTarget target(getContext());
    // ONNX dialect 全部非法
    target.addIllegalDialect<mlir::onnx::OnnxDialect>();
    // 目标方言：mydialect / func / arith / builtin 合法
    // builtin 包含 ModuleOp，必须显式合法化，否则 applyFullConversion 会因
    // ModuleOp 未能合法化而失败
    target.addLegalDialect<mydialect::MyDialect,
                           func::FuncDialect,
                           arith::ArithDialect,
                           mlir::BuiltinDialect>();
    // UnrealizedConversionCastOp 在函数签名转换时会自动插入（用于桥接旧/新类型）
    // 必须显式合法化，否则整个 pattern 应用被回滚
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    // func.func 签名合法性：当函数参数/返回类型包含 onnx builtin tensor 时
    // 需要转换为 mydialect 张量类型
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });

    // 注册 func.func 签名转换模式（处理函数参数/返回类型转换）
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   typeConverter);
    // 注册 func.return 操作数类型转换模式（处理函数返回值类型转换）
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createConvertONNXToMyDialectPass() {
  return std::make_unique<ConvertOnnxToMyDialectPass>();
}

//============================= Pass 注册 ===================================
// PassRegistration 默认构造使用 std::make_unique<ConvertOnnxToMyDialectPass>()
// pass 名字通过 pass 类的 static StringRef getName() / getDescription() 提供。
static mlir::PassRegistration<ConvertOnnxToMyDialectPass> passReg;
