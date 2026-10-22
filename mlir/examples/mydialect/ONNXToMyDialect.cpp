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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

//============================= ONNX stub dialect 实现 =======================
// onnx.cpp.inc 提供 OnnxDialect 的构造函数/析构函数/TypeID 定义
#include "onnx.cpp.inc"

// onnx-opdefs.cpp.inc 提供 ONNX 各 op 的方法定义
#define GET_OP_CLASSES
#include "onnx-opdefs.cpp.inc"

namespace mlir::onnx {

// ============================================================================
// ONNXConstantOp 自定义 assembler
// onnx-mlir 输出格式：%0 = onnx.Constant dense<1> : tensor<i64>
// （没有额外属性，没有 attr-dict）
// ODS assemblyFormat 强制 attr-dict 指令，但 op 没有额外 attribute 时
// 会强制期待 {} 或内联属性，导致 "expected ':'" 解析错误。
// 因此用 hasCustomAssemblyFormat=1 自己实现 parse/print。
// ============================================================================
mlir::ParseResult ONNXConstantOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  // onnx.Constant 的输入格式就是 ElementsAttr 的标准字符串：
  //   onnx.Constant dense<1> : tensor<i64>
  // 这里整个 `dense<1> : tensor<i64>` 是一个完整的 ElementsAttr，
  // parseAttribute 会一次性吃掉它（ElementsAttr 自带值和类型）。
  // 然后从 ElementsAttr 的 getType() 里提取 result type。
  mlir::Attribute valueAttrRaw;
  if (parser.parseAttribute(valueAttrRaw))
    return mlir::failure();
  auto valueAttr = mlir::dyn_cast_or_null<mlir::ElementsAttr>(valueAttrRaw);
  if (!valueAttr)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected ElementsAttr for onnx.Constant");
  result.getOrAddProperties<ONNXConstantOp::Properties>().value = valueAttr;
  result.addTypes(valueAttr.getType());
  return mlir::success();
}

void ONNXConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  // 直接打印完整 ElementsAttr（值 : 类型）—— 与 onnx-mlir 输出格式一致
  printer << getValueAttr();
}

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
    // 单规则：优先 RankedTensorType → MyTensorType，其余 identity
    addConversion([](mlir::Type ty) -> std::optional<mlir::Type> {
      if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(ty)) {
        if (!rtt.hasStaticShape()) return std::nullopt;
        auto ctx = ty.getContext();
        SmallVector<mlir::Attribute> shapeAttrVec;
        for (int64_t dim : rtt.getShape())
          shapeAttrVec.push_back(mlir::IntegerAttr::get(mlir::IndexType::get(ctx), dim));
        auto shapeAttr = mlir::ArrayAttr::get(ctx, shapeAttrVec);
        return mydialect::MyTensorType::get(ctx, shapeAttr, rtt.getElementType());
      }
      return ty;  // identity
    });
  }
};

//============================= 8 个 ConversionPattern =======================

// 通用 helper：把值通过 TypeConverter 转成目标类型，如果需要则 materialize cast
// 用于 ONNX op pattern 里手动 cast adaptor operands（避免反向 cast 套娃）。
static mlir::Value castIfNeeded(const mlir::TypeConverter &converter,
                                mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value v) {
  auto targetTy = converter.convertType(v.getType());
  if (!targetTy || targetTy == v.getType())
    return v;
  return builder.create<mlir::UnrealizedConversionCastOp>(loc, targetTy, v)
      .getResult(0);
}

// 通用 helper：把 RankedTensorType 转成 MyTensorType（如果是 tensor 类型）
static mlir::Type tensorTypeOrSelf(const mlir::TypeConverter &converter, mlir::Type t) {
  return converter.convertType(t);
}

// ONNX.Add -> mydialect.add_tensor（RewritePattern，不依赖 adaptor）
struct ConvertONNXAdd : public mlir::OpRewritePattern<mlir::onnx::ONNXAddOp> {
  ConvertONNXAdd(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXAddOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto outTy = tensorTypeOrSelf(converter, op.getResult().getType());
    if (!outTy) return failure();
    auto a = castIfNeeded(converter, rewriter, op.getLoc(), op.getA());
    auto b = castIfNeeded(converter, rewriter, op.getLoc(), op.getB());
    rewriter.replaceOpWithNewOp<mydialect::AddTensorOp>(op, outTy, a, b);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

// ONNX.Mul -> mydialect.mul_tensor
struct ConvertONNXMul : public mlir::OpRewritePattern<mlir::onnx::ONNXMulOp> {
  ConvertONNXMul(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXMulOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto outTy = tensorTypeOrSelf(converter, op.getResult().getType());
    if (!outTy) return failure();
    auto a = castIfNeeded(converter, rewriter, op.getLoc(), op.getA());
    auto b = castIfNeeded(converter, rewriter, op.getLoc(), op.getB());
    rewriter.replaceOpWithNewOp<mydialect::MulTensorOp>(op, outTy, a, b);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

// ONNX.Equal -> mydialect.cmpi_tensor（predicate = 4 表示 "eq"）
struct ConvertONNXEqual : public mlir::OpRewritePattern<mlir::onnx::ONNXEqualOp> {
  ConvertONNXEqual(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXEqualOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto inTy = tensorTypeOrSelf(converter, op.getA().getType());
    if (!inTy) return failure();
    auto lhsTensorTy = mlir::cast<mydialect::MyTensorType>(inTy);
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto outTy = mydialect::MyTensorType::get(
        rewriter.getContext(), lhsTensorTy.getShape(), i1Ty);
    auto predAttr = rewriter.getI64IntegerAttr(4);
    auto a = castIfNeeded(converter, rewriter, op.getLoc(), op.getA());
    auto b = castIfNeeded(converter, rewriter, op.getLoc(), op.getB());
    rewriter.replaceOpWithNewOp<mydialect::CmpITensorOp>(op, outTy, a, b, predAttr);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

// ONNX.Constant -> mydialect.constant_tensor
struct ConvertONNXConstant : public mlir::OpRewritePattern<mlir::onnx::ONNXConstantOp> {
  ConvertONNXConstant(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXConstantOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto valAttr = op.getValueAttr();
    auto st = valAttr.getShapedType();
    if (!st.hasStaticShape()) return failure();
    auto ctx = rewriter.getContext();
    SmallVector<mlir::Attribute> shapeAttrVec;
    for (int64_t d : st.getShape()) {
      shapeAttrVec.push_back(mlir::IntegerAttr::get(mlir::IndexType::get(ctx), d));
    }
    auto shapeAttr = mlir::ArrayAttr::get(ctx, shapeAttrVec);
    auto outTy = mydialect::MyTensorType::get(ctx, shapeAttr, st.getElementType());
    rewriter.replaceOpWithNewOp<mydialect::ConstantTensorOp>(op, outTy, valAttr);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

// ONNX.Identity -> mydialect.identity_tensor
struct ConvertONNXIdentity : public mlir::OpRewritePattern<mlir::onnx::ONNXIdentityOp> {
  ConvertONNXIdentity(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXIdentityOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto outTy = tensorTypeOrSelf(converter, op.getOutput().getType());
    if (!outTy) return failure();
    auto input = castIfNeeded(converter, rewriter, op.getLoc(), op.getInput());
    rewriter.replaceOpWithNewOp<mydialect::IdentityTensorOp>(op, outTy, input);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

// ONNX.Reshape -> mydialect.reshape_tensor
struct ConvertONNXReshape : public mlir::OpRewritePattern<mlir::onnx::ONNXReshapeOp> {
  ConvertONNXReshape(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXReshapeOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto outTy = tensorTypeOrSelf(converter, op.getOutput().getType());
    if (!outTy) return failure();
    auto data = castIfNeeded(converter, rewriter, op.getLoc(), op.getData());
    auto shape = castIfNeeded(converter, rewriter, op.getLoc(), op.getShape());
    rewriter.replaceOpWithNewOp<mydialect::ReshapeTensorOp>(op, outTy, data, shape);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

// ONNX.Transpose -> mydialect.transpose_tensor
struct ConvertONNXTranspose : public mlir::OpRewritePattern<mlir::onnx::ONNXTransposeOp> {
  ConvertONNXTranspose(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXTransposeOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto outTy = tensorTypeOrSelf(converter, op.getOutput().getType());
    if (!outTy) return failure();
    ArrayAttr permAttr = op.getPermAttr();
    if (!permAttr) {
      auto inTensorTy = mlir::dyn_cast<RankedTensorType>(op.getData().getType());
      if (!inTensorTy) return failure();
      int64_t rank = inTensorTy.getRank();
      SmallVector<mlir::Attribute> defaultPerm;
      for (int64_t i = rank - 1; i >= 0; --i) {
        defaultPerm.push_back(mlir::IntegerAttr::get(
            mlir::IndexType::get(rewriter.getContext()), i));
      }
      permAttr = mlir::ArrayAttr::get(rewriter.getContext(), defaultPerm);
    }
    auto data = castIfNeeded(converter, rewriter, op.getLoc(), op.getData());
    rewriter.replaceOpWithNewOp<mydialect::TransposeTensorOp>(op, outTy, data, permAttr);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

// ONNX.ReduceSum -> mydialect.reduce_sum_tensor
struct ConvertONNXReduceSum : public mlir::OpRewritePattern<mlir::onnx::ONNXReduceSumOp> {
  ConvertONNXReduceSum(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXReduceSumOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto outTy = tensorTypeOrSelf(converter, op.getOutput().getType());
    if (!outTy) return failure();
    ArrayAttr axesAttr = op.getAxesAttr();
    if (!axesAttr) axesAttr = rewriter.getArrayAttr({});
    BoolAttr keepDimsAttr = op.getKeepdimsAttr();
    if (!keepDimsAttr) keepDimsAttr = rewriter.getBoolAttr(false);
    auto data = castIfNeeded(converter, rewriter, op.getLoc(), op.getData());
    rewriter.replaceOpWithNewOp<mydialect::ReduceSumTensorOp>(
        op, outTy, data, axesAttr, keepDimsAttr);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

// arith.constant（结果类型为 ranked tensor）-> mydialect.constant_tensor
struct ConvertArithConstant : public mlir::OpRewritePattern<arith::ConstantOp> {
  ConvertArithConstant(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(arith::ConstantOp op,
      mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<RankedTensorType>(op.getType())) return failure();
    auto outTy = tensorTypeOrSelf(converter, op.getType());
    if (!outTy) return failure();
    auto valueAttr = llvm::cast<ElementsAttr>(op.getValue());
    rewriter.replaceOpWithNewOp<mydialect::ConstantTensorOp>(op, outTy, valueAttr);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

// ONNX.Loop -> mydialect.loop
// 用 RewritePattern 而不是 OpConversionPattern——后者需要 adaptor 把 operands 通过
// TypeConverter 转换，但 Loop 的 carry_inputs 可能是 func.func block arg（还没签名转换），
// 导致 adaptor 构造失败（<<UNKNOWN SSA VALUE>>）。RewritePattern 直接操作原始值。
// 简化版本：maxTripCount/cond 标量提取（从 ONNXConstant 直接取），
// body 内所有 tensor 类型保持原样（myLoop verify 已放宽接受 tensor）。
struct ConvertONNXLoop : public mlir::OpRewritePattern<mlir::onnx::ONNXLoopOp> {
public:
  using mlir::OpRewritePattern<mlir::onnx::ONNXLoopOp>::OpRewritePattern;
  ConvertONNXLoop(mlir::MLIRContext *ctx, mlir::TypeConverter &converter)
      : OpRewritePattern(ctx), converter(converter) {}

  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXLoopOp op,
      mlir::PatternRewriter &rewriter) const override {
    // 1. max_trip_count (tensor<i64>) → i64 标量
    mlir::Value maxTripTensor = op.getMaxTripCount();
    mlir::Value maxTripScalar;
    if (auto constOp = maxTripTensor.getDefiningOp<mlir::onnx::ONNXConstantOp>()) {
      auto attr = constOp.getValueAttr();
      if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr))
        maxTripScalar = rewriter.create<mlir::arith::ConstantOp>(
            op.getLoc(), rewriter.getI64IntegerAttr((*dense.begin()).getSExtValue()));
    }
    if (!maxTripScalar)
      return rewriter.notifyMatchFailure(op,
          "max_trip_count must be ONNXConstantOp with i64 value");

    // 2. cond (tensor<i1>) → i1 标量
    mlir::Value condTensor = op.getCond();
    mlir::Value condScalar;
    if (auto constOp = condTensor.getDefiningOp<mlir::onnx::ONNXConstantOp>()) {
      auto attr = constOp.getValueAttr();
      if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr))
        condScalar = rewriter.create<mlir::arith::ConstantOp>(
            op.getLoc(), rewriter.getBoolAttr((*dense.begin()).getBoolValue()));
    }
    if (!condScalar)
      return rewriter.notifyMatchFailure(op,
          "initCond must be ONNXConstantOp with i1 value");

    // 3. carry inputs: tensor → MyTensorType（立即 cast）
    SmallVector<mlir::Value> convertedCarries;
    mlir::OpBuilder builder(op);
    for (auto v : op.getCarriedInputs()) {
      auto targetTy = converter.convertType(v.getType());
      if (!targetTy) return failure();
      if (targetTy == v.getType()) {
        convertedCarries.push_back(v);
      } else {
        auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
            op.getLoc(), targetTy, v);
        convertedCarries.push_back(cast.getResult(0));
      }
    }
    // result types 也转
    SmallVector<mlir::Type> outTys;
    for (mlir::Type t : op.getResultTypes()) {
      auto converted = converter.convertType(t);
      if (!converted) return failure();
      outTys.push_back(converted);
    }

    // 4. 立即创建 mydialect::LoopOp（用 OpBuilder，不是 deferred 的 rewriter.create）
    auto newOp = builder.create<mydialect::LoopOp>(
        op.getLoc(), outTys, maxTripScalar, condScalar, convertedCarries);

    // 5. 把 ONNX body 移入新算子
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());

    // 6. body 里的 onnx.Yield → mydialect.loop_yield
    mlir::Block &bodyBlock = newOp.getBody().front();
    auto *oldYield = bodyBlock.getTerminator();
    if (auto onnxYield = mlir::dyn_cast<mlir::onnx::ONNXLoopYieldOp>(oldYield)) {
      SmallVector<mlir::Value> convertedCarryArgs;
      for (auto carry : onnxYield.getCarryArgs()) {
        auto targetTy = converter.convertType(carry.getType());
        if (!targetTy) return failure();
        if (targetTy == carry.getType()) {
          convertedCarryArgs.push_back(carry);
        } else {
          mlir::OpBuilder yieldBuilder(onnxYield);
          auto cast = yieldBuilder.create<mlir::UnrealizedConversionCastOp>(
              op.getLoc(), targetTy, carry);
          convertedCarryArgs.push_back(cast.getResult(0));
        }
      }
      mlir::OpBuilder yieldBuilder(onnxYield);
      yieldBuilder.create<mydialect::LoopYieldOp>(
          op.getLoc(), onnxYield.getCond(), convertedCarryArgs);
      onnxYield->erase();
    }

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }

private:
  mlir::TypeConverter &converter;
};

// ONNX.LoopYield -> mydialect.loop_yield（RewritePattern）
struct ConvertONNXLoopYield
    : public mlir::OpRewritePattern<mlir::onnx::ONNXLoopYieldOp> {
  ConvertONNXLoopYield(mlir::MLIRContext *ctx, mlir::TypeConverter &c)
      : OpRewritePattern(ctx), converter(c) {}
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXLoopYieldOp op,
      mlir::PatternRewriter &rewriter) const override {
    // cond 从 0维 tensor<i1> → tensor.extract 提取 i1 标量
    mlir::Value cond = op.getCond();
    auto *condDefOp = cond.getDefiningOp();
    if (auto mtt = converter.convertType(cond.getType())) {
      if (auto mttTy = mlir::dyn_cast<mydialect::MyTensorType>(mtt)) {
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
            op.getLoc(), mttTy, cond);
        cond = cast.getResult(0);
      }
    }
    // 如果 cond 还是 tensor 类型，做 tensor.extract 提 i1 标量
    if (auto st = mlir::dyn_cast<mlir::ShapedType>(cond.getType())) {
      if (st.getRank() == 0) {
        cond = rewriter.create<mlir::tensor::ExtractOp>(op.getLoc(), cond);
      }
    }
    // carry args 转换
    SmallVector<mlir::Value> convertedCarries;
    for (auto v : op.getCarryArgs()) {
      convertedCarries.push_back(castIfNeeded(converter, rewriter, op.getLoc(), v));
    }
    rewriter.replaceOpWithNewOp<mydialect::LoopYieldOp>(op, cond, convertedCarries);
    return success();
  }
private:
  mlir::TypeConverter &converter;
};

struct ConvertONNXEntryPoint : public mlir::OpRewritePattern<mlir::onnx::ONNXEntryPointOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::onnx::ONNXEntryPointOp op,
      mlir::PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//============================= Pattern 集合注册 =============================
namespace {
void populateOnnxToMyDialectConversionPatterns(RewritePatternSet &patterns,
    OnnxToMyDialectTypeConverter &typeConverter) {
  auto *ctx = patterns.getContext();
  patterns.add<ConvertONNXAdd>(ctx, typeConverter);
  patterns.add<ConvertONNXMul>(ctx, typeConverter);
  patterns.add<ConvertONNXEqual>(ctx, typeConverter);
  patterns.add<ConvertONNXConstant>(ctx, typeConverter);
  patterns.add<ConvertONNXIdentity>(ctx, typeConverter);
  patterns.add<ConvertONNXReshape>(ctx, typeConverter);
  patterns.add<ConvertONNXTranspose>(ctx, typeConverter);
  patterns.add<ConvertONNXReduceSum>(ctx, typeConverter);
  patterns.add<ConvertArithConstant>(ctx, typeConverter);
  patterns.add<ConvertONNXEntryPoint>(ctx);
  patterns.add<ConvertONNXLoopYield>(ctx, typeConverter);
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
    registry.insert<mlir::tensor::TensorDialect>(); // LoopYield 转换需要 tensor.extract
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OnnxToMyDialectTypeConverter typeConverter;

    // ======== Phase 1: RewritePattern 处理 ONNX Loop ========
    RewritePatternSet preLoopPatterns(&getContext());
    preLoopPatterns.add<ConvertONNXLoop>(&getContext(), typeConverter);
    (void)applyPatternsAndFoldGreedily(module, std::move(preLoopPatterns));

    // ======== Phase 2a: 所有剩余 ONNX op → mydialect op（RewritePattern）===
    // 全部用 RewritePattern + applyPatternsAndFoldGreedily，
    // 不经过 dialect conversion framework（它会强制 adaptor 和 materialization，
    // 导致 cast 套娃 + unresolved materialization 报错）。
    // 关键：这一步完成后，Loop body 内所有 ONNX op 都被替换成 mydialect op，
    // Loop body block arg 虽然还是 tensor 类型但已经没有 ONNX op 消费它了。
    {
      RewritePatternSet patterns(&getContext());
      populateOnnxToMyDialectConversionPatterns(patterns, typeConverter);
      (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
    }

    // ======== Phase 2b: Loop body block arg 类型转换 =======================
    // 此时 Loop body 里已经没有任何 ONNX op 了（Phase 2a 处理完），
    // 可以安全地把 body block arg 从 tensor setType 成 MyTensorType。
    // LoopOp 是 mydialect 合法 op，Phase 2c 的 func 签名转换不会碰它的 region。
    module.walk([&](mydialect::LoopOp loop) {
      mlir::Block &body = loop.getBody().front();
      for (unsigned i = 0; i < body.getNumArguments(); ++i) {
        mlir::BlockArgument arg = body.getArgument(i);
        mlir::Type oldTy = arg.getType();
        auto newTy = typeConverter.convertType(oldTy);
        if (!newTy || newTy == oldTy) continue;
        arg.setType(newTy);
      }
    });

    // ======== Phase 2c: func 签名 + return 转换（手动 walk）========
    module.walk([&](func::FuncOp op) {
      auto oldFnTy = op.getFunctionType();
      SmallVector<mlir::Type> newInputTys, newResultTys;
      for (auto t : oldFnTy.getInputs()) {
        auto nt = typeConverter.convertType(t);
        if (nt) newInputTys.push_back(nt);
        else return;
      }
      for (auto t : oldFnTy.getResults()) {
        auto nt = typeConverter.convertType(t);
        if (nt) newResultTys.push_back(nt);
        else return;
      }
      auto newFnTy = mlir::FunctionType::get(&getContext(), newInputTys, newResultTys);
      op.setType(newFnTy);
      // block arg 直接 setType（不插 cast，让 Phase 3 处理残留的 identity cast）
      mlir::Block &entry = op.getBody().front();
      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        auto newTy = newFnTy.getInput(i);
        entry.getArgument(i).setType(newTy);
      }
    });
    module.walk([&](func::ReturnOp op) {
      mlir::OpBuilder builder(op);
      SmallVector<mlir::Value> newOperands;
      for (auto v : op.getOperands()) {
        auto targetTy = typeConverter.convertType(v.getType());
        if (targetTy && targetTy != v.getType()) {
          newOperands.push_back(builder.create<mlir::UnrealizedConversionCastOp>(
              op.getLoc(), targetTy, v).getResult(0));
        } else {
          newOperands.push_back(v);
        }
      }
      builder.create<func::ReturnOp>(op.getLoc(), newOperands);
      op->erase();
    });

    // ======== Phase 3: 手动消掉残留的 cast ========
    for (int iter = 0; iter < 10; ++iter) {
      bool changed = false;
      SmallVector<mlir::UnrealizedConversionCastOp> toErase;
      module.walk([&](mlir::UnrealizedConversionCastOp cast) {
        // identity cast: 替换 uses + erase
        if (cast.getResult(0).getType() == cast.getOperand(0).getType()) {
          cast.getResult(0).replaceAllUsesWith(cast.getOperand(0));
          toErase.push_back(cast);
          changed = true;
          return;
        }
        // cast 套娃 A→B→A：替换外层 uses 为内层原操作数
        if (auto *innerOp = cast.getOperand(0).getDefiningOp()) {
          if (auto inner = mlir::dyn_cast_if_present<mlir::UnrealizedConversionCastOp>(innerOp)) {
            if (inner.getOperand(0).getType() == cast.getResult(0).getType()) {
              cast.getResult(0).replaceAllUsesWith(inner.getOperand(0));
              toErase.push_back(cast);
              changed = true;
              return;
            }
          }
        }
      });
      for (auto cast : toErase) cast->erase();
      if (!changed) break;
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
