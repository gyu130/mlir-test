#include "MyDialectOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define GET_OP_CLASSES
#include "mydialect-opdefs.cpp.inc"   // 引入完整类定义

using namespace mlir;
namespace mydialect {

// AddTensorOp/MulTensorOp/IdentityTensorOp 的 inferReturnTypes 由
// SameOperandsAndResultType trait 自动生成，无需手写。
// 此处仅保留 verify() 与其他需手写 inferReturnTypes 的 op。

// ---------------- AddTensorOp ----------------
LogicalResult AddTensorOp::verify() {
  auto lhsTy = llvm::dyn_cast<mydialect::MyTensorType>(getLhs().getType());
  auto rhsTy = llvm::dyn_cast<mydialect::MyTensorType>(getRhs().getType());
  if (!lhsTy) return emitOpError("lhs is not mydialect tensor type");
  if (!rhsTy) return emitOpError("rhs is not mydialect tensor type");

  if(lhsTy.getShape() != rhsTy.getShape())
    return emitOpError("lhs/rhs shape mismatch");
  if(lhsTy.getElementType() != rhsTy.getElementType())
    return emitOpError("lhs/rhs element type mismatch");
  return success();
}

// ---------------- MulTensorOp ----------------
LogicalResult MulTensorOp::verify() {
  auto lhsTy = llvm::dyn_cast<mydialect::MyTensorType>(getLhs().getType());
  auto rhsTy = llvm::dyn_cast<mydialect::MyTensorType>(getRhs().getType());
  if (!lhsTy) return emitOpError("lhs is not mydialect tensor type");
  if (!rhsTy) return emitOpError("rhs is not mydialect tensor type");

  if(lhsTy.getShape() != rhsTy.getShape())
    return emitOpError("lhs/rhs shape mismatch");
  if(lhsTy.getElementType() != rhsTy.getElementType())
    return emitOpError("lhs/rhs element type mismatch");
  return success();
}

// ---------------- FusedElementwiseOp ----------------
LogicalResult FusedElementwiseOp::verify() {
  auto resTy = llvm::dyn_cast<mydialect::MyTensorType>(getRes().getType());
  if (!resTy) return emitOpError("result is not mydialect tensor type");
  if (getInputs().empty())
    return emitOpError("expected at least one input tensor");

  // 所有输入张量与结果同 shape、同元素类型
  for (mlir::Value in : getInputs()) {
    auto inTy = llvm::dyn_cast<mydialect::MyTensorType>(in.getType());
    if (!inTy) return emitOpError("input is not mydialect tensor type");
    if (inTy.getShape() != resTy.getShape())
      return emitOpError("input/result shape mismatch");
    if (inTy.getElementType() != resTy.getElementType())
      return emitOpError("input/result element type mismatch");
  }

  // body：单块；块参数个数 == 输入个数，且类型 == 元素类型
  mlir::Block &blk = getBody().front();
  if (blk.getNumArguments() != getInputs().size())
    return emitOpError("body block argument count must equal input count");
  for (mlir::Value arg : blk.getArguments()) {
    if (arg.getType() != resTy.getElementType())
      return emitOpError("body block argument type must match element type");
  }

  // 终止符必须是 fused_yield，且产出值类型 == 元素类型
  auto yieldOp = llvm::dyn_cast<FusedYieldOp>(blk.getTerminator());
  if (!yieldOp)
    return emitOpError("body must be terminated by mydialect.fused_yield");
  if (yieldOp.getValue().getType() != resTy.getElementType())
    return emitOpError("fused_yield value type must match element type");
  return success();
}

// ---------------- DeviceExecOp ----------------
// region 终止符必须是 device_yield，且产出值个数/类型与 op 结果一致
LogicalResult DeviceExecOp::verify() {
  mlir::Block &blk = getBody().front();
  auto yieldOp = llvm::dyn_cast<DeviceYieldOp>(blk.getTerminator());
  if (!yieldOp)
    return emitOpError("body must be terminated by mydialect.device_yield");
  if (yieldOp.getNumOperands() != getNumResults())
    return emitOpError("device_yield operand count must equal result count");
  for (size_t i = 0; i < getNumResults(); ++i) {
    if (yieldOp.getOperand(i).getType() != getResult(i).getType())
      return emitOpError("device_yield operand type must match result type");
  }
  return success();
}

// ---------------- CmpITensorOp ----------------
// 结果类型：与输入同 shape，但元素类型为 i1
LogicalResult CmpITensorOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto lhsTy = mlir::cast<mydialect::MyTensorType>(operands[0].getType());
  auto shapeAttr = lhsTy.getShape();
  auto i1Ty = IntegerType::get(ctx, 1);
  auto outTy = MyTensorType::get(ctx, shapeAttr, i1Ty);
  inferredReturnTypes.push_back(outTy);
  return success();
}

LogicalResult CmpITensorOp::verify() {
  auto lhsTy = llvm::dyn_cast<mydialect::MyTensorType>(getLhs().getType());
  auto rhsTy = llvm::dyn_cast<mydialect::MyTensorType>(getRhs().getType());
  if (!lhsTy) return emitOpError("lhs is not mydialect tensor type");
  if (!rhsTy) return emitOpError("rhs is not mydialect tensor type");

  if(lhsTy.getShape() != rhsTy.getShape())
    return emitOpError("lhs/rhs shape mismatch");
  if(lhsTy.getElementType() != rhsTy.getElementType())
    return emitOpError("lhs/rhs element type mismatch");

  int64_t pred = static_cast<int64_t>(getPredicate());
  if (pred < 0 || pred > 6)
    return emitOpError("invalid comparison predicate value");
  return success();
}

// ---------------- ConstantTensorOp ----------------
LogicalResult ConstantTensorOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  // value 属性存储在 Properties 中（inherent attr），不在 attrs DictionaryAttr
  auto &props = *properties.as<const Properties *>();
  auto valueAttr = props.value;
  auto st = valueAttr.getShapedType();
  SmallVector<Attribute> shapeAttrVec;
  for(int64_t d : st.getShape()) {
    shapeAttrVec.push_back(IntegerAttr::get(IndexType::get(ctx), d));
  }
  auto shapeAttr = ArrayAttr::get(ctx, shapeAttrVec);
  auto outTy = MyTensorType::get(ctx, shapeAttr, st.getElementType());
  inferredReturnTypes.push_back(outTy);
  return success();
}

LogicalResult ConstantTensorOp::getConstantValue(Attribute &value) {
  auto props = getProperties();
  value = props.value;
  return success();
}

LogicalResult ConstantTensorOp::verify() {
  auto resTy = llvm::dyn_cast<mydialect::MyTensorType>(getRes().getType());
  auto elemAttr = getValue();
  if (!resTy) return emitOpError("result type is not mydialect tensor");

  if (resTy.getElementType() != elemAttr.getElementType())
    return emitOpError("element type mismatch between tensor type and constant attr");
  return success();
}

// ---------------- IdentityTensorOp ----------------
LogicalResult IdentityTensorOp::verify() {
  auto inTy = llvm::dyn_cast<mydialect::MyTensorType>(getIn().getType());
  auto resTy = llvm::dyn_cast<mydialect::MyTensorType>(getRes().getType());
  if (!inTy) return emitOpError("input is not mydialect tensor type");
  if (!resTy) return emitOpError("result is not mydialect tensor type");

  if(inTy != resTy)
    return emitOpError("input and result tensor type must be identical");
  return success();
}

// ---------------- ReshapeTensorOp ----------------
// 不实现 inferReturnTypes：结果类型由 IR 显式指定，pattern 转换时显式构建。
LogicalResult ReshapeTensorOp::verify() {
  auto dataTy = llvm::dyn_cast<mydialect::MyTensorType>(getData().getType());
  auto shapeTy = llvm::dyn_cast<mydialect::MyTensorType>(getShape().getType());
  auto resTy = llvm::dyn_cast<mydialect::MyTensorType>(getRes().getType());
  if (!dataTy) return emitOpError("data operand is not mydialect tensor");
  if (!shapeTy) return emitOpError("shape operand is not mydialect tensor");
  if (!resTy) return emitOpError("result is not mydialect tensor");

  if (shapeTy.getShape().size() != 1)
    return emitOpError("shape tensor must be rank-1");
  return success();
}

// ---------------- TransposeTensorOp ----------------
LogicalResult TransposeTensorOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto dataTy = llvm::cast<mydialect::MyTensorType>(operands[0].getType());
  // perm 属性存储在 Properties 中（inherent attr），不在 attrs DictionaryAttr
  auto &props = *properties.as<const Properties *>();
  auto permAttr = props.perm;
  SmallVector<Attribute> newShape;
  auto oldShape = dataTy.getShape();
  for(auto permItem : permAttr) {
    int64_t idx = llvm::cast<mlir::IntegerAttr>(permItem).getInt();
    newShape.push_back(oldShape[idx]);
  }
  auto outTy = MyTensorType::get(ctx, ArrayAttr::get(ctx, newShape), dataTy.getElementType());
  inferredReturnTypes.push_back(outTy);
  return success();
}

LogicalResult TransposeTensorOp::verify() {
  auto dataTy = llvm::dyn_cast<mydialect::MyTensorType>(getData().getType());
  auto resTy = llvm::dyn_cast<mydialect::MyTensorType>(getRes().getType());
  auto permAttr = getPerm();
  if (!dataTy) return emitOpError("data operand is not mydialect tensor");
  if (!resTy) return emitOpError("result is not mydialect tensor");

  int64_t rank = dataTy.getShape().size();
  if (permAttr.size() != rank)
    return emitOpError("perm array length must equal tensor rank");

  for (auto item : permAttr) {
    int64_t dimIdx = llvm::cast<mlir::IntegerAttr>(item).getInt();
    if (dimIdx < 0 || dimIdx >= rank)
      return emitOpError("transpose perm index out of range");
  }
  return success();
}

// ---------------- ReduceSumTensorOp ----------------
LogicalResult ReduceSumTensorOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto dataTy = llvm::cast<mydialect::MyTensorType>(operands[0].getType());
  // axes/keep_dims 属性存储在 Properties 中（inherent attr），不在 attrs DictionaryAttr
  auto &props = *properties.as<const Properties *>();
  auto axesAttr = props.axes;
  auto keepDimsAttr = props.keep_dims;
  bool keepDims = keepDimsAttr ? keepDimsAttr.getValue() : false;
  auto inShape = dataTy.getShape();
  SmallVector<Attribute> outShapeAttrs;
  for(auto idx : llvm::seq<int64_t>(0, inShape.size())) {
    bool isReduce = llvm::any_of(axesAttr, [&](Attribute a){
      return llvm::cast<mlir::IntegerAttr>(a).getInt() == idx;
    });
    if(!isReduce) {
      outShapeAttrs.push_back(inShape[idx]);
    } else if(keepDims) {
      outShapeAttrs.push_back(IntegerAttr::get(IndexType::get(ctx), 1));
    }
  }
  auto outTy = mydialect::MyTensorType::get(ctx, ArrayAttr::get(ctx, outShapeAttrs), dataTy.getElementType());
  inferredReturnTypes.push_back(outTy);
  return success();
}

LogicalResult ReduceSumTensorOp::verify() {
  auto dataTy = llvm::dyn_cast<mydialect::MyTensorType>(getData().getType());
  auto resTy = llvm::dyn_cast<mydialect::MyTensorType>(getRes().getType());
  auto axesAttr = getAxes();
  if (!dataTy) return emitOpError("data operand is not mydialect tensor");
  if (!resTy) return emitOpError("result is not mydialect tensor");

  int64_t rank = dataTy.getShape().size();
  for (auto item : axesAttr) {
    int64_t axis = llvm::cast<mlir::IntegerAttr>(item).getInt();
    if (axis < 0 || axis >= rank)
      return emitOpError("reduce axis out of tensor rank range");
  }
  return success();
}

// ---------------- LoopOp ----------------
LogicalResult LoopOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty())
    return emitOpError("loop body region must not be empty");
  Block &bodyBlock = bodyRegion.front();
  if (!bodyBlock.mightHaveTerminator())
    return emitOpError("loop body must have a terminator");
  auto yieldOp = llvm::dyn_cast<LoopYieldOp>(bodyBlock.getTerminator());
  if (!yieldOp)
    return emitOpError("loop body must end with mydialect.loop_yield");

  // 1. yield carryArgs 数量必须等于 initArgs 数量
  size_t carryNum = getInitArgs().size();
  if (yieldOp.getCarryArgs().size() != carryNum)
    return emitOpError("yield carry args count must match loop initArgs count");
  // 2. results 数量必须等于 initArgs 数量
  if (getResults().size() != carryNum)
    return emitOpError("loop results count must match initArgs count");

  // 3. 类型校验：yield carry 张量类型与 loop 结果类型一致
  //    接受 MyTensorType 或 RankedTensorType（ONNX 转换时 body 里的 carry 可能
  //    还是 builtin tensor，Phase 2 的 onnx.Add pattern 会转换它们）
  auto getShapeElem = [](mlir::Type ty) -> std::optional<std::pair<mlir::ArrayRef<int64_t>, mlir::Type>> {
    if (auto mtt = llvm::dyn_cast<mydialect::MyTensorType>(ty)) {
      SmallVector<int64_t> shape;
      for (auto a : mtt.getShape())
        shape.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());
      return std::make_pair(mlir::ArrayRef<int64_t>(shape), mtt.getElementType());
    }
    if (auto rtt = llvm::dyn_cast<mlir::RankedTensorType>(ty))
      return std::make_pair(rtt.getShape(), rtt.getElementType());
    return std::nullopt;
  };
  for (auto [yieldVal, resVal] :
       llvm::zip(yieldOp.getCarryArgs(), getResults())) {
    auto ys = getShapeElem(yieldVal.getType());
    auto rs = getShapeElem(resVal.getType());
    if (!ys || !rs)
      return emitOpError("carry/result must be tensor type");
    if (ys->first != rs->first || ys->second != rs->second)
      return emitOpError("yield carry tensor type mismatch with loop result type");
  }
  return success();
}

// ---------------- LoopYieldOp ----------------
LogicalResult LoopYieldOp::verify() {
  // cond：接受 i1 标量 或 0维 tensor<i1>
  auto ty = getCond().getType();
  bool condOk = ty.isInteger(1);
  if (!condOk) {
    if (auto rtt = llvm::dyn_cast<mlir::RankedTensorType>(ty))
      condOk = (rtt.getRank() == 0 && rtt.getElementType().isInteger(1));
    else if (auto mtt = llvm::dyn_cast<mydialect::MyTensorType>(ty))
      condOk = (mtt.getShape().empty() && mtt.getElementType().isInteger(1));
  }
  if (!condOk)
    return emitOpError("cond operand must be i1 or 0-d tensor<i1>");
  // carry args：接受 MyTensorType 或 RankedTensorType
  for (auto carry : getCarryArgs()) {
    if (!llvm::isa<mydialect::MyTensorType, mlir::RankedTensorType>(carry.getType()))
      return emitOpError("carry argument must be tensor type");
  }
  return success();
}

} // namespace mydialect
