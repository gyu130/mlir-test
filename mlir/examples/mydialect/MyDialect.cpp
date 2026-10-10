#include "MyDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

// TableGen 生成的 Dialect 定义：构造函数、析构函数、TypeID 定义。
// 构造函数内会调用 initialize()。
// TableGen 生成的自定义类型定义（WrappedI32Type）
// 放在 Dialect 定义 .inc 之前，让类型存储器/接口可后于 op 定义链接
#include "mydialect.cpp.inc"
#include "mydialect.typedefs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "mydialect.typedefs.cpp.inc"
#include "mlir/IR/OpImplementation.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mydialect {

// 注册 TableGen 生成的所有 Op 和自定义类型
void MyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mydialect-opdefs.cpp.inc"
      >();
  // WrappedI32Type：TableGen 生成 WrappedI32Type 类与 generatedTypeParser/
  // generatedTypePrinter 静态函数（见 mydialect.typedefs.cpp.inc），
  // parseType/printType 中调用它们。
  addTypes<
#define GET_TYPEDEF_LIST
#include "mydialect.typedefs.cpp.inc"
      >();
}

// 类型解析：mydialect.wrapped_i32
mlir::Type MyDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return nullptr;
  if (keyword == "wrapped_i32")
    return WrappedI32Type::get(parser.getContext());
  parser.emitError(parser.getNameLoc(), "unknown mydialect type: ") << keyword;
  return nullptr;
}

// 类型打印：直接输出 wrapped_i32
void MyDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  if (mlir::isa<WrappedI32Type>(type)) {
    printer << "wrapped_i32";
    return;
  }
  llvm_unreachable("unknown type in mydialect printType");
}

// -------- custom assembly for PrintOp --------
void PrintOp::print(mlir::OpAsmPrinter &printer) {
auto &os = printer.getStream();
  //os << "mydialect.print ";
  printer << " ";
  printer.printOperand(this->getValue());
  os << " : ";
  printer.printType(this->getValue().getType());

}

mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand valueOperand;
  mlir::Type valueType;

  if (parser.parseOperand(valueOperand))
    return mlir::failure();
  if (parser.parseColonType(valueType))
    return mlir::failure();
  if (parser.resolveOperand(valueOperand, valueType, result.operands))
    return mlir::failure();
  return mlir::success();
}

// -------- ForOp helpers & custom assembly --------
mlir::Value ForOp::getInductionVar() {
  return getBody().getArgument(0);
}

mlir::ValueRange ForOp::getRegionIterArgs() {
  return getBody().getArguments().drop_front();
}

void ForOp::print(mlir::OpAsmPrinter &p) {
  p << " " << getInductionVar() << " = " << getLower() << " to "
    << getUpper() << " step " << getStep();

  auto initArgs = getIterArgs();
  auto regionArgs = getRegionIterArgs();
  if (!initArgs.empty()) {
    p << " iter_args(";
    for (unsigned i = 0; i < initArgs.size(); ++i) {
      if (i > 0) p << ", ";
      p << regionArgs[i] << " = " << initArgs[i];
    }
    p << ")";
    p << " -> (" << initArgs.getTypes() << ")";
  }
  p << ' ';
  // 类型标注（本方言中 iv 与边界均为 i32）
  p << ": " << getLower().getType() << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!initArgs.empty());
  p.printOptionalAttrDict((*this)->getAttrs());
}

mlir::ParseResult ForOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  mlir::OpAsmParser::Argument iv;
  mlir::OpAsmParser::UnresolvedOperand lb, ub, step;

  // %iv = %lb to %ub step %step
  if (parser.parseOperand(iv.ssaName) || parser.parseEqual() ||
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return mlir::failure();

  // 可选 iter_args
  llvm::SmallVector<mlir::OpAsmParser::Argument, 4> regionArgs;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  regionArgs.push_back(iv);

  bool hasIterArgs = succeeded(parser.parseOptionalKeyword("iter_args"));
  if (hasIterArgs) {
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return mlir::failure();
  }

  // ": i32"
  mlir::Type type;
  if (parser.parseColon() || parser.parseType(type))
    return mlir::failure();

  // 设置 region 块参数类型
  regionArgs.front().type = type; // iv
  for (auto [iterArg, resType] :
       llvm::zip_equal(llvm::drop_begin(regionArgs), result.types))
    iterArg.type = resType;

  // 解析循环体
  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return mlir::failure();

  // 确保有 YieldOp terminator
  if (!body->empty()) {
    mlir::Block &blk = body->front();
    if (blk.empty() || !llvm::isa<mydialect::YieldOp>(blk.back())) {
      mlir::OpBuilder builder(parser.getContext());
      builder.setInsertionPointToEnd(&blk);
      mydialect::YieldOp::create(builder, result.location, mlir::ValueRange{});
    }
  }

  // 解析操作数
  if (parser.resolveOperand(lb, type, result.operands) ||
      parser.resolveOperand(ub, type, result.operands) ||
      parser.resolveOperand(step, type, result.operands))
    return mlir::failure();

  if (hasIterArgs) {
    for (auto [arg, operand, resType] :
         llvm::zip_equal(llvm::drop_begin(regionArgs), operands, result.types)) {
      arg.type = resType;
      if (parser.resolveOperand(operand, resType, result.operands))
        return mlir::failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  return mlir::success();
}

// divi_s verifier
mlir::LogicalResult mydialect::DivSIOp::verify() {
  mlir::Operation *op = getOperation();
  mlir::Value rhs = op->getOperand(1);
  mlir::APInt val;
  if (mlir::matchPattern(rhs, mlir::m_ConstantInt(&val))) {
    if (val.isZero()) {
      return op->emitOpError() << "divi_s: division by zero constant";
    }
  }
  return mlir::success();
}

mlir::LogicalResult CmpiOp::verify() {
  auto pred = getPredicate();
  llvm::SmallVector<llvm::StringRef,6> validPreds = {"slt","sgt","sle","sge","eq","ne"};
  bool ok = llvm::is_contained(validPreds, pred);
  if (!ok)
    return emitOpError() << "invalid predicate " << pred;
  return mlir::success();
}

// ---- AllocOp verify ----
mlir::LogicalResult AllocOp::verify() {
  auto memRefType = mlir::cast<mlir::MemRefType>(getRes().getType());
  if(memRefType.getRank() != 1)
    return emitError() << "alloc only support 1-D memref";
  if(memRefType.getElementType() != mlir::IntegerType::get(getContext(), 32))
    return emitError() << "alloc only support i32 element";
  return mlir::success();
}

// ---- LoadOp verify ----
mlir::LogicalResult LoadOp::verify() {
  auto mty = mlir::cast<mlir::MemRefType>(getMemref().getType());
  if(mty.getRank() != 1)
    return emitError() << "load only support 1-D memref";
  if(mty.getElementType() != getRes().getType())
    return emitError() << "load element type mismatch";
  return mlir::success();
}

// ---- StoreOp verify ----
mlir::LogicalResult StoreOp::verify() {
  auto mty = mlir::cast<mlir::MemRefType>(getMemref().getType());
  if(mty.getRank() != 1)
    return emitError() << "store only support 1-D memref";
  if(mty.getElementType() != getValue().getType())
    return emitError() << "store element type mismatch";
  return mlir::success();
}

// AddI32Op: fold(lhs + rhs)
::mlir::OpFoldResult mydialect::AddI32Op::fold(FoldAdaptor adaptor) {
  if (!adaptor.getLhs() || !adaptor.getRhs())
    return nullptr;
  auto lhsAttr = mlir::dyn_cast<::mlir::IntegerAttr>(adaptor.getLhs());
  auto rhsAttr = mlir::dyn_cast<::mlir::IntegerAttr>(adaptor.getRhs());
  if (!lhsAttr || !rhsAttr)
    return nullptr;

  int32_t a = lhsAttr.getValue().getSExtValue();
  int32_t b = rhsAttr.getValue().getSExtValue();
  auto resType = getResult().getType();
  return ::mlir::IntegerAttr::get(resType, a + b);
}

// MuliOp: fold(lhs * rhs)
::mlir::OpFoldResult mydialect::MuliOp::fold(FoldAdaptor adaptor) {
  if (!adaptor.getLhs() || !adaptor.getRhs())
    return nullptr;
  auto lhsAttr = mlir::dyn_cast<::mlir::IntegerAttr>(adaptor.getLhs());
  auto rhsAttr = mlir::dyn_cast<::mlir::IntegerAttr>(adaptor.getRhs());
  if (!lhsAttr || !rhsAttr)
    return nullptr;

  int32_t a = lhsAttr.getValue().getSExtValue();
  int32_t b = rhsAttr.getValue().getSExtValue();
  auto resType = getResult().getType();
  return ::mlir::IntegerAttr::get(resType, a * b);
}

// SubI32Op: fold(lhs - rhs)
::mlir::OpFoldResult mydialect::SubI32Op::fold(FoldAdaptor adaptor) {
  if (!adaptor.getLhs() || !adaptor.getRhs())
    return nullptr;
  auto lhsAttr = mlir::dyn_cast<::mlir::IntegerAttr>(adaptor.getLhs());
  auto rhsAttr = mlir::dyn_cast<::mlir::IntegerAttr>(adaptor.getRhs());
  if (!lhsAttr || !rhsAttr)
    return nullptr;

  int32_t a = lhsAttr.getValue().getSExtValue();
  int32_t b = rhsAttr.getValue().getSExtValue();
  auto resType = getResult().getType();
  return ::mlir::IntegerAttr::get(resType, a - b);
}

// DivSIOp: fold(lhs / rhs)，除数为0不折叠
::mlir::OpFoldResult mydialect::DivSIOp::fold(FoldAdaptor adaptor) {
  if (!adaptor.getLhs() || !adaptor.getRhs())
    return nullptr;
  auto lhsAttr = mlir::dyn_cast<::mlir::IntegerAttr>(adaptor.getLhs());
  auto rhsAttr = mlir::dyn_cast<::mlir::IntegerAttr>(adaptor.getRhs());
  if (!lhsAttr || !rhsAttr)
    return nullptr;

  int32_t a = lhsAttr.getValue().getSExtValue();
  int32_t b = rhsAttr.getValue().getSExtValue();
  if (b == 0)
    return nullptr;

  auto resType = getResult().getType();
  return ::mlir::IntegerAttr::get(resType, a / b);
}

/// 常量折叠接口：ForOp
mlir::LogicalResult ForOp::fold(FoldAdaptor adaptor,
                                llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
  if (!adaptor.getLower() || !adaptor.getUpper() || !adaptor.getStep())
    return mlir::failure();
  auto lowerAttr = mlir::dyn_cast<mlir::IntegerAttr>(adaptor.getLower());
  auto upperAttr = mlir::dyn_cast<mlir::IntegerAttr>(adaptor.getUpper());
  auto stepAttr  = mlir::dyn_cast<mlir::IntegerAttr>(adaptor.getStep());
  if(!lowerAttr || !upperAttr || !stepAttr){
      return mlir::failure();
  }
  int32_t lo = (int32_t)lowerAttr.getInt();
  int32_t hi = (int32_t)upperAttr.getInt();
  int32_t st = (int32_t)stepAttr.getInt();

  // case1：循环一次都不执行，直接返回iter_args初始值
  bool noIteration = ((st>0 && lo >= hi) || (st<0 && lo <= hi));
  if(noIteration){
      // results 填充所有 iter_args 操作数
      for(size_t i=3; i < getNumOperands(); ++i){
          results.push_back(getOperand(i));
      }
      return mlir::success();
  }

  // case2：有限小常量循环完整求值由 ConstantEvaluateForPattern 完成（见 pass.cpp）
  return mlir::failure();
}

} // namespace mydialect

// ============== canonicalize patterns（TD 中 hasCanonicalizer = 1）==============
namespace {

// 判断 Value 是否为值为 intVal 的常量整数
static bool isConstantInt(mlir::Value v, int64_t intVal) {
  mlir::APInt cst;
  return mlir::matchPattern(v, mlir::m_ConstantInt(&cst)) &&
         cst.getSExtValue() == intVal;
}

// addi: x + 0 -> x / 0 + x -> x
struct AddiZeroCanon : public mlir::OpRewritePattern<mydialect::AddI32Op> {
  using OpRewritePattern<mydialect::AddI32Op>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mydialect::AddI32Op op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (isConstantInt(op.getRhs(), 0)) {
      rewriter.replaceOp(op, op.getLhs());
      return mlir::success();
    }
    if (isConstantInt(op.getLhs(), 0)) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }
    return mlir::failure();
  }
};

// subi: x - 0 -> x
struct SubiZeroCanon : public mlir::OpRewritePattern<mydialect::SubI32Op> {
  using OpRewritePattern<mydialect::SubI32Op>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mydialect::SubI32Op op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (isConstantInt(op.getRhs(), 0)) {
      rewriter.replaceOp(op, op.getLhs());
      return mlir::success();
    }
    return mlir::failure();
  }
};

// muli: x * 1 -> x / 1 * x -> x；x * 0 -> 0 / 0 * x -> 0
struct MuliOneCanon : public mlir::OpRewritePattern<mydialect::MuliOp> {
  using OpRewritePattern<mydialect::MuliOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mydialect::MuliOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (isConstantInt(op.getRhs(), 1)) {
      rewriter.replaceOp(op, op.getLhs());
      return mlir::success();
    }
    if (isConstantInt(op.getLhs(), 1)) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }
    // x * 0 -> 0
    if (isConstantInt(op.getRhs(), 0) || isConstantInt(op.getLhs(), 0)) {
      auto zero = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(), rewriter.getI32IntegerAttr(0));
      rewriter.replaceOp(op, zero.getResult());
      return mlir::success();
    }
    return mlir::failure();
  }
};

// divi_s: x / 1 -> x
struct DiviOneCanon : public mlir::OpRewritePattern<mydialect::DivSIOp> {
  using OpRewritePattern<mydialect::DivSIOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mydialect::DivSIOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (isConstantInt(op.getRhs(), 1)) {
      rewriter.replaceOp(op, op.getLhs());
      return mlir::success();
    }
    return mlir::failure();
  }
};

} // anonymous namespace

namespace mydialect {

void AddI32Op::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                           mlir::MLIRContext *context) {
  results.add<AddiZeroCanon>(context);
}

void SubI32Op::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                           mlir::MLIRContext *context) {
  results.add<SubiZeroCanon>(context);
}

void MuliOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                         mlir::MLIRContext *context) {
  results.add<MuliOneCanon>(context);
}

void DivSIOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                          mlir::MLIRContext *context) {
  results.add<DiviOneCanon>(context);
}

} // namespace mydialect

namespace mydialect {
#define GEN_PASS_REGISTRATION
#include "MyDialectPasses.h.inc"

void registerMyDialectPasses() {
  registerPasses();
}
} // namespace mydialect

// 生成各 Op 的方法定义（build/create/verify 等）及 TypeID 定义
#define GET_OP_CLASSES
#include "mydialect-opdefs.cpp.inc"

