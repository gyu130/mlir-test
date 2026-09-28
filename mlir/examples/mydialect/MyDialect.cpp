#include "MyDialect.h"

// TableGen 生成的 Dialect 定义：构造函数、析构函数、TypeID 定义。
// 构造函数内会调用 initialize()。
#include "mydialect.cpp.inc"
#include "mlir/IR/OpImplementation.h"
#include "mlir/AsmParser/AsmParser.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"

namespace {

struct PrintOpLowering : public mlir::OpRewritePattern<mydialect::PrintOp> {
  using OpRewritePattern<mydialect::PrintOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mydialect::PrintOp printOp,
                                      mlir::PatternRewriter &rewriter) const override {
  auto i32Ty = rewriter.getI32Type();
  // ✅ 修复：使用 LLVMVoidType，不是 getNoneType()
  auto voidTy = mlir::LLVM::LLVMVoidType::get(rewriter.getContext());
  auto funcTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {i32Ty});

  auto funcSym = mlir::SymbolRefAttr::get(printOp.getContext(), "my_runtime_print_i32");
  rewriter.create<mlir::LLVM::CallOp>(printOp.getLoc(), funcTy, funcSym, printOp.getOperand());
  rewriter.eraseOp(printOp);
  return mlir::success();


  }
};
} // anonymous namespace


namespace mydialect {

// 注册 TableGen 生成的所有 Op（PrintOp 等）
void MyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mydialect-opdefs.cpp.inc"
      >();
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

mlir::LogicalResult PrintOp::verify() {
	    mlir::Type opTy = getOperand().getType();
	    //auto intTy = dyn_cast<mlir::InterType>();
	    if (!opTy.isInteger()){
	        return emitOpError() << "operand must be integer type, but got " << opTy;         
	    }
	    if (opTy.getIntOrFloatBitWidth() != 32) {
        	return emitOpError() << "operand must be i32, got " << opTy;
    	    }

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


struct PrintOpToLLVMRewrite : public mlir::OpRewritePattern<mydialect::PrintOp> {
  using OpRewritePattern<mydialect::PrintOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mydialect::PrintOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    // 构造LLVM函数类型 void(i32)
    auto i32Ty = rewriter.getI32Type();
    auto voidTy = mlir::LLVM::LLVMVoidType::get(rewriter.getContext());
    auto funcTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {i32Ty});

    // 获取符号，对应我们宿主 extern "C" my_runtime_print_i32
    auto func = mlir::SymbolRefAttr::get(op.getContext(), "my_runtime_print_i32");
    rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), funcTy, func, op.getOperand());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};


// 对外暴露，mydriver里面可以调用这个pattern集合
void populatePrintOpToLLVMPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<PrintOpToLLVMRewrite>(patterns.getContext());
  //patterns.add<PrintOpLowering>(patterns.getContext());
}

} // namespace mydialect


// 生成各 Op 的方法定义（build/create/verify 等）及 TypeID 定义
#define GET_OP_CLASSES
#include "mydialect-opdefs.cpp.inc"

