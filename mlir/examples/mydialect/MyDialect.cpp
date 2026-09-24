#include "MyDialect.h"

// TableGen 生成的 Dialect 定义：构造函数、析构函数、TypeID 定义。
// 构造函数内会调用 initialize()。
#include "mydialect.cpp.inc"
#include "mlir/IR/OpImplementation.h"
#include "mlir/AsmParser/AsmParser.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/SymbolTable.h"

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
	    if (!opTy.isInteger()){
	        return emitOpError() << "operand must be integer type, but got " << opTy;         
	    }
	    if (opTy.getIntOrFloatBitWidth() != 32) {
        	return emitOpError() << "operand must be i32, got " << opTy;
    	    }

	    return mlir::success();
     }

struct PrintOpToLLVMRewrite : public mlir::OpRewritePattern<mydialect::PrintOp> {
  using OpRewritePattern<mydialect::PrintOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mydialect::PrintOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    // 构造LLVM函数类型 void(i32)
    auto i32Ty = rewriter.getI32Type();
    auto funcTy = mlir::LLVM::LLVMFunctionType::get(rewriter.getNoneType(), {i32Ty});

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
}

} // namespace mydialect


// 生成各 Op 的方法定义（build/create/verify 等）及 TypeID 定义
#define GET_OP_CLASSES
#include "mydialect-opdefs.cpp.inc"
