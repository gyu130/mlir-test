#include "SimpleCParser.h"
#include "MyDialect.h"
#include "MyDialectOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include <cctype>

using namespace mlir;
namespace mydialect {

SimpleCParser::SimpleCParser(OpBuilder &b, ModuleOp mod) : builder(b), module(mod) {}

Token SimpleCParser::nextToken() {
  // skip whitespace
  while (pos < src.size() && std::isspace(src[pos])) pos++;
  if (pos >= src.size()) return {TokenKind::Eof, "", 0};
  char c = src[pos];
  Token t;
  t.line = 0;
  t.str = ""; // 初始化清空
  if (std::isalpha(c) || c == '_') {
    size_t start = pos;
    while (pos < src.size() && (std::isalnum(src[pos]) || src[pos] == '_')) pos++;
    t.str = src.substr(start, pos - start);
    t.kind = llvm::StringSwitch<TokenKind>(t.str)
      .Case("int", TokenKind::IntKw)
      .Case("main", TokenKind::MainKw)
      .Case("print", TokenKind::PrintKw)
      .Case("return", TokenKind::ReturnKw)
      .Default(TokenKind::Ident);
    return t;
  }
  if (std::isdigit(c)) {
    size_t start = pos;
    while (pos < src.size() && std::isdigit(src[pos])) pos++;
    t.str = src.substr(start, pos - start);
    t.kind = TokenKind::Number;
    return t;
  }
  // ========== 单字符符号，每个case设置str并且return ==========
  switch(c) {
    case '+': pos++; t.kind = TokenKind::Plus; t.str = "+"; break;
    case '-': pos++; t.kind = TokenKind::Minus; t.str = "-"; break;
    case '*': pos++; t.kind = TokenKind::Mul; t.str = "*"; break;
    case '/': pos++; t.kind = TokenKind::Div; t.str = "/"; break;
    case '(': pos++; t.kind = TokenKind::LParen; t.str = "("; break;
    case ')': pos++; t.kind = TokenKind::RParen; t.str = ")"; break;
    case '{': pos++; t.kind = TokenKind::LBrace; t.str = "{"; break;
    case '}': pos++; t.kind = TokenKind::RBrace; t.str = "}"; break;
    case ';': pos++; t.kind = TokenKind::Semi; t.str = ";"; break;
    case ',': pos++; t.kind = TokenKind::Comma; t.str = ","; break;
    case '=': pos++; t.kind = TokenKind::Eq; t.str = "="; break;
    default:
      t.kind = TokenKind::Eof;
      t.str = "";
      return t;
  }
  return t;
}

void SimpleCParser::consume() {
  curTok = nextToken();
}
bool SimpleCParser::expect(TokenKind k) {
  if (curTok.kind == k) { consume(); return true; }
  return false;
}

Value SimpleCParser::makeConstI32(int64_t val) {
  // 先构造一个内置的 tensor<i32> 类型
  mlir::RankedTensorType builtinTensorTy = mlir::RankedTensorType::get({}, builder.getI32Type());
  // 0维标量tensor的DenseElementsAttr
  int32_t scalarVal = static_cast<int32_t>(val);
  
  llvm::SmallVector<int32_t, 1> data = {scalarVal};
  auto attr = mlir::DenseElementsAttr::get(builtinTensorTy, llvm::ArrayRef<int32_t>(data));
  // 2. 构造你自定义的 MyTensorType：shape为空，元素i32
  auto shapeAttr = mlir::ArrayAttr::get(builder.getContext(), {});
  auto ty = mydialect::MyTensorType::get(builder.getContext(), shapeAttr, builder.getI32Type());
  // 然后创建 constant_tensor op，返回 MyTensorType
  auto constOp = builder.create<mydialect::ConstantTensorOp>(builder.getUnknownLoc(), ty, attr);
  return constOp.getResult();	
}

Value SimpleCParser::makeBinOp(Value lhs, Value rhs, TokenKind op) {
  auto loc = builder.getUnknownLoc();
  mlir::ImplicitLocOpBuilder ilb(loc, builder);	
  switch(op) {
    case TokenKind::Plus:
      return mydialect::AddTensorOp::create(ilb, lhs.getType(), lhs, rhs);
    case TokenKind::Mul:
      return mydialect::MulTensorOp::create(ilb, lhs.getType(), lhs, rhs);
    // 你后续再加 sub/div
    default:
      llvm::errs() << "unsupported bin op\n";
      return nullptr;
  }
}

LogicalResult SimpleCParser::parse(const std::string &src_) {
  src = src_;
  pos = 0;
  consume();
  return parseFunc();
}

LogicalResult SimpleCParser::parseFunc() {
  // int main() { ... }
  if (!expect(TokenKind::IntKw)) return failure();
  if (!expect(TokenKind::MainKw)) return failure();
  if (!expect(TokenKind::LParen)) return failure();
  if (!expect(TokenKind::RParen)) return failure();
  if (!expect(TokenKind::LBrace)) return failure();

  // 创建 func.func @main_graph() -> MyTensorType<i32>
  // 注意：shape 参数必须显式传空 ArrayAttr；直接写 {} 会 value-initialize
  // 出 null ArrayAttr，打印时触发 MyTensorType::print 的 invalid-shape 分支
  auto funcTy = FunctionType::get(builder.getContext(), {}, {});
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "main_graph", funcTy);
  module.push_back(func);
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  while (curTok.kind != TokenKind::RBrace && curTok.kind != TokenKind::Eof) {
    if (failed(parseStmt(entry))) return failure();
  }
  if (!expect(TokenKind::RBrace)) return failure();

  // main 函数末尾必须有 func.return terminator
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  return success();
}

LogicalResult SimpleCParser::parseStmt(Block *block) {
   switch(curTok.kind) {
    case TokenKind::IntKw:
      return parseDecl(block);
    case TokenKind::PrintKw:
        return parseCall(block);
    case TokenKind::Ident:
      // Ident 也可能是表达式（变量）开头的表达式语句
    case TokenKind::Number:
    case TokenKind::LParen:
      // 数字 / ( 开头，属于表达式语句
      break;
    case TokenKind::ReturnKw:
      return parseReturn(block);
    default:
      llvm::errs() << "bad stmt\n";
      return failure();
  }
  // 走到这里：当做表达式语句 expr ;
  Value v = parseExpr();
  if (!v) return failure();
  if (!expect(TokenKind::Semi)) return failure();
  // 裸表达式，没有赋值，值丢弃；MLIR里直接啥也不用创建
  return success();
}

LogicalResult SimpleCParser::parseDecl(Block *block) {
  consume(); // int
  std::string name = curTok.str;
  if (!expect(TokenKind::Ident)) return failure();
  Value val;
  if (curTok.kind == TokenKind::Eq) {
    consume();
    val = parseExpr();
    if (!val) return failure();
  } else {
    llvm::errs() << "need init\n";
    return failure();
  }
  if (!expect(TokenKind::Semi)) {
    llvm::errs() << "[parseDecl expect semi failed! curTok str=[" << curTok.str << "]\n";
    return failure();
  }
  symTable[name] = val;
  return success();
}

LogicalResult SimpleCParser::parseCall(Block *block) {
  consume(); // print
  if (!expect(TokenKind::LParen)) {
    llvm::errs() << "expect '(' after print\n";
    return failure();
  }
  Value arg = parseExpr();
  if (!arg) return failure();
  if (!expect(TokenKind::RParen)) {
    llvm::errs() << "expect ')' after print arg\n";
    return failure();
  }
  if (!expect(TokenKind::Semi)) {
    llvm::errs() << "expect ';' after print\n";
    return failure();
  }

  builder.create<PrintOp>(builder.getUnknownLoc(), arg);
  return success();
}

LogicalResult SimpleCParser::parseReturn(Block *block) {
  consume(); // return
  Value retVal = parseExpr();
  if (!retVal) return failure();
  if (!expect(TokenKind::Semi)) return failure();
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), retVal);
  return success();
}

Value SimpleCParser::parseExpr() {
  Value v = parseMulExpr();
  while (curTok.kind == TokenKind::Plus || curTok.kind == TokenKind::Minus) {
    TokenKind op = curTok.kind;
    consume();
    Value rhs = parseMulExpr();
    v = makeBinOp(v, rhs, op);
  }
  return v;
}

Value SimpleCParser::parseMulExpr() {
  Value v = parsePrimary();
  while (curTok.kind == TokenKind::Mul || curTok.kind == TokenKind::Div) {
    TokenKind op = curTok.kind;
    consume();
    Value rhs = parsePrimary();
    v = makeBinOp(v, rhs, op);
  }
  return v;
}

Value SimpleCParser::parsePrimary() {
  Value res;
  if (curTok.kind == TokenKind::Number) {
    int64_t num = std::stoll(curTok.str);
    consume();
    res = makeConstI32(num);
  } else if (curTok.kind == TokenKind::Ident) {
    std::string nm = curTok.str;
    consume();

    // hw_mul(a, b) —— 硬件乘法表达式
    // SimpleCParser 全程用 MyTensorType，HwMulOp 是纯标量 op。
    // 用 TensorToScalarOp 提取 → HwMulOp → ScalarToTensorOp wrap。
    if (nm == "hw_mul" && curTok.kind == TokenKind::LParen) {
      consume(); // (
      Value lhs = parseExpr();
      if (!expect(TokenKind::Comma)) {
        llvm::errs() << "hw_mul: expected ',' between operands\n";
        return nullptr;
      }
      Value rhs = parseExpr();
      if (!expect(TokenKind::RParen)) {
        llvm::errs() << "hw_mul: expected ')'\n";
        return nullptr;
      }

      mlir::ImplicitLocOpBuilder ilb(builder.getUnknownLoc(), builder);
      auto shapeAttr = mlir::ArrayAttr::get(builder.getContext(), {});
      auto tensorTy = mydialect::MyTensorType::get(builder.getContext(), shapeAttr,
                                                    builder.getI32Type());

      // TensorToScalarOp: MyTensorType → i32（显式传 result=i32）
      auto lhsI32 = ilb.create<mydialect::TensorToScalarOp>(
          builder.getI32Type(), lhs);
      auto rhsI32 = ilb.create<mydialect::TensorToScalarOp>(
          builder.getI32Type(), rhs);

      // HwMulOp: i32 × i32 → i32
      auto hwMulOp = mydialect::HwMulOp::create(ilb, ilb.getI32Type(),
                                                  lhsI32.getResult(),
                                                  rhsI32.getResult());

      // ScalarToTensorOp: i32 → MyTensorType（显式传 result=MyTensorType）
      auto wrapOp = ilb.create<mydialect::ScalarToTensorOp>(
          tensorTy, hwMulOp.getResult());
      return wrapOp.getResult();
    }

    auto it = symTable.find(nm);
    if (it == symTable.end()) {
      llvm::errs() << "undefined var " << nm << "\n";
      return nullptr;
    }
    res = it->second;
  } else if (curTok.kind == TokenKind::LParen) {
    consume();
    res = parseExpr();
    if (!expect(TokenKind::RParen)) return nullptr;
  } else {
    llvm::errs() << "bad primary\n";
    return nullptr;
  }
  return res;
}

} // namespace mydialect

