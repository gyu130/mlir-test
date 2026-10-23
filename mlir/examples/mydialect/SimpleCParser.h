#ifndef SIMPLE_C_PARSER_H
#define SIMPLE_C_PARSER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <string>
#include <vector>

namespace mydialect {

// Token 类型
enum class TokenKind {
  Eof,
  IntKw, MainKw, PrintKw, ReturnKw,
  Ident, Number,
  Plus, Minus, Mul, Div,
  LParen, RParen, LBrace, RBrace, Semi, Eq
};

struct Token {
  TokenKind kind;
  std::string str;
  size_t line;
};

struct SimpleCParser {
  SimpleCParser(mlir::OpBuilder &builder, mlir::ModuleOp mod);

  // 入口：C源码字符串 -> 填充 module
  mlir::LogicalResult parse(const std::string &src);

private:
  // Lexer
  std::string src;
  size_t pos = 0;
  Token curTok;
  mlir::OpBuilder &builder;
  mlir::ModuleOp module;
  // 符号表：变量名 -> mlir::Value
  llvm::StringMap<mlir::Value> symTable;

  Token nextToken();
  void consume();
  bool expect(TokenKind k);

  // Parser 递归函数
  mlir::LogicalResult parseFunc();
  mlir::LogicalResult parseStmt(mlir::Block *block);
  mlir::LogicalResult parseDecl(mlir::Block *block);
  mlir::LogicalResult parseCall(mlir::Block *block);
  mlir::LogicalResult parseReturn(mlir::Block *block);
  mlir::Value parseExpr();
  mlir::Value parseMulExpr();
  mlir::Value parsePrimary();

  // 工具：生成 mydialect 常量 / binary op
  mlir::Value makeConstI32(int64_t val);
  mlir::Value makeBinOp(mlir::Value lhs, mlir::Value rhs, TokenKind op);
};

} // namespace mydialect
#endif

