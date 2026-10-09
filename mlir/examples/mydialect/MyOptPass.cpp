#include "MyOptPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "MyDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mydialect {

// 生成 impl::MyOptPassBase（getArgument="my-opt"、getDescription、
// createMyOptPass 由 friend 自动提供）以及 registerMyOptPass()
#define GEN_PASS_DEF_MYOPTPASS
#include "MyOptPass.h.inc"

#define GEN_PASS_REGISTRATION_MYOPTPASS
#include "MyOptPass.h.inc"

namespace {

struct MyOptPass : public impl::MyOptPassBase<MyOptPass> {
  void runOnOperation() override;
};

} // namespace

void MyOptPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  // 遍历 module 内所有 func.func
  module.walk([&](mlir::func::FuncOp func) {
    // ========== 优化1：删除连续重复 mydialect.print ==========
    // 扫描块内操作，连续相同 operand 的 print，只保留第一条
    for (mlir::Block &block : func.getBlocks()) {
      llvm::SmallVector<mlir::Operation *, 8> toErase;
      mlir::Operation *lastPrint = nullptr;
      mlir::Value lastPrintVal;

      for (mlir::Operation &op : block) {
        if (auto printOp = mlir::dyn_cast<mydialect::PrintOp>(&op)) {
          mlir::Value val = printOp.getValue();
          if (lastPrint && lastPrintVal == val) {
            // 和上一条 print 完全一样，标记删除
            toErase.push_back(&op);
          } else {
            lastPrint = &op;
            lastPrintVal = val;
          }
          continue;
        }
        // 遇到非 print 操作，打断连续 print 序列
        lastPrint = nullptr;
        lastPrintVal = nullptr;
      }
      for (mlir::Operation *op : toErase) op->erase();
    }

    // ========== 优化2：简单 CSE：块内相同的 arith.addi ==========
    // 同一 Block 内操作数完全相同的 addi，复用首次结果
    for (mlir::Block &block : func.getBlocks()) {
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value> addMap;
      llvm::SmallVector<mlir::Operation *, 8> toErase;
      for (mlir::Operation &op : block) {
        auto addOp = mlir::dyn_cast<mlir::arith::AddIOp>(&op);
        if (!addOp)
          continue;
        auto key = std::make_pair(addOp.getLhs(), addOp.getRhs());
        auto it = addMap.find(key);
        if (it != addMap.end()) {
          // 已有相同 addi，所有使用替换成旧结果，删除当前 add
          addOp.getResult().replaceAllUsesWith(it->second);
          toErase.push_back(&op);
        } else {
          addMap[key] = addOp.getResult();
        }
      }
      for (mlir::Operation *op : toErase) op->erase();
    }
  });
}

} // namespace mydialect

// 文件级自动注册：链接本文件的二进制（mydriver / mlir-opt）
// 无需显式调用即可获得 --my-opt 命令行选项
static bool _myOptPassAutoRegistered = []() {
  mydialect::registerMyOptPass();
  return true;
}();
