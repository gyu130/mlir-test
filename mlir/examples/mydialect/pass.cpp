#include "pass.h"
#include "MyDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"

namespace {
// ============ PrintOp: mydialect.print → llvm.call ============
struct PrintOpLowering : public mlir::OpConversionPattern<mydialect::PrintOp> {
  using OpConversionPattern<mydialect::PrintOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::PrintOp op,
      typename mydialect::PrintOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sym = mlir::SymbolRefAttr::get(rewriter.getContext(), "my_runtime_print_i32");
    mlir::LLVM::CallOp::create(rewriter, loc, mlir::TypeRange{}, sym, adaptor.getValue());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ============ AddI32Op: mydialect.addi → arith.addi ============
struct AddI32OpLowering : public mlir::OpConversionPattern<mydialect::AddI32Op> {
  using OpConversionPattern<mydialect::AddI32Op>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::AddI32Op op,
      typename mydialect::AddI32Op::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newAdd = mlir::arith::AddIOp::create(
        rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newAdd.getResult());
    return mlir::success();
  }
};

// ============ MuliOp: mydialect.muli → arith.muli ============
struct MuliOpLowering : public mlir::OpConversionPattern<mydialect::MuliOp> {
  using OpConversionPattern<mydialect::MuliOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::MuliOp op,
      typename mydialect::MuliOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newMul = mlir::arith::MulIOp::create(
        rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newMul.getResult());
    return mlir::success();
  }
};

// ============ SubI32Op: mydialect.subi → arith.subi ============
struct SubI32OpLowering : public mlir::OpConversionPattern<mydialect::SubI32Op> {
  using OpConversionPattern<mydialect::SubI32Op>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::SubI32Op op,
      typename mydialect::SubI32Op::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newSub = mlir::arith::SubIOp::create(
        rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newSub.getResult());
    return mlir::success();
  }
};

// ============ DivSIOp: mydialect.divi_s → arith.divis ============
struct DivSIOpLowering : public mlir::OpConversionPattern<mydialect::DivSIOp> {
  using OpConversionPattern<mydialect::DivSIOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::DivSIOp op,
      typename mydialect::DivSIOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newDiv = mlir::arith::DivSIOp::create(
        rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newDiv.getResult());
    return mlir::success();
  }
};

// ============ Lower Pass ============
struct LowerMyDialectPrintPass
    : public mlir::PassWrapper<LowerMyDialectPrintPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMyDialectPrintPass)
  llvm::StringRef getArgument() const final {
    return "lower-mydialect-print";
  }
  llvm::StringRef getDescription() const final {
    return "Lower mydialect.print / addi / muli to Arith/LLVM dialect";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<
        mydialect::MyDialect,
        mlir::func::FuncDialect,
        mlir::LLVM::LLVMDialect,
        mlir::arith::ArithDialect,
        mlir::scf::SCFDialect>();
  }
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();
    // --- 1. 插入外部 runtime 函数声明 ---
    mlir::OpBuilder moduleBuilder(module.getBody(), module.getBody()->begin());
    mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
    auto llvmFuncTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {i32Ty}, false);
    mlir::LLVM::LLVMFuncOp::create(moduleBuilder,
        moduleBuilder.getUnknownLoc(),
        "my_runtime_print_i32",
        llvmFuncTy,
        mlir::LLVM::Linkage::External);
    // --- 2. ConversionTarget：标记 mydialect op 为非法 ---
    mlir::ConversionTarget target(*ctx);
    target.addIllegalOp<mydialect::PrintOp>();
    target.addIllegalOp<mydialect::AddI32Op>();
    target.addIllegalOp<mydialect::MuliOp>();
    target.addIllegalOp<mydialect::SubI32Op>();
    target.addIllegalOp<mydialect::DivSIOp>();
    target.addIllegalOp<mydialect::CmpiOp>();
    target.addIllegalOp<mydialect::YieldOp>();
    target.addIllegalOp<mydialect::ForOp>();
    // 允许的目标方言
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    // --- 3. 注册 patterns ---
    mlir::RewritePatternSet patterns(ctx);
    mydialect::populateMyDialectToLowerPatterns(patterns);
    // --- 4. applyPartialConversion ---
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// 折叠 mydialect.muli (MuliOp)
struct FoldMuliPattern : public mlir::RewritePattern {
  FoldMuliPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(mydialect::MuliOp::getOperationName(), 1, ctx) {}

  llvm::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto mulOp = llvm::cast<mydialect::MuliOp>(op);
    auto lhsConst = mulOp.getLhs().getDefiningOp<mlir::arith::ConstantOp>();
    auto rhsConst = mulOp.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
    if (!lhsConst || !rhsConst)
      return mlir::failure();

    auto lhsAttr = llvm::dyn_cast<mlir::IntegerAttr>(lhsConst.getValue());
    auto rhsAttr = llvm::dyn_cast<mlir::IntegerAttr>(rhsConst.getValue());
    if (!lhsAttr || !rhsAttr)
      return mlir::failure();

    int64_t lhs = lhsAttr.getInt();
    int64_t rhs = rhsAttr.getInt();
    int64_t res = lhs * rhs;

    // ========== 修复这一行 ==========
    mlir::ImplicitLocOpBuilder ib(op->getLoc(), rewriter);
    auto newConst = mlir::arith::ConstantOp::create(ib, rewriter.getI32IntegerAttr(res));
    rewriter.replaceOp(op, newConst);
    return mlir::success();
  }
};

// 折叠 mydialect.addi (AddI32Op)
struct FoldAddiPattern : public mlir::RewritePattern {
  FoldAddiPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(mydialect::AddI32Op::getOperationName(), 1, ctx) {}

  llvm::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto addOp = llvm::cast<mydialect::AddI32Op>(op);
    auto lhsConst = addOp.getLhs().getDefiningOp<mlir::arith::ConstantOp>();
    auto rhsConst = addOp.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
    if (!lhsConst || !rhsConst)
      return mlir::failure();

    auto lhsAttr = llvm::dyn_cast<mlir::IntegerAttr>(lhsConst.getValue());
    auto rhsAttr = llvm::dyn_cast<mlir::IntegerAttr>(rhsConst.getValue());
    if (!lhsAttr || !rhsAttr)
      return mlir::failure();

    int64_t lhs = lhsAttr.getInt();
    int64_t rhs = rhsAttr.getInt();
    int64_t res = lhs + rhs;

    // ========== 修复这一行 ==========
    mlir::ImplicitLocOpBuilder ib(op->getLoc(), rewriter);
    auto newConst = mlir::arith::ConstantOp::create(ib, rewriter.getI32IntegerAttr(res));
    rewriter.replaceOp(op, newConst);
    return mlir::success();
  }
};

// Print算子常量折叠Pattern
struct ConstantFoldMyDialectPattern : public mlir::RewritePattern {
  ConstantFoldMyDialectPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(mydialect::PrintOp::getOperationName(), 1, ctx) {}
  llvm::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    llvm::errs() << "Hit mydialect print op: " << op->getName() << "\n";
    auto constOp = op->getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
    if (!constOp)
      return mlir::failure();
    mlir::Attribute attr = constOp.getValue();
    auto typedAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr);
    if (!typedAttr)
      return mlir::failure();
    // ========== 修复这一行 ==========
    llvm::errs() << "[DEBUG] PrintOp sees constant value: " << typedAttr.getInt() << "\n";
    return mlir::failure(); //没有改动IR，返回success会触发无限迭代
  }
};

// 从常量循环边界提取 lo/hi/st 与迭代次数；st==0 或非常量返回 false
static bool getConstLoopTripCount(mydialect::ForOp forOp,
                                  int32_t &lo, int32_t &hi, int32_t &st,
                                  int &iterCnt) {
  auto lowerCst = forOp.getLower().getDefiningOp<mlir::arith::ConstantOp>();
  auto upperCst = forOp.getUpper().getDefiningOp<mlir::arith::ConstantOp>();
  auto stepCst  = forOp.getStep().getDefiningOp<mlir::arith::ConstantOp>();
  if (!lowerCst || !upperCst || !stepCst)
    return false;
  lo = (int32_t)llvm::dyn_cast<mlir::IntegerAttr>(lowerCst.getValue()).getInt();
  hi = (int32_t)llvm::dyn_cast<mlir::IntegerAttr>(upperCst.getValue()).getInt();
  st = (int32_t)llvm::dyn_cast<mlir::IntegerAttr>(stepCst.getValue()).getInt();
  if (st > 0)
    iterCnt = (hi <= lo) ? 0 : (hi - lo + st - 1) / st;
  else if (st < 0)
    iterCnt = (hi >= lo) ? 0 : (lo - hi - st - 1) / (-st);
  else
    return false; // step=0，UB，不优化
  return true;
}

// mydialect.for 常量循环求值pattern：边界/iter_args 均为常量、
// 循环体只含纯算术 op 时，直接解释执行算出结果常量
struct ConstantEvaluateForPattern : public mlir::OpRewritePattern<mydialect::ForOp> {
  // benefit=2：优先于循环展开（benefit=1）
  ConstantEvaluateForPattern(mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<mydialect::ForOp>(ctx, /*benefit=*/2) {}

  mlir::LogicalResult matchAndRewrite(mydialect::ForOp forOp,
                                      mlir::PatternRewriter &rewriter) const override {
    int32_t lo, hi, st;
    int iterCnt;
    const int MAX_EVAL_ITER = 16;
    if (!getConstLoopTripCount(forOp, lo, hi, st, iterCnt) ||
        iterCnt > MAX_EVAL_ITER)
      return mlir::failure();

    // iter_args 初始值必须是常量
    llvm::SmallVector<int32_t> accVals;
    for (mlir::Value v : forOp.getIterArgs()) {
      mlir::APInt cst;
      if (!mlir::matchPattern(v, mlir::m_ConstantInt(&cst)))
        return mlir::failure();
      accVals.push_back((int32_t)cst.getSExtValue());
    }

    // 循环体必须只含可求值的纯算术 op / yield
    mlir::Block &body = forOp.getBody().front();
    for (mlir::Operation &op : body) {
      if (!llvm::isa<mydialect::AddI32Op, mydialect::MuliOp,
                     mydialect::SubI32Op, mydialect::DivSIOp,
                     mydialect::YieldOp>(op))
        return mlir::failure();
    }

    // 解释执行：env 把循环体内 Value 映射到本轮迭代的常量值
    llvm::DenseMap<mlir::Value, int32_t> env;
    auto evalOperand = [&](mlir::Value v, int32_t &out) -> bool {
      auto it = env.find(v);
      if (it != env.end()) {
        out = it->second;
        return true;
      }
      // 循环体外定义的常量（如 %one）也可参与求值
      mlir::APInt cst;
      if (mlir::matchPattern(v, mlir::m_ConstantInt(&cst))) {
        out = (int32_t)cst.getSExtValue();
        return true;
      }
      return false;
    };

    for (int it = 0; it < iterCnt; ++it) {
      // 绑定 iv 与 iter_args 块参数
      env[body.getArgument(0)] = lo + it * st;
      for (unsigned i = 0; i < accVals.size(); ++i)
        env[body.getArgument(i + 1)] = accVals[i];

      for (mlir::Operation &op : body) {
        if (auto yieldOp = llvm::dyn_cast<mydialect::YieldOp>(op)) {
          for (auto [idx, val] : llvm::enumerate(yieldOp.getValues())) {
            int32_t res;
            if (!evalOperand(val, res))
              return mlir::failure();
            accVals[idx] = res;
          }
          break;
        }
        int32_t lhs, rhs;
        if (!evalOperand(op.getOperand(0), lhs) ||
            !evalOperand(op.getOperand(1), rhs))
          return mlir::failure();
        int32_t res;
        if (llvm::isa<mydialect::AddI32Op>(op)) res = lhs + rhs;
        else if (llvm::isa<mydialect::MuliOp>(op)) res = lhs * rhs;
        else if (llvm::isa<mydialect::SubI32Op>(op)) res = lhs - rhs;
        else { // divi_s
          if (rhs == 0)
            return mlir::failure(); // 除零不折叠
          res = lhs / rhs;
        }
        env[op.getResult(0)] = res;
      }
    }

    // 用算出的常量替换整个 mydialect.for
    llvm::SmallVector<mlir::Value> results;
    for (int32_t val : accVals) {
      auto cst = mlir::arith::ConstantOp::create(
          rewriter, forOp.getLoc(), rewriter.getI32IntegerAttr(val));
      results.push_back(cst.getResult());
    }
    rewriter.replaceOp(forOp, results);
    return mlir::success();
  }
};

// mydialect.for 常量循环展开pattern：边界常量、迭代次数较小时，
// 把循环体复制 iterCnt 份，iv 替换为常量，iter_args 逐轮串联
struct ConstantUnrollForPattern : public mlir::OpRewritePattern<mydialect::ForOp> {
  using OpRewritePattern<mydialect::ForOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mydialect::ForOp forOp,
                                      mlir::PatternRewriter &rewriter) const override {
    int32_t lo, hi, st;
    int iterCnt;
    const int MAX_UNROLL = 16;
    if (!getConstLoopTripCount(forOp, lo, hi, st, iterCnt) ||
        iterCnt > MAX_UNROLL)
      return mlir::failure();

    // 0 次迭代：结果直接是 iter_args（ForOp::fold 也覆盖此情形）
    if (iterCnt == 0) {
      rewriter.replaceOp(forOp, forOp.getIterArgs());
      return mlir::success();
    }

    mlir::Block &body = forOp.getBody().front();
    mlir::IRMapping mapping;
    llvm::SmallVector<mlir::Value> curAcc(forOp.getIterArgs().begin(),
                                          forOp.getIterArgs().end());
    rewriter.setInsertionPoint(forOp);
    for (int it = 0; it < iterCnt; ++it) {
      // iv -> 本轮常量
      auto ivCst = mlir::arith::ConstantOp::create(
          rewriter, forOp.getLoc(), rewriter.getI32IntegerAttr(lo + it * st));
      mapping.map(body.getArgument(0), ivCst.getResult());
      // iter_args 块参数 -> 上一轮结果
      for (unsigned i = 0; i < curAcc.size(); ++i)
        mapping.map(body.getArgument(i + 1), curAcc[i]);
      // 克隆循环体（除 yield）；未映射的操作数（循环体外值）保持原样
      for (mlir::Operation &op : body.without_terminator())
        rewriter.clone(op, mapping);
      // yield 的操作数成为下一轮 acc
      auto yieldOp = llvm::cast<mydialect::YieldOp>(body.getTerminator());
      curAcc.clear();
      for (mlir::Value v : yieldOp.getValues())
        curAcc.push_back(mapping.lookupOrDefault(v));
    }
    rewriter.replaceOp(forOp, curAcc);
    return mlir::success();
  }
};

struct LoopInvariantHoistPattern : public mlir::OpRewritePattern<mydialect::ForOp> {
  using OpRewritePattern<mydialect::ForOp>::OpRewritePattern;
  bool isDefinedOutside(mlir::Value val, mlir::Region &loopBody) const {
      // 循环体自身的块参数（iv / iter_args）定义在循环内部，不能外提
      if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(val))
        return !loopBody.isAncestor(blockArg.getOwner()->getParent());
      auto defOp = val.getDefiningOp();
      if(!defOp) return true; // 其它块参数，保守视为外部
      return !loopBody.isAncestor(defOp->getParentRegion());
  }

  mlir::LogicalResult matchAndRewrite(mydialect::ForOp forOp,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Region &bodyRegion = forOp.getBody();
    mlir::Block &bodyBlock = bodyRegion.front();
    llvm::SmallVector<mlir::Operation*,8> hoistCandidates;
    for(auto &op : bodyBlock){
        if(llvm::isa<mydialect::YieldOp>(op)) break;
        bool allOutside = true;
        for(auto operand : op.getOperands()){
            if(!isDefinedOutside(operand, bodyRegion)){
                allOutside = false;
                break;
            }
        }
        if(allOutside){
            hoistCandidates.push_back(&op);
        }
    }
    if(hoistCandidates.empty()) return mlir::failure();
    rewriter.setInsertionPoint(forOp);
    for(auto op : hoistCandidates){
        rewriter.moveOpBefore(op, forOp);
    }
    return mlir::success();
  }
};

struct ConstantFoldMyDialectPass
    : public mlir::PassWrapper<ConstantFoldMyDialectPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldMyDialectPass)
  llvm::StringRef getArgument() const final {
    return "constant-fold-mydialect";
  }
  llvm::StringRef getDescription() const final {
    return "Constant fold mydialect ops";
  }
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FoldMuliPattern,
	    FoldAddiPattern,
	    ConstantFoldMyDialectPattern,
	    ConstantEvaluateForPattern,
	    ConstantUnrollForPattern,
            LoopInvariantHoistPattern>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};


// ============ CmpiOp: mydialect.cmpi → arith.cmpi ============
struct CmpiOpLowering : public mlir::OpConversionPattern<mydialect::CmpiOp> {
  using OpConversionPattern<mydialect::CmpiOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mydialect::CmpiOp op,
      typename mydialect::CmpiOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto predStr = op.getPredicate();

    mlir::arith::CmpIPredicate pred;
    if (predStr == "slt") pred = mlir::arith::CmpIPredicate::slt;
    else if (predStr == "sgt") pred = mlir::arith::CmpIPredicate::sgt;
    else if (predStr == "sle") pred = mlir::arith::CmpIPredicate::sle;
    else if (predStr == "sge") pred = mlir::arith::CmpIPredicate::sge;
    else if (predStr == "eq")  pred = mlir::arith::CmpIPredicate::eq;
    else if (predStr == "ne")  pred = mlir::arith::CmpIPredicate::ne;
    else return rewriter.notifyMatchFailure(op, "unsupported predicate: " + predStr);

    auto newCmp = mlir::arith::CmpIOp::create(
        rewriter, loc, pred, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newCmp.getResult());
    return mlir::success();
  }
};

struct ForOpLowering : public mlir::OpConversionPattern<mydialect::ForOp> {
  using mlir::OpConversionPattern<mydialect::ForOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(mydialect::ForOp op,
                                      mydialect::ForOp::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto castToIndex = [&](mlir::Value v) {
      return mlir::arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), v);
    };
    mlir::Value lower = castToIndex(adaptor.getLower());
    mlir::Value upper = castToIndex(adaptor.getUpper());
    mlir::Value step  = castToIndex(adaptor.getStep());

    // -------- 循环携带初始值（iter_args）---------
    llvm::SmallVector<mlir::Value> scfInitVals;
    for(auto v : adaptor.getIterArgs()){
        scfInitVals.push_back(v);
    }

    // 创建 scf.for，带 iter_args
    auto scfFor = mlir::scf::ForOp::create(rewriter, loc,
        lower, upper, step, scfInitVals);

    mlir::Block *dstBody = scfFor.getBody();
    // 注意：带 iter_args 时 scf::ForOp 的 builder 不会自动插入 scf.yield，
    // 仅无 iter_args 时才存在默认 terminator 需要删除
    if (!dstBody->empty())
      rewriter.eraseOp(dstBody->getTerminator());

    // scf.for body 参数顺序：%iv(index), %arg0, %arg1 ...
    rewriter.setInsertionPointToStart(dstBody);
    // 把 index 类型iv转回i32，给mydialect body使用
    mlir::Value ivI32 = mlir::arith::IndexCastOp::create(rewriter, loc,
        rewriter.getI32Type(), scfFor.getInductionVar());

    // scfFor body块参数 [0]=iv(index), [1...]=iter_args(i32)
    // mydialect for body块参数 [0]=iv(i32), [1...]=iter_args(i32)
    llvm::SmallVector<mlir::Value> bodyReplacements;
    bodyReplacements.push_back(ivI32);
    for(size_t i = 1; i < dstBody->getNumArguments(); ++i){
        bodyReplacements.push_back(dstBody->getArgument(i));
    }

    mlir::Block *srcBody = &op.getBody().front();
    // inline原mydialect for body，替换块参数
    rewriter.inlineBlockBefore(srcBody, dstBody, dstBody->end(), bodyReplacements);

    // 把 mydialect.yield %v0, %v1 → scf.yield %v0, %v1
    auto yieldOp = llvm::cast<mydialect::YieldOp>(dstBody->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    mlir::scf::YieldOp::create(rewriter, loc, yieldOp.getValues());
    rewriter.eraseOp(yieldOp);

    // 替换原mydialect.for的结果为scf.for结果
    rewriter.replaceOp(op, scfFor.getResults());
    return mlir::success();
  }
};


} // anonymous namespace

namespace mydialect {
void populateMyDialectToLowerPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<PrintOpLowering, 
	AddI32OpLowering, 
	MuliOpLowering, 
	SubI32OpLowering, 
	DivSIOpLowering, 
	CmpiOpLowering,
	ForOpLowering>(patterns.getContext());
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerMyDialectPrintPass() {
  return std::make_unique<LowerMyDialectPrintPass>();
}

void registerConstantFoldMyDialectPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ConstantFoldMyDialectPass>();
  });
}

std::unique_ptr<mlir::Pass> createConstantFoldMyDialectPass() {
  return std::make_unique<ConstantFoldMyDialectPass>();
}
} // namespace mydialect

static mlir::PassRegistration<LowerMyDialectPrintPass> reg;
static mlir::PassRegistration<ConstantFoldMyDialectPass> regFold;

#include "mlir/IR/DialectRegistry.h"
extern "C" void mlirRegisterAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mydialect::MyDialect>();
}

