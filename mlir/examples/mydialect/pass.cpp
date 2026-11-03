#include "pass.h"
#include "MyDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/SymbolTable.h"

namespace {

// ============ 公共 helper ============
// 把 adaptor 操作数（可能仍是 MyTensorType，上游 op 未转换）补 cast 到 memref。
static mlir::Value ensureMemref(mlir::Value v, const mlir::TypeConverter &tc,
                                mlir::ConversionPatternRewriter &rw,
                                mlir::Location loc) {
  if (mlir::isa<mlir::MemRefType>(v.getType()))
    return v;
  return mlir::UnrealizedConversionCastOp::create(
             rw, loc, tc.convertType(v.getType()), v)
      .getResult(0);
}

// 按静态 shape 生成嵌套 scf.for；rank==0 时直接调用 body({})。
// 函数内部持有 InsertionGuard，返回后插入点停留在循环 nest 之外。
static void emitLoopNest(
    mlir::ConversionPatternRewriter &rw, mlir::Location loc,
    llvm::ArrayRef<int64_t> shape,
    llvm::function_ref<void(llvm::ArrayRef<mlir::Value>)> body) {
  mlir::OpBuilder::InsertionGuard guard(rw);
  llvm::SmallVector<mlir::Value, 4> ivs;
  for (int64_t d : shape) {
    auto lb = mlir::arith::ConstantIndexOp::create(rw, loc, 0);
    auto ub = mlir::arith::ConstantIndexOp::create(rw, loc, d);
    auto st = mlir::arith::ConstantIndexOp::create(rw, loc, 1);
    auto loop = mlir::scf::ForOp::create(rw, loc, lb, ub, st);
    ivs.push_back(loop.getInductionVar());
    rw.setInsertionPointToStart(loop.getBody());
  }
  body(ivs);
}

// ============ WrapOp: mydialect.wrap i32→wrapped_i32，降级时透传 ============
// wrapped_i32 经 TypeConverter 转 i32，wrap 语义上就是 identity，直接替换
struct WrapOpLowering : public mlir::OpConversionPattern<mydialect::WrapOp> {
  using OpConversionPattern<mydialect::WrapOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::WrapOp op,
      typename mydialect::WrapOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // 结果类型经 converter 为 i32，adaptor.getInput() 也是 i32，直接透传
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

// ============ UnwrapOp: mydialect.unwrap wrapped_i32→i32，降级时透传 ============
struct UnwrapOpLowering : public mlir::OpConversionPattern<mydialect::UnwrapOp> {
  using OpConversionPattern<mydialect::UnwrapOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::UnwrapOp op,
      typename mydialect::UnwrapOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value input = adaptor.getInput();
    // 输入可能仍是 wrapped_i32，显式擦除为 i32 后透传
    if (mlir::isa<mydialect::WrappedI32Type>(input.getType())) {
      input = rewriter.create<mlir::UnrealizedConversionCastOp>(
          op.getLoc(), rewriter.getI32Type(), input).getResult(0);
    }
    rewriter.replaceOp(op, input);
    return mlir::success();
  }
};

// ============ PrintOp: mydialect.print → llvm.call ============
struct PrintOpLowering : public mlir::OpConversionPattern<mydialect::PrintOp> {
  using OpConversionPattern<mydialect::PrintOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::PrintOp op,
      typename mydialect::PrintOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sym = mlir::SymbolRefAttr::get(rewriter.getContext(), "my_runtime_print_i32");
    mlir::Value val = adaptor.getValue();

    // ---- tensor 类型（converter 已把 MyTensorType 转为 memref）----
    // 逐元素打印：按静态 shape 生成嵌套 scf.for，body 内 memref.load
    // 后调用 my_runtime_print_i32
    if (auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(val.getType())) {
      if (llvm::any_of(memrefTy.getShape(), [](int64_t d) {
            return mlir::ShapedType::isDynamic(d);
          }))
        return rewriter.notifyMatchFailure(op,
                                           "print: dynamic shape not supported");

      mlir::OpBuilder::InsertionGuard guard(rewriter);
      llvm::SmallVector<mlir::Value, 4> ivs;
      for (int64_t d : memrefTy.getShape()) {
        auto lb = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto ub = mlir::arith::ConstantIndexOp::create(rewriter, loc, d);
        auto step = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
        auto loop = mlir::scf::ForOp::create(rewriter, loc, lb, ub, step);
        ivs.push_back(loop.getInductionVar());
        // 下一层循环/最终 load 都建在本层 body 内
        rewriter.setInsertionPointToStart(loop.getBody());
      }
      mlir::Value elem = rewriter.create<mlir::memref::LoadOp>(loc, val, ivs);
      mlir::LLVM::CallOp::create(rewriter, loc, mlir::TypeRange{}, sym, elem);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // ---- 标量：wrapped_i32 显式擦除为 i32 ----
    // 上游 wrap 可能尚未转换，此时 val 仍是 wrapped_i32；
    // 插入 unrealized_conversion_cast，待 wrap 转换后自动折叠。
    if (mlir::isa<mydialect::WrappedI32Type>(val.getType())) {
      val = rewriter.create<mlir::UnrealizedConversionCastOp>(
          loc, rewriter.getI32Type(), val).getResult(0);
    }
    mlir::LLVM::CallOp::create(rewriter, loc, mlir::TypeRange{}, sym, val);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ============ PrintStrOp: mydialect.print_str → llvm.call @my_runtime_print_string ============
struct PrintStrOpLowering : public mlir::OpConversionPattern<mydialect::PrintStrOp> {
  using OpConversionPattern<mydialect::PrintStrOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::PrintStrOp op,
      typename mydialect::PrintStrOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sym = mlir::SymbolRefAttr::get(rewriter.getContext(), "my_runtime_print_string");
    mlir::LLVM::CallOp::create(rewriter, loc, mlir::TypeRange{}, sym, adaptor.getStr());
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

// ============ 6.1 张量级常量折叠 ============
// add_tensor(constant_tensor, constant_tensor) 与
// mul_tensor(...) 直接算出新常量：逐元素对 DenseElementsAttr 求值，
// 结果仍为 constant_tensor（形状/元素类型与输入一致）。
//
// 逐元素求值辅助函数（非模板，避免模板内依赖名的 getValues<> 解析歧义）：
// 不支持的类型返回 nullopt
static std::optional<mlir::ElementsAttr>
evalTensorBinaryConst(mlir::ElementsAttr lhs, mlir::ElementsAttr rhs,
                      bool isAdd) {
  auto lhsAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(lhs);
  auto rhsAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(rhs);
  if (!lhsAttr || !rhsAttr)
    return std::nullopt;

  auto shapedTy = llvm::cast<mlir::ShapedType>(lhsAttr.getType());

  if (llvm::isa<mlir::DenseIntElementsAttr>(lhsAttr)) {
    if (!llvm::isa<mlir::DenseIntElementsAttr>(rhsAttr))
      return std::nullopt;
    llvm::SmallVector<llvm::APInt> vals;
    for (auto [l, r] : llvm::zip(lhsAttr.getValues<llvm::APInt>(),
                                 rhsAttr.getValues<llvm::APInt>()))
      vals.push_back(isAdd ? l + r : l * r);
    return mlir::ElementsAttr(mlir::DenseIntElementsAttr::get(shapedTy, vals));
  }

  if (llvm::isa<mlir::DenseFPElementsAttr>(lhsAttr)) {
    if (!llvm::isa<mlir::DenseFPElementsAttr>(rhsAttr))
      return std::nullopt;
    llvm::SmallVector<llvm::APFloat> vals;
    for (auto [l, r] : llvm::zip(lhsAttr.getValues<llvm::APFloat>(),
                                 rhsAttr.getValues<llvm::APFloat>())) {
      llvm::APFloat v = l;
      if (isAdd)
        v.add(r, llvm::APFloat::rmNearestTiesToEven);
      else
        v.multiply(r, llvm::APFloat::rmNearestTiesToEven);
      vals.push_back(v);
    }
    return mlir::ElementsAttr(mlir::DenseFPElementsAttr::get(shapedTy, vals));
  }

  return std::nullopt;
}

template <typename OpTy, bool IsAdd>
struct FoldTensorBinaryPattern : public mlir::RewritePattern {
  FoldTensorBinaryPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(OpTy::getOperationName(), 1, ctx) {}

  llvm::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto binOp = llvm::cast<OpTy>(op);
    auto lhsC = binOp.getLhs().template getDefiningOp<mydialect::ConstantTensorOp>();
    auto rhsC = binOp.getRhs().template getDefiningOp<mydialect::ConstantTensorOp>();
    if (!lhsC || !rhsC)
      return mlir::failure();

    auto newAttr =
        evalTensorBinaryConst(lhsC.getValue(), rhsC.getValue(), IsAdd);
    if (!newAttr)
      return mlir::failure();

    // 结果 MyTensorType 由原 op 决定（shape/元素类型与常量 attr 一致）
    auto resTy = llvm::cast<mydialect::MyTensorType>(binOp.getResult().getType());
    rewriter.replaceOpWithNewOp<mydialect::ConstantTensorOp>(op, resTy,
                                                             *newAttr);
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
	    FoldTensorBinaryPattern<mydialect::AddTensorOp, /*IsAdd=*/true>,
	    FoldTensorBinaryPattern<mydialect::MulTensorOp, /*IsAdd=*/false>,
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

struct MyAllocToMemRef : public mlir::OpConversionPattern<mydialect::AllocOp> {
  using mlir::OpConversionPattern<mydialect::AllocOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::AllocOp op, mydialect::AllocOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto memTy = mlir::cast<mlir::MemRefType>(op.getRes().getType());
    // memref.alloca 的 size 操作数要求 index 类型，先做 i32→index 转换
    auto indexSize = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), adaptor.getSize());
    auto newAlloc = rewriter.create<mlir::memref::AllocaOp>(
        loc, memTy, mlir::ValueRange{indexSize});
    rewriter.replaceOp(op, newAlloc.getResult());
    return mlir::success();
  }
};

struct MyLoadToMemRef : public mlir::OpConversionPattern<mydialect::LoadOp> {
  using mlir::OpConversionPattern<mydialect::LoadOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::LoadOp op, mydialect::LoadOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // memref.load 的 index 操作数要求 index 类型
    auto indexVar = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), adaptor.getIndex());
    auto load = rewriter.create<mlir::memref::LoadOp>(
        loc, adaptor.getMemref(), mlir::ValueRange{indexVar});
    rewriter.replaceOp(op, load.getResult());
    return mlir::success();
  }
};

struct MyStoreToMemRef : public mlir::OpConversionPattern<mydialect::StoreOp> {
  using mlir::OpConversionPattern<mydialect::StoreOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::StoreOp op, mydialect::StoreOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // memref.store 的 index 操作数要求 index 类型
    auto indexVar = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), adaptor.getIndex());
    rewriter.create<mlir::memref::StoreOp>(
        loc, adaptor.getValue(), adaptor.getMemref(),
        mlir::ValueRange{indexVar});
    rewriter.eraseOp(op);
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


// ============ ConstantTensorOp: mydialect.constant_tensor → memref.global + get_global ============
// 张量常量的值是静态的 ElementsAttr，降级为模块级常量全局 memref，
// 使用点替换为 memref.get_global。global 需要唯一符号名，用 SymbolTable 查重。
struct ConstantTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::ConstantTensorOp> {
  using mlir::OpConversionPattern<mydialect::ConstantTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::ConstantTensorOp op, mydialect::ConstantTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resTy = mlir::cast<mlir::MemRefType>(
        getTypeConverter()->convertType(op.getRes().getType()));

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "constant_tensor not inside module");

    // 生成模块内唯一符号名
    mlir::SymbolTable symTable(module);
    std::string name;
    unsigned id = 0;
    do {
      name = ("__mydialect_cst_" + llvm::Twine(id++)).str();
    } while (symTable.lookup(name));

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<mlir::memref::GlobalOp>(
          loc, name, rewriter.getStringAttr("private"), resTy,
          op.getValue(), /*constant=*/true, /*alignment=*/mlir::IntegerAttr());
    }
    auto getGlobal =
        rewriter.create<mlir::memref::GetGlobalOp>(loc, resTy, name);
    rewriter.replaceOp(op, getGlobal.getResult());
    return mlir::success();
  }
};

// ============ AddTensorOp: mydialect.add_tensor → memref 逐元素加法 ============
// adaptor 操作数已是 memref；为结果分配新 memref，按静态 shape 生成嵌套
// scf.for（0 维张量无循环直接 load/add/store），结果写入新分配的 memref。
struct AddTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::AddTensorOp> {
  using mlir::OpConversionPattern<mydialect::AddTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::AddTensorOp op, mydialect::AddTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value lhs = ensureMemref(adaptor.getLhs(), tc, rewriter, loc);
    mlir::Value rhs = ensureMemref(adaptor.getRhs(), tc, rewriter, loc);
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    mlir::Type elemTy = resTy.getElementType();
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      mlir::Value lv = rewriter.create<mlir::memref::LoadOp>(loc, lhs, idxs);
      mlir::Value rv = rewriter.create<mlir::memref::LoadOp>(loc, rhs, idxs);
      mlir::Value sum = mlir::isa<mlir::IntegerType>(elemTy)
        ? (mlir::Value)mlir::arith::AddIOp::create(rewriter, loc, lv, rv)
        : (mlir::Value)mlir::arith::AddFOp::create(rewriter, loc, lv, rv);
      rewriter.create<mlir::memref::StoreOp>(loc, sum, res, idxs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ MulTensorOp: mydialect.mul_tensor → memref 逐元素乘法 ============
// 与 AddTensorOpLowering 同构（C 前端 7*6 与 ONNX Mul 都会产生 mul_tensor）：
// adaptor 操作数已是 memref；0 维张量无循环直接 load/mul/store，
// 多维张量按静态 shape 生成嵌套 scf.for
struct MulTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::MulTensorOp> {
  using mlir::OpConversionPattern<mydialect::MulTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::MulTensorOp op, mydialect::MulTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value lhs = ensureMemref(adaptor.getLhs(), tc, rewriter, loc);
    mlir::Value rhs = ensureMemref(adaptor.getRhs(), tc, rewriter, loc);
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    mlir::Type elemTy = resTy.getElementType();
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      mlir::Value lv = rewriter.create<mlir::memref::LoadOp>(loc, lhs, idxs);
      mlir::Value rv = rewriter.create<mlir::memref::LoadOp>(loc, rhs, idxs);
      mlir::Value prod = mlir::isa<mlir::IntegerType>(elemTy)
        ? (mlir::Value)mlir::arith::MulIOp::create(rewriter, loc, lv, rv)
        : (mlir::Value)mlir::arith::MulFOp::create(rewriter, loc, lv, rv);
      rewriter.create<mlir::memref::StoreOp>(loc, prod, res, idxs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ SubTensorOp: mydialect.sub_tensor → 同构 Add/Mul，arith.subi/subf ============
struct SubTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::SubTensorOp> {
  using mlir::OpConversionPattern<mydialect::SubTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::SubTensorOp op, mydialect::SubTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value lhs = ensureMemref(adaptor.getLhs(), tc, rewriter, loc);
    mlir::Value rhs = ensureMemref(adaptor.getRhs(), tc, rewriter, loc);
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    mlir::Type elemTy = resTy.getElementType();
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      mlir::Value lv = rewriter.create<mlir::memref::LoadOp>(loc, lhs, idxs);
      mlir::Value rv = rewriter.create<mlir::memref::LoadOp>(loc, rhs, idxs);
      mlir::Value diff = mlir::isa<mlir::IntegerType>(elemTy)
        ? (mlir::Value)mlir::arith::SubIOp::create(rewriter, loc, lv, rv)
        : (mlir::Value)mlir::arith::SubFOp::create(rewriter, loc, lv, rv);
      rewriter.create<mlir::memref::StoreOp>(loc, diff, res, idxs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ DivTensorOp: mydialect.div_tensor → 同构 Add/Mul，arith.divis/divf ============
// 注：整数除法走 DivSIOp（有符号）；无静态除零 verifier（逐元素除零是运行时语义）。
struct DivTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::DivTensorOp> {
  using mlir::OpConversionPattern<mydialect::DivTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::DivTensorOp op, mydialect::DivTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value lhs = ensureMemref(adaptor.getLhs(), tc, rewriter, loc);
    mlir::Value rhs = ensureMemref(adaptor.getRhs(), tc, rewriter, loc);
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    mlir::Type elemTy = resTy.getElementType();
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      mlir::Value lv = rewriter.create<mlir::memref::LoadOp>(loc, lhs, idxs);
      mlir::Value rv = rewriter.create<mlir::memref::LoadOp>(loc, rhs, idxs);
      mlir::Value q = mlir::isa<mlir::IntegerType>(elemTy)
        ? (mlir::Value)mlir::arith::DivSIOp::create(rewriter, loc, lv, rv)
        : (mlir::Value)mlir::arith::DivFOp::create(rewriter, loc, lv, rv);
      rewriter.create<mlir::memref::StoreOp>(loc, q, res, idxs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ FusedElementwiseOp: → 单个循环 nest + 内联 region 标量计算 ============
// 与 AddTensorOpLowering 同构地生成循环骨架；循环体内不再固定 add/mul，
// 而是把 fused region 的标量表达式按块参数映射内联（clone + IRMapping），
// fused_yield 的值 store 到结果 memref。多次融合只产生一个循环 nest，
// 这就是融合的收益：中间结果不再落地 memref。
struct FusedElementwiseOpLowering
    : public mlir::OpConversionPattern<mydialect::FusedElementwiseOp> {
  using mlir::OpConversionPattern<
      mydialect::FusedElementwiseOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::FusedElementwiseOp op,
      mydialect::FusedElementwiseOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    llvm::SmallVector<mlir::Value> inputs;
    for (mlir::Value v : adaptor.getInputs())
      inputs.push_back(ensureMemref(v, tc, rewriter, loc));
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      mlir::IRMapping mapping;
      mlir::Block &blk = op.getBody().front();
      for (auto [arg, in] : llvm::zip(blk.getArguments(), inputs)) {
        mlir::Value elem =
            rewriter.create<mlir::memref::LoadOp>(loc, in, idxs);
        mapping.map(arg, elem);
      }
      mlir::Value result;
      for (mlir::Operation &inner : blk) {
        if (auto y = mlir::dyn_cast<mydialect::FusedYieldOp>(inner)) {
          result = mapping.lookup(y.getValue());
          continue;
        }
        rewriter.clone(inner, mapping);
      }
      rewriter.create<mlir::memref::StoreOp>(loc, result, res, idxs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ CmpITensorOp: mydialect.cmpi_tensor → memref 逐元素比较 ============
// predicate 采用 arith::CmpIPredicate 枚举值（eq=0..uge=9），直接 static_cast。
// 结果 memref 元素类型为 i1，逐元素 load-lhs/rhs → arith.cmpi → store。
struct CmpITensorOpLowering
    : public mlir::OpConversionPattern<mydialect::CmpITensorOp> {
  using mlir::OpConversionPattern<mydialect::CmpITensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::CmpITensorOp op, mydialect::CmpITensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));

    // predicate 采用 arith::CmpIPredicate 枚举值（0..9）
    int64_t predVal = op.getPredicate();
    if (predVal < 0 || predVal > 9)
      return rewriter.notifyMatchFailure(op, "invalid predicate range");
    auto pred = static_cast<mlir::arith::CmpIPredicate>(predVal);

    mlir::Value lhs = ensureMemref(adaptor.getLhs(), tc, rewriter, loc);
    mlir::Value rhs = ensureMemref(adaptor.getRhs(), tc, rewriter, loc);
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      mlir::Value lv = rewriter.create<mlir::memref::LoadOp>(loc, lhs, idxs);
      mlir::Value rv = rewriter.create<mlir::memref::LoadOp>(loc, rhs, idxs);
      mlir::Value cmp = mlir::arith::CmpIOp::create(rewriter, loc, pred, lv, rv);
      rewriter.create<mlir::memref::StoreOp>(loc, cmp, res, idxs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ IdentityTensorOp: mydialect.identity_tensor → 零拷贝别名 ============
// identity 语义即透传；verifier 已保证输入/结果类型相同。
// 直接把 adaptor.getIn()（已是 memref 或经 ensureMemref cast）替换 op。
struct IdentityTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::IdentityTensorOp> {
  using mlir::OpConversionPattern<
      mydialect::IdentityTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::IdentityTensorOp op,
      mydialect::IdentityTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    // 注意：此处选择别名而非 memref.copy。
    // 前提：当前 mydialect 除显式 memref.store 外不引入原地写算子，
    // 多个别名指向同一 buffer 在当前 pass 下不会产生写冲突。
    mlir::Value src = ensureMemref(adaptor.getIn(), tc, rewriter, loc);
    rewriter.replaceOp(op, src);
    return mlir::success();
  }
};

// ============ ReshapeTensorOp: mydialect.reshape_tensor → memref.reinterpret_cast ============
// reshape 语义是元素总数不变的布局重解释，结果 memref 与输入共享底层 buffer，
// 只是 size/stride/offset 不同。降级时丢掉运行时 shape 操作数
// （它的作用已由 IR 显式结果类型表达）。
// 约束：src 与 res 的元素总数必须相等（由 ReshapeTensorOp::verify 保证）。
// 注：src 的 layout 必须是 identity（本项目所有 memref 来自 memref.alloc），
// reinterpret_cast 的 offset 语义才能正确。若将来引入带 offset/stride 的
// 非 identity 视图，需要改为 memref.copy 或改用 collapse_shape/expand_shape。
struct ReshapeTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::ReshapeTensorOp> {
  using mlir::OpConversionPattern<
      mydialect::ReshapeTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::ReshapeTensorOp op,
      mydialect::ReshapeTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    if (!resTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shape not supported");
    mlir::Value src = ensureMemref(adaptor.getData(), tc, rewriter, loc);

    // 行主序 strides
    llvm::SmallVector<int64_t> sizes(resTy.getShape().begin(),
                                     resTy.getShape().end());
    llvm::SmallVector<int64_t> strides(sizes.size(), 1);
    for (int i = sizes.size() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * sizes[i + 1];

    auto view = mlir::memref::ReinterpretCastOp::create(
        rewriter, loc, resTy, src, /*offset*/ int64_t{0}, sizes, strides);
    rewriter.replaceOp(op, view.getResult());
    return mlir::success();
  }
};

// ============ TransposeTensorOp: mydialect.transpose_tensor → memref 转置拷贝 ============
// 按 perm 属性生成循环：输出第 j 维迭代变量 iv 映射到输入第 perm[j] 维索引。
// 0 维 / rank-1（perm=[0]）退化为单元素拷贝。
struct TransposeTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::TransposeTensorOp> {
  using mlir::OpConversionPattern<
      mydialect::TransposeTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::TransposeTensorOp op,
      mydialect::TransposeTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value src = ensureMemref(adaptor.getData(), tc, rewriter, loc);

    llvm::SmallVector<int64_t> perm;
    for (mlir::Attribute a : op.getPerm())
      perm.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());

    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> outIvs) {
      llvm::SmallVector<mlir::Value> inIvs(perm.size());
      for (auto [j, iv] : llvm::enumerate(outIvs))
        inIvs[perm[j]] = iv;
      mlir::Value v = rewriter.create<mlir::memref::LoadOp>(loc, src, inIvs);
      rewriter.create<mlir::memref::StoreOp>(loc, v, res, outIvs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ ReduceSumTensorOp: mydialect.reduce_sum_tensor → 清零 + 累加循环 ============
// 两步法：
//   1) 遍历输出 shape 所有元素写零（memref.alloc 不保证零初始化）；
//   2) 遍历输入 shape，把每个输入元素累加到对应的输出槽
//      （被归约维用 0，保留维直接用输入索引）。
// 单线程下每个输出元素只被一个线程写，无需 atomic。
struct ReduceSumTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::ReduceSumTensorOp> {
  using mlir::OpConversionPattern<
      mydialect::ReduceSumTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::ReduceSumTensorOp op,
      mydialect::ReduceSumTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value src = ensureMemref(adaptor.getData(), tc, rewriter, loc);
    auto srcTy = mlir::cast<mlir::MemRefType>(src.getType());
    mlir::Type elemTy = resTy.getElementType();
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);

    bool keepDims = op.getKeepDims();
    llvm::SmallDenseSet<int64_t> axes;
    for (mlir::Attribute a : op.getAxes())
      axes.insert(mlir::cast<mlir::IntegerAttr>(a).getInt());

    // 1) 清零（memref.alloc 不保证零初始化）
    auto zero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(elemTy));
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> outIvs) {
      rewriter.create<mlir::memref::StoreOp>(loc, zero, res, outIvs);
    });

    // 2) 按输入遍历累加
    auto c0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    int64_t srcRank = srcTy.getRank();
    llvm::SmallVector<int64_t> srcShape(srcTy.getShape().begin(),
                                         srcTy.getShape().end());
    emitLoopNest(rewriter, loc, srcShape,
                 [&](llvm::ArrayRef<mlir::Value> inIvs) {
      llvm::SmallVector<mlir::Value> outIvs;
      for (int64_t d = 0; d < srcRank; ++d) {
        if (axes.contains(d)) {
          if (keepDims) outIvs.push_back(c0.getResult());
        } else {
          outIvs.push_back(inIvs[d]);
        }
      }
      mlir::Value v   = rewriter.create<mlir::memref::LoadOp>(loc, src, inIvs);
      mlir::Value acc = rewriter.create<mlir::memref::LoadOp>(loc, res, outIvs);
      mlir::Value sum = mlir::isa<mlir::IntegerType>(elemTy)
        ? (mlir::Value)mlir::arith::AddIOp::create(rewriter, loc, acc, v)
        : (mlir::Value)mlir::arith::AddFOp::create(rewriter, loc, acc, v);
      rewriter.create<mlir::memref::StoreOp>(loc, sum, res, outIvs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ ReluTensorOp: mydialect.relu_tensor → 逐元素 max(0, x) ============
// Arith dialect 没有 arith.maxsi/maxf，所以用 cmpi/cmpf + select 组合：
//   整数: cmpi sgt(x, 0) → select(x, 0, flag)
//   浮点: cmpf ogt(x, 0.0) → select(x, 0.0, flag)
struct ReluTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::ReluTensorOp> {
  using mlir::OpConversionPattern<mydialect::ReluTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::ReluTensorOp op, mydialect::ReluTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value src = ensureMemref(adaptor.getX(), tc, rewriter, loc);
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    mlir::Type elemTy = resTy.getElementType();

    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      mlir::Value x = rewriter.create<mlir::memref::LoadOp>(loc, src, idxs);
      mlir::Value zero;
      if (mlir::isa<mlir::IntegerType>(elemTy)) {
        zero = mlir::arith::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(elemTy));
        auto flag = mlir::arith::CmpIOp::create(
            rewriter, loc, mlir::arith::CmpIPredicate::sgt, x, zero);
        auto relu = mlir::arith::SelectOp::create(rewriter, loc, flag, x, zero);
        rewriter.create<mlir::memref::StoreOp>(loc, relu, res, idxs);
      } else if (mlir::isa<mlir::FloatType>(elemTy)) {
        zero = mlir::arith::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(elemTy));
        auto flag = mlir::arith::CmpFOp::create(
            rewriter, loc, mlir::arith::CmpFPredicate::OGT, x, zero);
        auto relu = mlir::arith::SelectOp::create(rewriter, loc, flag, x, zero);
        rewriter.create<mlir::memref::StoreOp>(loc, relu, res, idxs);
      } else {
        rewriter.notifyMatchFailure(op, "unsupported element type for relu");
      }
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ============ MatMulTensorOp: mydialect.matmul_tensor → 3 层循环 ============
// Y[M,N] = Σ_k A[m,k] * B[k,n]
// 实现：
//   1) 分配 Y[M,N]，清零（memref.alloc 不保证零初始化）
//   2) 外层 m,n 遍历输出位置
//   3) 内层 k 累加：Y[m,n] += A[m,k] * B[k,n]
// 约束：rank=2 静态 shape（由 verifier 保证）。
struct MatMulTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::MatMulTensorOp> {
  using mlir::OpConversionPattern<mydialect::MatMulTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::MatMulTensorOp op, mydialect::MatMulTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto aTy = mlir::cast<mlir::MemRefType>(tc.convertType(op.getA().getType()));
    auto bTy = mlir::cast<mlir::MemRefType>(tc.convertType(op.getB().getType()));
    auto resTy = mlir::cast<mlir::MemRefType>(tc.convertType(op.getY().getType()));
    mlir::Value A = ensureMemref(adaptor.getA(), tc, rewriter, loc);
    mlir::Value B = ensureMemref(adaptor.getB(), tc, rewriter, loc);

    if (!aTy.hasStaticShape() || !bTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shape not supported");
    int64_t M = aTy.getShape()[0];
    int64_t K = aTy.getShape()[1];
    int64_t N = bTy.getShape()[1];
    mlir::Type elemTy = resTy.getElementType();

    auto Y = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    auto zeroVal = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(elemTy));

    // 1) 清零 Y（M*N 次，用 emitLoopNest 两层）
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> mnIvs) {
      rewriter.create<mlir::memref::StoreOp>(loc, zeroVal, Y, mnIvs);
    });

    // 2) m, n 外层循环（遍历输出位置）
    auto c0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto cK = mlir::arith::ConstantIndexOp::create(rewriter, loc, K);
    auto c1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);

    // 为嵌套 m 循环建立 insert guard
    mlir::OpBuilder::InsertionGuard outerGuard(rewriter);

    // --- m 循环 ---
    auto mLoop = mlir::scf::ForOp::create(rewriter, loc, c0,
        mlir::arith::ConstantIndexOp::create(rewriter, loc, M), c1);
    auto mIv = mLoop.getInductionVar();
    rewriter.setInsertionPointToStart(mLoop.getBody());

    // --- n 循环 ---
    auto nLoop = mlir::scf::ForOp::create(rewriter, loc, c0,
        mlir::arith::ConstantIndexOp::create(rewriter, loc, N), c1);
    auto nIv = nLoop.getInductionVar();
    rewriter.setInsertionPointToStart(nLoop.getBody());

    // --- k 循环（累加）---
    auto kLoop = mlir::scf::ForOp::create(rewriter, loc, c0, cK, c1);
    auto kIv = kLoop.getInductionVar();
    rewriter.setInsertionPointToStart(kLoop.getBody());

    mlir::Value aEl = rewriter.create<mlir::memref::LoadOp>(loc, A, mlir::ValueRange{mIv, kIv});
    mlir::Value bEl = rewriter.create<mlir::memref::LoadOp>(loc, B, mlir::ValueRange{kIv, nIv});
    mlir::Value prod = mlir::isa<mlir::IntegerType>(elemTy)
      ? (mlir::Value)mlir::arith::MulIOp::create(rewriter, loc, aEl, bEl)
      : (mlir::Value)mlir::arith::MulFOp::create(rewriter, loc, aEl, bEl);
    mlir::Value acc = rewriter.create<mlir::memref::LoadOp>(loc, Y, mlir::ValueRange{mIv, nIv});
    mlir::Value sum = mlir::isa<mlir::IntegerType>(elemTy)
      ? (mlir::Value)mlir::arith::AddIOp::create(rewriter, loc, acc, prod)
      : (mlir::Value)mlir::arith::AddFOp::create(rewriter, loc, acc, prod);
    rewriter.create<mlir::memref::StoreOp>(loc, sum, Y, mlir::ValueRange{mIv, nIv});

    // k / n / m scf.for 默认会自动 yield，无需额外操作
    rewriter.replaceOp(op, Y.getResult());
    return mlir::success();
  }
};

// ============ LoopOp: mydialect.loop → scf.while ============
// 对标 ONNX Loop 语义：cond && (i < maxTrip) 时执行循环体，
// 循环体产出 (newCond, newCarries)。
// scf.while 携带值布局：(i: i64, cond: i1, carries: memref...)，
//   before 块：计算 keepGoing = cond && (i < maxTrip)，scf.condition 转发全部参数；
//   after 块：inline 原循环体，loop_yield (cond, carries) →
//             scf.yield (i+1, cond, carries)。
// while 结果前两个是 (finalI, finalCond)，丢弃，仅用 finalCarries 替换原 op。
struct LoopOpLowering : public mlir::OpConversionPattern<mydialect::LoopOp> {
  using mlir::OpConversionPattern<mydialect::LoopOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::LoopOp op, mydialect::LoopOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 防御：上游 op 未转换时 initArgs 可能仍是 MyTensorType，补 cast 到 memref
    auto ensureConverted = [&](mlir::Value v) -> mlir::Value {
      mlir::Type targetTy = getTypeConverter()->convertType(v.getType());
      if (targetTy == v.getType())
        return v;
      return rewriter.create<mlir::UnrealizedConversionCastOp>(loc, targetTy, v)
          .getResult(0);
    };

    // while 携带初始值：(i=0:i64, initCond, initArgs...)
    auto zeroI64 = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(0));
    llvm::SmallVector<mlir::Value> inits;
    inits.push_back(zeroI64.getResult());
    inits.push_back(adaptor.getInitCond());
    for (auto v : adaptor.getInitArgs())
      inits.push_back(ensureConverted(v));

    llvm::SmallVector<mlir::Type> resTypes;
    for (auto v : inits)
      resTypes.push_back(v.getType());

    mlir::Value maxTrip = adaptor.getMaxTripCount();
    // 注意：必须显式传 before/after builder 回调。scf::WhileOp 的 ODS 只声明
    // 带回调的 4 参 builder；省掉回调会落到 TableGen 默认 aggregate builder，
    // 只创建空 region（无 entry block），getBeforeBody() 对空 region 解引用。
    // 这里传空回调让块（含块参数）先建好，create 返回后再填充块内容。
    auto whileOp = mlir::scf::WhileOp::create(
        rewriter, loc, resTypes, inits,
        [](mlir::OpBuilder &, mlir::Location, mlir::ValueRange) {},
        [](mlir::OpBuilder &, mlir::Location, mlir::ValueRange) {});

    // ---- before 块：keepGoing = cond && (i < maxTrip) ----
    mlir::Block *beforeBlock = whileOp.getBeforeBody();
    rewriter.setInsertionPointToStart(beforeBlock);
    auto cmp = mlir::arith::CmpIOp::create(rewriter, loc,
        mlir::arith::CmpIPredicate::slt, beforeBlock->getArgument(0), maxTrip);
    auto keep = mlir::arith::AndIOp::create(rewriter, loc,
        beforeBlock->getArgument(1), cmp.getResult());
    mlir::scf::ConditionOp::create(rewriter, loc, keep.getResult(),
                                   beforeBlock->getArguments());

    // ---- after 块：inline 原循环体 ----
    // inline 之前先检查 yield 的 cond 是否直接透传 body block arg：
    // ConversionPatternRewriter 的块参数替换是延迟生效的，inline 后再查
    // yieldOp.getCond() 拿到的仍是旧 block arg（MyTensor 类型），framework 收尾
    // 时会反插 i1->MyTensor 的 materialization，与 ensureConverted 的
    // MyTensor->memref cast 连成两端类型不同的链，无法被 canonicalize 折叠。
    mlir::Block *srcBody = &op.getBody().front();
    auto srcYield =
        llvm::cast<mydialect::LoopYieldOp>(srcBody->getTerminator());
    int condBlockArgIdx = -1;
    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(srcYield.getCond()))
      if (barg.getOwner() == srcBody)
        condBlockArgIdx = barg.getArgNumber();

    mlir::Block *afterBlock = whileOp.getAfterBody();
    llvm::SmallVector<mlir::Value> replacements(afterBlock->getArguments().begin(),
                                                afterBlock->getArguments().end());
    rewriter.inlineBlockBefore(srcBody, afterBlock, afterBlock->end(),
                               replacements);

    // loop_yield (cond, carries...) → scf.yield (i+1, cond, carries...)
    auto yieldOp =
        llvm::cast<mydialect::LoopYieldOp>(afterBlock->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    auto oneI64 = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto nextI = mlir::arith::AddIOp::create(rewriter, loc,
        afterBlock->getArgument(0), oneI64.getResult());
    llvm::SmallVector<mlir::Value> yieldVals;
    yieldVals.push_back(nextI.getResult());
    // cond 必须是 i1 标量。ONNX 转换来的 body 里 cond 常是 0 维张量，分两种情况：
    //   1. cond 直接透传 body block arg（inline 前已记录位置）——after 块同位置
    //      的参数就是 while 携带的 i1 标量，直接使用；
    //   2. body 内真实算出的 0 维张量（如 cmpi_tensor）——转 memref 后 load 出 i1，
    //      该 cast 会在上游 op 转换完成后被 framework 消解。
    mlir::Value condI1;
    if (condBlockArgIdx >= 0 &&
        afterBlock->getArgument(condBlockArgIdx).getType().isInteger(1)) {
      condI1 = afterBlock->getArgument(condBlockArgIdx);
    } else {
      mlir::Value v = yieldOp.getCond();
      if (v.getType().isInteger(1)) {
        condI1 = v;
      } else {
        mlir::Value mv = ensureConverted(v); // 0-d tensor -> memref<i1>
        condI1 = rewriter.create<mlir::memref::LoadOp>(loc, mv,
                                                       mlir::ValueRange{})
                     .getResult();
      }
    }
    yieldVals.push_back(condI1);
    // carry 值必须转成 memref：若上游（如 add_tensor）尚未转换，
    // 直接用旧值会让 scf.yield 操作数残留 tensor 类型，与 before 块参数不匹配
    for (auto v : yieldOp.getCarryArgs())
      yieldVals.push_back(ensureConverted(v));
    mlir::scf::YieldOp::create(rewriter, loc, yieldVals);
    rewriter.eraseOp(yieldOp);

    // 结果：丢弃 (finalI, finalCond)，保留 finalCarries
    rewriter.replaceOp(op, whileOp.getResults().drop_front(2));
    return mlir::success();
  }
};

// ========== TensorToScalarOpLowering: MyTensorType(0维) → i32 ==========
// SimpleCParser 桥接 op，MyTensorType 经 converter 转 memref<i32>（0维），
// 用 memref.load 取出标量。
struct TensorToScalarOpLowering
    : public mlir::OpConversionPattern<mydialect::TensorToScalarOp> {
  using OpConversionPattern<mydialect::TensorToScalarOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::TensorToScalarOp op,
      typename mydialect::TensorToScalarOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // adaptor.getInput() 已经是 converter 处理后的 memref<i32>（0维）
    // 0 维 memref.load 不需要 index
    auto load = rewriter.create<mlir::memref::LoadOp>(
        op.getLoc(), adaptor.getInput(), mlir::ValueRange{});
    rewriter.replaceOp(op, load.getResult());
    return mlir::success();
  }
};

// ========== ScalarToTensorOpLowering: i32 → MyTensorType(0维) ==========
// 桥接 op，result 经 converter 转 memref<i32>，用 memref.alloc + store 实现。
struct ScalarToTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::ScalarToTensorOp> {
  using OpConversionPattern<mydialect::ScalarToTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::ScalarToTensorOp op,
      typename mydialect::ScalarToTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto convertedTy = getTypeConverter()->convertType(op.getOutput().getType());
    if (!convertedTy) {
      return mlir::failure();
    }
    auto memrefTy = mlir::cast<mlir::MemRefType>(convertedTy);
    auto loc = op.getLoc();
    // 0 维 memref.alloc 不需要 dynamic sizes
    auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, memrefTy);
    rewriter.create<mlir::memref::StoreOp>(
        loc, adaptor.getInput(), alloc.getResult(), mlir::ValueRange{});
    rewriter.replaceOp(op, alloc.getResult());
    return mlir::success();
  }
};

// ========== ReduceMaxTensorOp: reduce_max_tensor → 循环 + cmp max ==========
// 与 ReduceSum 同构，但归约时用 max 而非 sum：
//   初始化 -inf（浮点）或 min（整数），然后 cmp max 累积。
struct ReduceMaxTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::ReduceMaxTensorOp> {
  using mlir::OpConversionPattern<
      mydialect::ReduceMaxTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::ReduceMaxTensorOp op,
      mydialect::ReduceMaxTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value src = ensureMemref(adaptor.getData(), tc, rewriter, loc);
    auto srcTy = mlir::cast<mlir::MemRefType>(src.getType());
    mlir::Type elemTy = resTy.getElementType();
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);

    bool keepDims = op.getKeepDims();
    llvm::SmallDenseSet<int64_t> axes;
    for (mlir::Attribute a : op.getAxes())
      axes.insert(mlir::cast<mlir::IntegerAttr>(a).getInt());

    // 1) 初始化：-inf（浮点）或 int min（整数）
    mlir::Value initVal;
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(elemTy)) {
      llvm::APFloat negInf = llvm::APFloat::getInf(ft.getFloatSemantics(), /*Negative=*/true);
      initVal = mlir::arith::ConstantOp::create(
          rewriter, loc, mlir::FloatAttr::get(ft, negInf));
    } else if (auto it = mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
      initVal = mlir::arith::ConstantOp::create(
          rewriter, loc,
          mlir::IntegerAttr::get(it, llvm::APInt::getSignedMinValue(it.getWidth())));
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported element type for reduce_max");
    }
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> outIvs) {
      rewriter.create<mlir::memref::StoreOp>(loc, initVal, res, outIvs);
    });

    // 2) 遍历输入，逐元素 cmp max
    auto c0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    int64_t srcRank = srcTy.getRank();
    llvm::SmallVector<int64_t> srcShape(srcTy.getShape().begin(),
                                         srcTy.getShape().end());
    emitLoopNest(rewriter, loc, srcShape,
                 [&](llvm::ArrayRef<mlir::Value> inIvs) {
      llvm::SmallVector<mlir::Value> outIvs;
      for (int64_t d = 0; d < srcRank; ++d) {
        if (axes.contains(d)) {
          if (keepDims) outIvs.push_back(c0.getResult());
        } else {
          outIvs.push_back(inIvs[d]);
        }
      }
      mlir::Value v = rewriter.create<mlir::memref::LoadOp>(loc, src, inIvs);
      mlir::Value acc = rewriter.create<mlir::memref::LoadOp>(loc, res, outIvs);
      mlir::Value newVal;
      if (mlir::isa<mlir::FloatType>(elemTy)) {
        auto pred = rewriter.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGT, v, acc);
        newVal = rewriter.create<mlir::arith::SelectOp>(loc, pred, v, acc);
      } else {
        auto pred = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sgt, v, acc);
        newVal = rewriter.create<mlir::arith::SelectOp>(loc, pred, v, acc);
      }
      rewriter.create<mlir::memref::StoreOp>(loc, newVal, res, outIvs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ========== ExpTensorOp: exp_tensor → 循环 + math.exp ==========
struct ExpTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::ExpTensorOp> {
  using mlir::OpConversionPattern<mydialect::ExpTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::ExpTensorOp op,
      mydialect::ExpTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value src = ensureMemref(adaptor.getX(), tc, rewriter, loc);
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    emitLoopNest(rewriter, loc, resTy.getShape(),
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      mlir::Value x = rewriter.create<mlir::memref::LoadOp>(loc, src, idxs);
      mlir::Value ex = rewriter.create<mlir::math::ExpOp>(loc, x);
      rewriter.create<mlir::memref::StoreOp>(loc, ex, res, idxs);
    });
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ========== ConcatTensorOp: concat_tensor → 逐层循环 + memref.store ==========
// 沿 axis 把所有输入拷贝到输出。实现：对每个输入 in，计算其在输出中的
// 起始偏移 offset[axis]（累加前面输入的 shape[axis]），然后双层循环
// （外层遍历 in 的所有非 axis 维，内层遍历 axis 维）做 copy。
struct ConcatTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::ConcatTensorOp> {
  using mlir::OpConversionPattern<
      mydialect::ConcatTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::ConcatTensorOp op,
      mydialect::ConcatTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    auto res = rewriter.create<mlir::memref::AllocOp>(loc, resTy);
    auto rank = (int64_t)resTy.getRank();

    int64_t rawAxis = (int64_t)op.getAxis();
    int64_t axis = rawAxis < 0 ? rawAxis + rank : rawAxis;

    // 先把每个输入转成 memref
    llvm::SmallVector<mlir::Value> srcs;
    for (auto v : adaptor.getInputs()) {
      srcs.push_back(ensureMemref(v, tc, rewriter, loc));
    }
    if (srcs.empty())
      return rewriter.notifyMatchFailure(op, "no inputs to concat");

    int64_t offset = 0;
    auto c0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    for (auto &srcVal : srcs) {
      auto srcTy = mlir::cast<mlir::MemRefType>(srcVal.getType());
      llvm::SmallVector<int64_t> srcShape(srcTy.getShape().begin(),
                                         srcTy.getShape().end());
      int64_t axisSize = srcShape[axis];
      if (axisSize == 0) continue;

      // emitLoopNest 遍历 src 的所有维度，body 内：
      //   outIdx[axis] = inIdx[axis] + offset  （index 加法）
      //   其余 outIdx[d] = inIdx[d]
      emitLoopNest(rewriter, loc, srcShape,
                   [&](llvm::ArrayRef<mlir::Value> inIvs) {
        llvm::SmallVector<mlir::Value> outIvs;
        for (int64_t d = 0; d < rank; ++d) {
          if (d == axis) {
            auto offVal = mlir::arith::ConstantIndexOp::create(
                rewriter, loc, offset);
            auto outIdx = mlir::arith::AddIOp::create(
                rewriter, loc, inIvs[d], offVal);
            outIvs.push_back(outIdx.getResult());
          } else {
            outIvs.push_back(inIvs[d]);
          }
        }
        mlir::Value v = rewriter.create<mlir::memref::LoadOp>(loc, srcVal, inIvs);
        rewriter.create<mlir::memref::StoreOp>(loc, v, res, outIvs);
      });
      offset += axisSize;
    }
    rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

// ========== Conv2dTensorOp: conv2d_tensor → 5 层 scf.for ==========
// 基础 NCHW 实现：
//   for n, c_out, oh, ow, c_in, kh, kw:
//     ih = oh*stride_h + kh*dilation_h - pad_top
//     iw = ow*stride_w + kw*dilation_w - pad_left
//     acc += input[n, c_in, ih, iw] * weight[c_out, c_in, kh, kw]
// 为了简化，我们把 c_in/kh/kw 展开成单一的累加循环（emitLoopNest 3 维），
// 外层 n/c_out/oh/ow 手工构造，body 内累加并 store 回 res[n, c_out, oh, ow]。
struct Conv2dTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::Conv2dTensorOp> {
  using mlir::OpConversionPattern<mydialect::Conv2dTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::Conv2dTensorOp op,
      mydialect::Conv2dTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getOutput().getType()));
    auto input = ensureMemref(adaptor.getInput(), tc, rewriter, loc);
    auto weight = ensureMemref(adaptor.getWeight(), tc, rewriter, loc);
    auto inputTy = mlir::cast<mlir::MemRefType>(input.getType());
    auto weightTy = mlir::cast<mlir::MemRefType>(weight.getType());
    mlir::Type elemTy = resTy.getElementType();
    auto out = rewriter.create<mlir::memref::AllocOp>(loc, resTy);

    int64_t N = inputTy.getShape()[0];
    int64_t C_in = inputTy.getShape()[1];
    int64_t H = inputTy.getShape()[2];
    int64_t W = inputTy.getShape()[3];
    int64_t C_out = weightTy.getShape()[0];
    int64_t kH = weightTy.getShape()[2];
    int64_t kW = weightTy.getShape()[3];
    int64_t OH = resTy.getShape()[2];
    int64_t OW = resTy.getShape()[3];

    // 属性处理
    auto padsAttr = op.getPadsAttr();  // OptionalAttr<ArrayAttr> → generated accessor
    auto stridesAttr = op.getStrides();
    auto dilationsAttr = op.getDilations();
    int64_t padTop = 0, padLeft = 0;
    int64_t strideH = 1, strideW = 1;
    int64_t dilH = 1, dilW = 1;
    if (padsAttr && padsAttr.size() >= 2) {
      padTop = mlir::cast<mlir::IntegerAttr>(padsAttr[0]).getInt();
      padLeft = mlir::cast<mlir::IntegerAttr>(padsAttr[1]).getInt();
    }
    if (stridesAttr && !stridesAttr.empty()) {
      strideH = mlir::cast<mlir::IntegerAttr>(stridesAttr[0]).getInt();
      if (stridesAttr.size() > 1)
        strideW = mlir::cast<mlir::IntegerAttr>(stridesAttr[1]).getInt();
    }
    if (dilationsAttr && !dilationsAttr.empty()) {
      dilH = mlir::cast<mlir::IntegerAttr>(dilationsAttr[0]).getInt();
      if (dilationsAttr.size() > 1)
        dilW = mlir::cast<mlir::IntegerAttr>(dilationsAttr[1]).getInt();
    }

    auto c0i = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(elemTy));
    auto c0idx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    mlir::OpBuilder::InsertionGuard outerGuard(rewriter);

    // 外层 4 层：N, C_out, OH, OW —— 用手工 scf.for + emitLoopNest 不太方便，
    // 简化：用 4 层手工 scf.for 嵌套（避免 InsertionGuard 问题）。
    auto mLoop = mlir::scf::ForOp::create(
        rewriter, loc, c0idx.getResult(),
        mlir::arith::ConstantIndexOp::create(rewriter, loc, N),
        c0idx.getResult());
    rewriter.setInsertionPointToStart(mLoop.getBody());

    auto coLoop = mlir::scf::ForOp::create(
        rewriter, loc, c0idx.getResult(),
        mlir::arith::ConstantIndexOp::create(rewriter, loc, C_out),
        c0idx.getResult());
    rewriter.setInsertionPointToStart(coLoop.getBody());

    auto ohLoop = mlir::scf::ForOp::create(
        rewriter, loc, c0idx.getResult(),
        mlir::arith::ConstantIndexOp::create(rewriter, loc, OH),
        c0idx.getResult());
    rewriter.setInsertionPointToStart(ohLoop.getBody());

    auto owLoop = mlir::scf::ForOp::create(
        rewriter, loc, c0idx.getResult(),
        mlir::arith::ConstantIndexOp::create(rewriter, loc, OW),
        c0idx.getResult());
    rewriter.setInsertionPointToStart(owLoop.getBody());

    // 清零 out[n, co, oh, ow]
    auto nIv = mLoop.getBody()->getArgument(0);
    auto coIv = coLoop.getBody()->getArgument(0);
    auto ohIv = ohLoop.getBody()->getArgument(0);
    auto owIv = owLoop.getBody()->getArgument(0);
    rewriter.create<mlir::memref::StoreOp>(
        loc, c0i.getResult(), out, mlir::ValueRange{nIv, coIv, ohIv, owIv});

    // 累加循环：emitLoopNest({C_in, kH, kW})
    llvm::SmallVector<int64_t> accShape = {C_in, kH, kW};
    emitLoopNest(rewriter, loc, accShape,
                 [&](llvm::ArrayRef<mlir::Value> accIvs) {
      auto ciIv = accIvs[0];
      auto khIv = accIvs[1];
      auto kwIv = accIvs[2];

      // ih = oh * strideH + kh * dilH - padTop
      auto ih0 = mlir::arith::MulIOp::create(rewriter, loc, ohIv,
          mlir::arith::ConstantIndexOp::create(rewriter, loc, strideH));
      auto ih1 = mlir::arith::MulIOp::create(rewriter, loc, khIv,
          mlir::arith::ConstantIndexOp::create(rewriter, loc, dilH));
      auto ih2 = mlir::arith::AddIOp::create(rewriter, loc, ih0.getResult(), ih1.getResult());
      auto ih = mlir::arith::SubIOp::create(rewriter, loc, ih2.getResult(),
          mlir::arith::ConstantIndexOp::create(rewriter, loc, padTop));

      auto iw0 = mlir::arith::MulIOp::create(rewriter, loc, owIv,
          mlir::arith::ConstantIndexOp::create(rewriter, loc, strideW));
      auto iw1 = mlir::arith::MulIOp::create(rewriter, loc, kwIv,
          mlir::arith::ConstantIndexOp::create(rewriter, loc, dilW));
      auto iw2 = mlir::arith::AddIOp::create(rewriter, loc, iw0.getResult(), iw1.getResult());
      auto iw = mlir::arith::SubIOp::create(rewriter, loc, iw2.getResult(),
          mlir::arith::ConstantIndexOp::create(rewriter, loc, padLeft));

      auto inVal = rewriter.create<mlir::memref::LoadOp>(
          loc, input, mlir::ValueRange{nIv, ciIv, ih.getResult(), iw.getResult()});
      auto wVal = rewriter.create<mlir::memref::LoadOp>(
          loc, weight, mlir::ValueRange{coIv, ciIv, khIv, kwIv});
      auto prod = mlir::isa<mlir::IntegerType>(elemTy)
          ? (mlir::Value)mlir::arith::MulIOp::create(rewriter, loc, inVal, wVal)
          : (mlir::Value)mlir::arith::MulFOp::create(rewriter, loc, inVal, wVal);
      auto acc = rewriter.create<mlir::memref::LoadOp>(
          loc, out, mlir::ValueRange{nIv, coIv, ohIv, owIv});
      auto sum = mlir::isa<mlir::IntegerType>(elemTy)
          ? (mlir::Value)mlir::arith::AddIOp::create(rewriter, loc, acc, prod)
          : (mlir::Value)mlir::arith::AddFOp::create(rewriter, loc, acc, prod);
      rewriter.create<mlir::memref::StoreOp>(
          loc, sum, out, mlir::ValueRange{nIv, coIv, ohIv, owIv});
    });

    rewriter.replaceOp(op, out.getResult());
    return mlir::success();
  }
};

// ========== SoftmaxTensorOp: softmax_tensor → 3 次扫描 ==========
// softmax(x, axis) = exp(x - max_axis(x)) / sum_axis(exp(x - max_axis(x)))
//
// 实现策略：
// 1. 扫描输入，沿 axis 找 max → 存入 max_buf（axis 维 size=1）
// 2. 扫描输入，计算 exp(x - max_buf)，同时沿 axis 累积 sum → exp_buf + sum_buf
// 3. 扫描 exp_buf，计算 exp / sum_buf → 存入 output
//
// 每次扫描用 emitLoopNest 遍历输入 shape，body 内手工从 full index 构造
// max_buf/sum_buf 的索引（axis 维固定为 0）。
struct SoftmaxTensorOpLowering
    : public mlir::OpConversionPattern<mydialect::SoftmaxTensorOp> {
  using mlir::OpConversionPattern<mydialect::SoftmaxTensorOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mydialect::SoftmaxTensorOp op,
      mydialect::SoftmaxTensorOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto &tc = *getTypeConverter();
    auto resTy = mlir::cast<mlir::MemRefType>(
        tc.convertType(op.getRes().getType()));
    mlir::Value src = ensureMemref(adaptor.getX(), tc, rewriter, loc);
    auto srcTy = mlir::cast<mlir::MemRefType>(src.getType());
    mlir::Type elemTy = resTy.getElementType();
    auto out = rewriter.create<mlir::memref::AllocOp>(loc, resTy);

    int64_t rank = (int64_t)srcTy.getRank();
    int64_t rawAxis = (int64_t)op.getAxis();
    int64_t axis = rawAxis < 0 ? rawAxis + rank : rawAxis;

    // max_buf 和 sum_buf 的 shape：axis 维 = 1，其余同 src
    llvm::SmallVector<int64_t> reducedShape;
    for (int64_t d = 0; d < rank; ++d)
      reducedShape.push_back(d == axis ? 1 : srcTy.getShape()[d]);
    auto reducedMemrefTy = mlir::MemRefType::get(reducedShape, elemTy);
    auto maxBuf = rewriter.create<mlir::memref::AllocOp>(loc, reducedMemrefTy);
    auto sumBuf = rewriter.create<mlir::memref::AllocOp>(loc, reducedMemrefTy);

    // 构造 buf 的索引：axis 维固定 0
    auto c0idx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto makeBufIdxs = [&](llvm::ArrayRef<mlir::Value> fullIdxs) {
      llvm::SmallVector<mlir::Value> out;
      for (int64_t d = 0; d < rank; ++d)
        out.push_back(d == axis ? c0idx.getResult() : fullIdxs[d]);
      return out;
    };

    auto srcShapeVec = llvm::SmallVector<int64_t>(
        srcTy.getShape().begin(), srcTy.getShape().end());

    // Pass 1: 沿 axis 扫描找 max → max_buf
    mlir::Value negInfInit;
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(elemTy)) {
      negInfInit = mlir::arith::ConstantOp::create(rewriter, loc,
          mlir::FloatAttr::get(ft,
              llvm::APFloat::getInf(ft.getFloatSemantics(), /*Negative=*/true)));
    } else if (auto it = mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
      negInfInit = mlir::arith::ConstantOp::create(rewriter, loc,
          mlir::IntegerAttr::get(it, llvm::APInt::getSignedMinValue(it.getWidth())));
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported element type for softmax");
    }
    emitLoopNest(rewriter, loc, reducedShape,
                 [&](llvm::ArrayRef<mlir::Value> outIvs) {
      rewriter.create<mlir::memref::StoreOp>(loc, negInfInit, maxBuf, outIvs);
      rewriter.create<mlir::memref::StoreOp>(loc, negInfInit, sumBuf, outIvs);
    });

    // Pass 1: 扫描 src 沿 axis 累积 max 到 max_buf
    emitLoopNest(rewriter, loc, srcShapeVec,
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      auto x = rewriter.create<mlir::memref::LoadOp>(loc, src, idxs);
      auto bufIdxs = makeBufIdxs(idxs);
      auto acc = rewriter.create<mlir::memref::LoadOp>(loc, maxBuf, bufIdxs);
      mlir::Value pred, newVal;
      if (mlir::isa<mlir::FloatType>(elemTy)) {
        pred = rewriter.create<mlir::arith::CmpFOp>(loc,
            mlir::arith::CmpFPredicate::OGT, x, acc);
      } else {
        pred = rewriter.create<mlir::arith::CmpIOp>(loc,
            mlir::arith::CmpIPredicate::sgt, x, acc);
      }
      newVal = rewriter.create<mlir::arith::SelectOp>(loc, pred, x, acc);
      rewriter.create<mlir::memref::StoreOp>(loc, newVal, maxBuf, bufIdxs);
    });

    // Pass 2: 扫描 src，计算 exp(x - max) + 累积 sum
    auto zeroVal = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(elemTy));
    emitLoopNest(rewriter, loc, srcShapeVec,
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      auto x = rewriter.create<mlir::memref::LoadOp>(loc, src, idxs);
      auto bufIdxs = makeBufIdxs(idxs);
      auto mx = rewriter.create<mlir::memref::LoadOp>(loc, maxBuf, bufIdxs);
      mlir::Value diff;
      if (mlir::isa<mlir::FloatType>(elemTy))
        diff = rewriter.create<mlir::arith::SubFOp>(loc, x, mx);
      else
        diff = rewriter.create<mlir::arith::SubIOp>(loc, x, mx);
      auto ex = rewriter.create<mlir::math::ExpOp>(loc, diff);

      // 存 exp 到 out（临时），同时累加 sum
      rewriter.create<mlir::memref::StoreOp>(loc, ex, out, idxs);
      auto acc = rewriter.create<mlir::memref::LoadOp>(loc, sumBuf, bufIdxs);
      mlir::Value newSum;
      if (mlir::isa<mlir::FloatType>(elemTy))
        newSum = rewriter.create<mlir::arith::AddFOp>(loc, acc, ex);
      else
        newSum = rewriter.create<mlir::arith::AddIOp>(loc, acc, ex);
      rewriter.create<mlir::memref::StoreOp>(loc, newSum, sumBuf, bufIdxs);
    });

    // Pass 3: 扫描 out（存的是 exp），除以 sum
    emitLoopNest(rewriter, loc, srcShapeVec,
                 [&](llvm::ArrayRef<mlir::Value> idxs) {
      auto ex = rewriter.create<mlir::memref::LoadOp>(loc, out, idxs);
      auto bufIdxs = makeBufIdxs(idxs);
      auto sum = rewriter.create<mlir::memref::LoadOp>(loc, sumBuf, bufIdxs);
      mlir::Value result;
      if (mlir::isa<mlir::FloatType>(elemTy))
        result = rewriter.create<mlir::arith::DivFOp>(loc, ex, sum);
      else
        result = rewriter.create<mlir::arith::DivSIOp>(loc, ex, sum);
      rewriter.create<mlir::memref::StoreOp>(loc, result, out, idxs);
    });

    rewriter.replaceOp(op, out.getResult());
    return mlir::success();
  }
};

} // anonymous namespace

namespace mydialect {
void populateMyDialectToLowerPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::TypeConverter &converter) {
  patterns.add<WrapOpLowering,
	UnwrapOpLowering,
	PrintOpLowering,
	PrintStrOpLowering,
	AddI32OpLowering,
	MuliOpLowering,
	SubI32OpLowering,
	DivSIOpLowering,
	CmpiOpLowering,
	ForOpLowering,
	MyAllocToMemRef,
	MyLoadToMemRef,
	MyStoreToMemRef,
	ConstantTensorOpLowering,
	AddTensorOpLowering,
	MulTensorOpLowering,
	SubTensorOpLowering,
	DivTensorOpLowering,
	FusedElementwiseOpLowering,
	CmpITensorOpLowering,
	IdentityTensorOpLowering,
	ReshapeTensorOpLowering,
	TransposeTensorOpLowering,
	ReduceSumTensorOpLowering,
	ReluTensorOpLowering,
	MatMulTensorOpLowering,
	ReduceMaxTensorOpLowering,
	ExpTensorOpLowering,
	ConcatTensorOpLowering,
	Conv2dTensorOpLowering,
	SoftmaxTensorOpLowering,
	LoopOpLowering,
	TensorToScalarOpLowering,
	ScalarToTensorOpLowering>(converter, patterns.getContext());
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

static mlir::PassRegistration<ConstantFoldMyDialectPass> regFold;

#include "mlir/IR/DialectRegistry.h"
extern "C" void mlirRegisterAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mydialect::MyDialect>();
}

