// LoopOptimizePass.cpp：--mydialect-loop-optimize
// 针对 lowering 后生成的 scf.for 循环 nest 做三类优化：
//
//   1) 循环融合  (fusion)       ：相邻、同边界、无迭代携带值、无跨循环内存依赖的
//                                两个 scf.for 合并成一个
//   2) 循环交换  (interchange)  ：完美嵌套的两层循环，当外层 IV 落在 memref 最内维
//                                索引位置时交换内外层，改善访存局部性
//   3) 向量化    (vectorize)    ：最内层逐元素循环（load→纯算→store）在 trip count
//                                为 vector-width 整数倍时改写成 vector.transfer_read/
//                                write + 向量算术，引入 vector 方言做 SIMD
//
// 三类优化各自独立开关，按 融合 → 交换 → 向量化 的顺序各跑一遍 greedy。
#include "MyOptPass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace mlir;

namespace mydialect {
namespace {

// ==================== 公共辅助 ====================

// 剥离 memref 视图类操作，拿到底层 memref
static Value stripMemrefViews(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
            memref::ReinterpretCastOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    break;
  }
  return v;
}

// 保守别名判断：只有"能证明不别名"的情形返回 false。
//   - 新鲜分配（alloc/alloca）与任何预先存在的值（函数参数、其他分配、
//     global）都不别名；
//   - 两个不同新鲜分配也不别名。
static bool isFreshAlloc(Value v) {
  Operation *def = v.getDefiningOp();
  return def && isa<memref::AllocOp, memref::AllocaOp>(def);
}

static bool mayAlias(Value a, Value b) {
  if (a == b)
    return true;
  Value ba = stripMemrefViews(a);
  Value bb = stripMemrefViews(b);
  if (ba == bb)
    return true;
  if (isFreshAlloc(ba) || isFreshAlloc(bb))
    return false;
  return true;
}

// 收集循环体内的 memref 读集合 / 写集合
static void collectMemrefAccesses(Operation *root,
                                  SmallVectorImpl<Value> &reads,
                                  SmallVectorImpl<Value> &writes) {
  root->walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op))
      reads.push_back(load.getMemref());
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      writes.push_back(store.getMemref());
  });
}

// 两个循环边界是否等价（SSA 值相同，或常量值相同）
static bool sameBound(Value a, Value b) {
  if (a == b)
    return true;
  auto ca = getConstantIntValue(a);
  auto cb = getConstantIntValue(b);
  return ca && cb && *ca == *cb;
}

// v 的值是否（传递地）依赖 iv，且依赖链限制在 region 内
static bool dependsOn(Value v, Value iv, Region &region,
                      SmallPtrSetImpl<Value> &visited) {
  if (v == iv)
    return true;
  if (!visited.insert(v).second)
    return false;
  Operation *def = v.getDefiningOp();
  if (!def || !region.isAncestor(def->getParentRegion()))
    return false;
  return llvm::any_of(def->getOperands(),
                      [&](Value o) { return dependsOn(o, iv, region, visited); });
}

static bool dependsOn(Value v, Value iv, Region &region) {
  SmallPtrSet<Value, 16> visited;
  return dependsOn(v, iv, region, visited);
}

// ==================== 1) 循环融合 ====================
// 相邻两个 scf.for，满足：
//   - 均无 iter_args（无跨迭代携带依赖）
//   - 边界（lb/ub/step）等价
//   - 无跨循环内存依赖（写集合与另一循环的读写集合均不别名）
// 合并后两个 body 在同一轮迭代内顺序执行，语义等价。
struct LoopFusionPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 0)
      return failure();
    auto next = dyn_cast_or_null<scf::ForOp>(op->getNextNode());
    if (!next || next.getNumResults() != 0)
      return failure();
    if (!sameBound(op.getLowerBound(), next.getLowerBound()) ||
        !sameBound(op.getUpperBound(), next.getUpperBound()) ||
        !sameBound(op.getStep(), next.getStep()))
      return failure();

    // 依赖检查：任一循环的写入不得与另一循环的读/写别名
    SmallVector<Value, 8> reads1, writes1, reads2, writes2;
    collectMemrefAccesses(op, reads1, writes1);
    collectMemrefAccesses(next, reads2, writes2);
    for (Value w : writes1)
      for (Value r : llvm::concat<Value>(reads2, writes2))
        if (mayAlias(w, r))
          return failure();
    for (Value w : writes2)
      for (Value r : reads1)
        if (mayAlias(w, r))
          return failure();

    Block *dest = op.getBody();
    rewriter.eraseOp(dest->getTerminator());
    // 把 loop2 的 body 整体搬到 loop2 的末尾，块参数（IV）替换为 loop1 的 IV
    rewriter.inlineBlockBefore(next.getBody(), dest, dest->end(),
                               ValueRange{op.getInductionVar()});
    rewriter.eraseOp(next);
    return success();
  }
};

// ==================== 2) 循环交换 ====================
// 完美嵌套两层循环（外层体内除一个内层 scf.for 外只有纯算子），且：
//   - 两者均无 iter_args
//   - body 内写入的 memref 不被 body 读取（无原地更新 → 迭代间无依赖）
//   - 外层 IV 出现在 memref 最内维索引位置（内层 IV 不在），交换可改善局部性
struct LoopInterchangePattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  // 统计内层 body 的访问中，"最内维索引依赖 outerIv 不依赖 innerIv" 的个数，
  // 以及反过来"依赖 innerIv" 的个数。前者多为正收益。
  static void countInnerDimUses(Operation *root, Value outerIv, Value innerIv,
                                int64_t &outerUses, int64_t &innerUses) {
    Region &region = root->getRegion(0);
    root->walk([&](Operation *op) {
      SmallVector<Value, 4> indices;
      if (auto load = dyn_cast<memref::LoadOp>(op))
        indices.assign(load.getIndices().begin(), load.getIndices().end());
      else if (auto store = dyn_cast<memref::StoreOp>(op))
        indices.assign(store.getIndices().begin(), store.getIndices().end());
      else
        return;
      if (indices.empty())
        return;
      Value last = indices.back();
      bool usesOuter = dependsOn(last, outerIv, region);
      bool usesInner = dependsOn(last, innerIv, region);
      if (usesOuter && !usesInner)
        ++outerUses;
      if (usesInner && !usesOuter)
        ++innerUses;
    });
  }

  LogicalResult matchAndRewrite(scf::ForOp outer,
                                PatternRewriter &rewriter) const override {
    if (outer.getNumResults() != 0)
      return failure();
    if (!llvm::hasSingleElement(outer.getRegion()))
      return failure();

    // 外层体内：恰好一个内层 scf.for，其余必须是纯算子
    scf::ForOp inner;
    for (Operation &op : outer.getBody()->without_terminator()) {
      if (auto f = dyn_cast<scf::ForOp>(op)) {
        if (inner)
          return failure();
        inner = f;
      } else if (!isMemoryEffectFree(&op)) {
        return failure();
      }
    }
    if (!inner || inner.getNumResults() != 0 ||
        !llvm::hasSingleElement(inner.getRegion()))
      return failure();

    // 无原地更新：写集合与读集合不别名 → 迭代间无依赖，交换合法
    SmallVector<Value, 8> reads, writes;
    collectMemrefAccesses(inner, reads, writes);
    for (Value w : writes)
      for (Value r : reads)
        if (mayAlias(w, r))
          return failure();

    // 收益判定（方向性：交换后不再满足，避免反复交换）
    int64_t outerUses = 0, innerUses = 0;
    countInnerDimUses(inner, outer.getInductionVar(), inner.getInductionVar(),
                      outerUses, innerUses);
    if (outerUses <= innerUses)
      return failure();

    // 变换：新外层 = 原内层边界，新内层 = 原外层边界，
    // 原两层 body（纯算子 + 内层 body）按 IV 映射顺序克隆进新内层体内。
    Location loc = outer.getLoc();
    rewriter.setInsertionPoint(outer);
    auto newOuter = scf::ForOp::create(rewriter, loc, inner.getLowerBound(),
                                       inner.getUpperBound(),
                                       inner.getStep());
    rewriter.setInsertionPointToStart(newOuter.getBody());
    auto newInner = scf::ForOp::create(rewriter, loc, outer.getLowerBound(),
                                       outer.getUpperBound(),
                                       outer.getStep());
    rewriter.setInsertionPointToStart(newInner.getBody());

    IRMapping mapping;
    mapping.map(outer.getInductionVar(), newInner.getInductionVar());
    mapping.map(inner.getInductionVar(), newOuter.getInductionVar());
    for (Operation &op : outer.getBody()->without_terminator())
      if (!isa<scf::ForOp>(op))
        rewriter.clone(op, mapping);
    for (Operation &op : inner.getBody()->without_terminator())
      rewriter.clone(op, mapping);

    rewriter.eraseOp(outer);
    return success();
  }
};

// ==================== 3) 向量化 ====================
// 最内层 scf.for（lb/ub/step 均为常量、trip count 是 vector-width 的整数倍），
// body 形如：
//     %0 = memref.load %a[..., %i]
//     %1 = memref.load %b[..., %i]
//     %2 = arith.addf %0, %1
//     memref.store %2, %c[..., %i]
// 改写成：
//     scf.for %i = lb to ub step V {
//       %0 = vector.transfer_read %a[..., %i], pad : ..., vector<Vxf32>
//       %1 = vector.transfer_read %b[..., %i], pad : ..., vector<Vxf32>
//       %2 = arith.addf %0, %1 : vector<Vxf32>
//       vector.transfer_write %2, %c[..., %i] : vector<Vxf32>, ...
//     }
// 合法性：所有访问的最后一维索引恰为 IV（无 i±k 偏移、无跨 lane 依赖）；
//         其余索引在循环外定义（循环不变）。
struct VectorizeInnerLoopPattern : public OpRewritePattern<scf::ForOp> {
  VectorizeInnerLoopPattern(MLIRContext *ctx, int64_t vectorWidth)
      : OpRewritePattern<scf::ForOp>(ctx), vectorWidth(vectorWidth) {}

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewwriter) const override {
    PatternRewriter &rewriter = rewwriter;
    if (loop.getNumResults() != 0 || !llvm::hasSingleElement(loop.getRegion()))
      return failure();

    auto lbCst = getConstantIntValue(loop.getLowerBound());
    auto ubCst = getConstantIntValue(loop.getUpperBound());
    auto stCst = getConstantIntValue(loop.getStep());
    if (!lbCst || !ubCst || !stCst || *stCst != 1)
      return failure();
    int64_t tripCount = *ubCst - *lbCst;
    if (tripCount <= 0 || tripCount % vectorWidth != 0)
      return failure();

    Value iv = loop.getInductionVar();
    Block &body = *loop.getBody();

    // ---- 校验 body 结构：load(s) + 纯算 + 恰好一个 store ----
    Type elemTy = nullptr;
    int64_t numStores = 0;
    for (Operation &op : body.without_terminator()) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        auto idx = load.getIndices();
        if (idx.empty() || idx.back() != iv)
          return failure();
        for (Value other : idx.drop_back())
          if (!isDefinedOutsideLoop(other, loop))
            return failure();
        if (!elemTy)
          elemTy = load.getMemRefType().getElementType();
        else if (load.getMemRefType().getElementType() != elemTy)
          return failure();
        continue;
      }
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        ++numStores;
        auto idx = store.getIndices();
        if (idx.empty() || idx.back() != iv)
          return failure();
        for (Value other : idx.drop_back())
          if (!isDefinedOutsideLoop(other, loop))
            return failure();
        if (!elemTy)
          elemTy = store.getMemRefType().getElementType();
        else if (store.getMemRefType().getElementType() != elemTy)
          return failure();
        continue;
      }
      // 其余必须是纯标量算子，且结果类型就是元素类型（避免 index/i1 等）
      if (!isMemoryEffectFree(&op))
        return failure();
      for (Type t : op.getResultTypes())
        if (t != elemTy)
          return failure();
    }
    if (numStores != 1)
      return failure();

    // ---- 变换 ----
    Location loc = loop.getLoc();
    VectorType vecTy = VectorType::get({vectorWidth}, elemTy);

    // padding 零值：放在循环外，供 transfer_read 在越界时使用
    Value padding;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(loop);
      padding = arith::ConstantOp::create(rewriter, loc,
                                          rewriter.getZeroAttr(elemTy));
    }

    Value vecStep = arith::ConstantIndexOp::create(rewriter, loc, vectorWidth);
    auto newLoop = scf::ForOp::create(rewriter, loc, loop.getLowerBound(),
                                      loop.getUpperBound(), vecStep);
    rewriter.setInsertionPointToStart(newLoop.getBody());
    Value newIv = newLoop.getInductionVar();

    IRMapping mapping;
    mapping.map(iv, newIv);
    for (Operation &op : body.without_terminator()) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        SmallVector<Value, 4> indices(load.getIndices().begin(),
                                      load.getIndices().end());
        indices.back() = newIv;
        auto read = vector::TransferReadOp::create(
            rewriter, loc, vecTy, load.getMemref(), indices,
            std::optional<Value>(padding));
        mapping.map(load.getResult(), read.getResult());
        continue;
      }
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        auto idx = store.getIndices();
        SmallVector<Value, 4> indices(idx.begin(), idx.end());
        indices.back() = newIv;
        Value vec = mapping.lookup(store.getValue());
        vector::TransferWriteOp::create(rewriter, loc, vec, store.getMemref(),
                                        indices);
        continue;
      }
      // 纯标量算子：克隆后把结果类型改成向量，算术/比较等 SameOperandsAndResultType
      // 之外的算子也能保持元素级语义
      Operation *cloned = rewriter.clone(op, mapping);
      for (OpResult res : cloned->getResults())
        if (res.getType() == elemTy)
          res.setType(vecTy);
    }

    rewriter.eraseOp(loop);
    return success();
  }

  static bool isDefinedOutsideLoop(Value v, scf::ForOp loop) {
    if (auto blockArg = dyn_cast<BlockArgument>(v))
      return !loop.getRegion().isAncestor(blockArg.getOwner()->getParent());
    Operation *def = v.getDefiningOp();
    return def && !loop.getRegion().isAncestor(def->getParentRegion());
  }

  int64_t vectorWidth;
};

// ==================== Pass ====================
struct LoopOptimizePass
    : public PassWrapper<LoopOptimizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopOptimizePass)

  LoopOptimizePass() = default;
  LoopOptimizePass(const LoopOptimizePass &other) : PassWrapper(other) {}

  Option<bool> enableFusion{
      *this, "fusion",
      llvm::cl::desc("Fuse adjacent scf.for loops with identical bounds"),
      llvm::cl::init(true)};
  Option<bool> enableInterchange{
      *this, "interchange",
      llvm::cl::desc("Interchange nested loops to improve memory locality"),
      llvm::cl::init(true)};
  Option<bool> enableVectorize{
      *this, "vectorize",
      llvm::cl::desc("SIMD-vectorize the innermost elementwise loop"),
      llvm::cl::init(true)};
  Option<int64_t> vectorWidth{
      *this, "vector-width",
      llvm::cl::desc("Vectorization width (number of elements)"),
      llvm::cl::init(4)};

  StringRef getArgument() const final { return "mydialect-loop-optimize"; }
  StringRef getDescription() const final {
    return "Loop fusion / interchange / vectorization for elementwise scf.for nests";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, memref::MemRefDialect,
                    vector::VectorDialect>();
  }

  // 分阶段跑 greedy：融合 → 交换 → 向量化
  template <typename... Patterns>
  void runStage() {
    RewritePatternSet patterns(&getContext());
    patterns.add<Patterns...>(&getContext());
    GreedyRewriteConfig config;
    config.setMaxIterations(16);
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }

  void runOnOperation() override {
    if (vectorWidth <= 0) {
      signalPassFailure();
      return;
    }
    if (enableFusion)
      runStage<LoopFusionPattern>();
    if (enableInterchange)
      runStage<LoopInterchangePattern>();
    if (enableVectorize) {
      RewritePatternSet patterns(&getContext());
      patterns.add<VectorizeInnerLoopPattern>(&getContext(), vectorWidth.getValue());
      GreedyRewriteConfig config;
      config.setMaxIterations(16);
      config.setUseTopDownTraversal(true);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
        signalPassFailure();
    }
  }
};

static PassRegistration<LoopOptimizePass> regLoopOptimize;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createLoopOptimizePass() {
  return std::make_unique<LoopOptimizePass>();
}
} // namespace mydialect
