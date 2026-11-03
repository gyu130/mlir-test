// AffineOptimizePass.cpp：--mydialect-affine-opt
// 在 affine.for 循环 nest 上做三类循环变换，复用 affine 方言的循环分析：
//
//   1) 循环融合 (fusion)     ：相邻、边界相同、无跨循环内存依赖的 affine.for
//                             合并（与 scf 版本同构）
//   2) 循环分块 (tiling)     ：对完美嵌套的 affine.for band 调用 affine 的
//                             tilePerfectlyNested，strip-mine 出分块循环 + 块内循环
//   3) 向量化   (vectorize)  ：最内层逐元素 affine.for（load→纯算→store）在
//                             trip count 为 vector-width 整数倍时改写为
//                             vector.transfer_read/write + 向量算术
//
//   三类变换各自独立开关，按 融合 → 向量化 → 分块 的顺序执行
//   （向量化依赖常量 trip count，分块会破坏该前提，故先向量化）。
#include "MyOptPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

// 新鲜分配（alloc/alloca）
static bool isFreshAlloc(Value v) {
  Operation *def = v.getDefiningOp();
  return def && isa<memref::AllocOp, memref::AllocaOp>(def);
}

// 保守别名判断：新鲜分配与预先存在的值、或两个不同新鲜分配都不别名
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

// 两个 affine 边界是否等价（affine map 相同且操作数逐个相同）
static bool sameAffineBound(affine::AffineBound a, affine::AffineBound b) {
  return a.getMap() == b.getMap() &&
         llvm::equal(a.getOperands(), b.getOperands());
}

// v 是否定义在循环外（循环 region 之外）
static bool isDefinedOutsideAffineLoop(Value v, affine::AffineForOp loop) {
  Region *region = &loop->getRegion(0);
  if (auto blockArg = dyn_cast<BlockArgument>(v))
    return !region->isAncestor(blockArg.getOwner()->getParent());
  Operation *def = v.getDefiningOp();
  return def && !region->isAncestor(def->getParentRegion());
}

// ==================== 1) 循环融合 ====================
struct AffineFusionPattern : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 0 || !llvm::hasSingleElement(op.getRegion()))
      return failure();
    auto next = dyn_cast_or_null<affine::AffineForOp>(op->getNextNode());
    if (!next || next.getNumResults() != 0 ||
        !llvm::hasSingleElement(next.getRegion()))
      return failure();
    if (!sameAffineBound(op.getLowerBound(), next.getLowerBound()) ||
        !sameAffineBound(op.getUpperBound(), next.getUpperBound()) ||
        op.getStepAsInt() != next.getStepAsInt())
      return failure();

    // 任一循环的写入不得与另一循环的读/写别名
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
    rewriter.inlineBlockBefore(next.getBody(), dest, dest->end(),
                               ValueRange{op.getInductionVar()});
    rewriter.eraseOp(next);
    return success();
  }
};

// ==================== 2) 循环分块 ====================
// 对每棵完美嵌套 affine.for 树的根循环调用 affine 的 tilePerfectlyNested。
static void tileAffineNests(ModuleOp module, uint64_t tileSize) {
  SmallVector<affine::AffineForOp, 16> roots;
  module.walk([&](affine::AffineForOp loop) {
    if (!isa<affine::AffineForOp>(loop->getParentOp()))
      roots.push_back(loop);
  });
  for (affine::AffineForOp root : roots) {
    SmallVector<affine::AffineForOp, 4> band;
    affine::getPerfectlyNestedLoops(band, root);
    if (band.empty())
      continue;
    // 只对 step 为 1 的循环做分块（向量化后的循环 step 已是向量宽度，
    // 再分块只会产生退化的单次迭代内层循环）。
    if (!llvm::all_of(band, [](affine::AffineForOp loop) {
          return loop.getStepAsInt() == 1;
        }))
      continue;
    SmallVector<unsigned, 4> tileSizes(band.size(),
                                       static_cast<unsigned>(tileSize));
    SmallVector<affine::AffineForOp, 8> tiled;
    (void)affine::tilePerfectlyNested(band, tileSizes, &tiled);
  }
}

// ==================== 3) 向量化 ====================
struct AffineVectorizePattern : public OpRewritePattern<affine::AffineForOp> {
  AffineVectorizePattern(MLIRContext *ctx, int64_t vectorWidth)
      : OpRewritePattern<affine::AffineForOp>(ctx), vectorWidth(vectorWidth) {}

  LogicalResult matchAndRewrite(affine::AffineForOp loop,
                                PatternRewriter &rewriter) const override {
    if (loop.getNumResults() != 0 || !llvm::hasSingleElement(loop.getRegion()))
      return failure();
    if (!loop.hasConstantBounds() || loop.getStepAsInt() != 1)
      return failure();
    int64_t lb = loop.getConstantLowerBound();
    int64_t ub = loop.getConstantUpperBound();
    int64_t tripCount = ub - lb;
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
          if (!isDefinedOutsideAffineLoop(other, loop))
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
          if (!isDefinedOutsideAffineLoop(other, loop))
            return failure();
        if (!elemTy)
          elemTy = store.getMemRefType().getElementType();
        else if (store.getMemRefType().getElementType() != elemTy)
          return failure();
        continue;
      }
      // 其余必须是纯标量算子，且结果类型就是元素类型
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

    // padding 零值放在循环外
    Value padding;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(loop);
      padding = arith::ConstantOp::create(rewriter, loc,
                                          rewriter.getZeroAttr(elemTy));
    }

    // 新循环：lb/ub 不变，step 改为 vector-width
    auto newLoop = affine::AffineForOp::create(rewriter, loc, lb, ub, vectorWidth);
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
      // 纯标量算子：克隆后把元素类型的结果改成向量类型
      Operation *cloned = rewriter.clone(op, mapping);
      for (OpResult res : cloned->getResults())
        if (res.getType() == elemTy)
          res.setType(vecTy);
    }

    rewriter.eraseOp(loop);
    return success();
  }

  int64_t vectorWidth;
};

// ==================== Pass ====================
struct AffineOptimizePass
    : public PassWrapper<AffineOptimizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineOptimizePass)

  AffineOptimizePass() = default;
  AffineOptimizePass(const AffineOptimizePass &other) : PassWrapper(other) {}

  Option<bool> enableFusion{
      *this, "fusion",
      llvm::cl::desc("Fuse adjacent affine.for loops with identical bounds"),
      llvm::cl::init(true)};
  Option<bool> enableTiling{
      *this, "tiling",
      llvm::cl::desc("Tile perfectly nested affine.for loop bands"),
      llvm::cl::init(true)};
  Option<bool> enableVectorize{
      *this, "vectorize",
      llvm::cl::desc("SIMD-vectorize the innermost elementwise affine.for"),
      llvm::cl::init(true)};
  Option<int64_t> tileSize{
      *this, "tile-size", llvm::cl::desc("Tiling factor for affine loop bands"),
      llvm::cl::init(4)};
  Option<int64_t> vectorWidth{
      *this, "vector-width",
      llvm::cl::desc("Vectorization width (number of elements)"),
      llvm::cl::init(4)};

  StringRef getArgument() const final { return "mydialect-affine-opt"; }
  StringRef getDescription() const final {
    return "Loop fusion / tiling / vectorization on affine.for loops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, vector::VectorDialect>();
  }

  void runFusion() {
    RewritePatternSet patterns(&getContext());
    patterns.add<AffineFusionPattern>(&getContext());
    GreedyRewriteConfig config;
    config.setMaxIterations(16);
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
  }

  void runVectorize() {
    RewritePatternSet patterns(&getContext());
    patterns.add<AffineVectorizePattern>(&getContext(), vectorWidth.getValue());
    GreedyRewriteConfig config;
    config.setMaxIterations(16);
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
  }

  void runOnOperation() override {
    if (tileSize <= 0 || vectorWidth <= 0) {
      signalPassFailure();
      return;
    }
    if (enableFusion)
      runFusion();
    // 向量化要求 trip count 为编译期常量，分块会引入动态内层边界，
    // 所以先向量化再分块。
    if (enableVectorize)
      runVectorize();
    if (enableTiling)
      tileAffineNests(getOperation(), static_cast<uint64_t>(tileSize.getValue()));
  }
};

static PassRegistration<AffineOptimizePass> regAffineOptimize;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createAffineOptimizePass() {
  return std::make_unique<AffineOptimizePass>();
}
} // namespace mydialect
