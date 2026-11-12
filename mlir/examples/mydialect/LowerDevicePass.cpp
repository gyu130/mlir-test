// LowerDevicePass.cpp：--mydialect-lower-device
// 异构部署边界落地：
//   target=cpu    : 内联 mydialect.device_exec region，退回 host 执行
//   target=device : 保留 host/device 边界
//   target=gpu    : 保留边界，把 region 内 hw_matmul 与逐元素张量算子
//                   （add/mul/sub/div/relu/identity/cmpi）替换为手写
//                   gpu.launch kernel（不接 nvvm/rocdl 降级，用户按需接
//                   MLIR GPU 后端）
#include "MyOptPass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;

namespace mydialect {
namespace {

static void inlineDevice(DeviceExecOp op) {
  Block &body = op.getBody().front();
  auto yield = cast<DeviceYieldOp>(body.getTerminator());
  for (unsigned i = 0; i < op.getNumResults(); ++i)
    op.getResult(i).replaceAllUsesWith(yield.getOperand(i));
  yield->erase();
  op->getBlock()->getOperations().splice(Block::iterator(op),
                                         body.getOperations());
  op->erase();
}

static Value getMemDim(OpBuilder &b, Location loc, Value memref, int rank,
                       int axis) {
  auto ty = cast<MemRefType>(memref.getType());
  int64_t s = ty.getDimSize(axis);
  if (!ShapedType::isDynamic(s))
    return arith::ConstantIndexOp::create(b, loc, s);
  auto cidx = arith::ConstantIndexOp::create(b, loc, axis);
  return memref::DimOp::create(b, loc, memref, cidx);
}

static Value ceilDiv(OpBuilder &b, Location loc, Value a, Value bVal) {
  return arith::CeilDivSIOp::create(b, loc, a, bVal);
}

// MyTensorType → 等 shape 的 memref（rank、静态维度保持一致）
static Type tensorTypeToMemrefType(Type ty) {
  auto tt = dyn_cast<MyTensorType>(ty);
  if (!tt)
    return ty;
  SmallVector<int64_t> dims;
  for (Attribute a : tt.getShape())
    dims.push_back(cast<IntegerAttr>(a).getInt());
  return MemRefType::get(dims, tt.getElementType());
}

// flat index → 多维下标：从最内维开始 idx[d] = cur % size[d]; cur /= size[d]
static void emitMultiDimIndices(OpBuilder &b, Location loc, Value flatIdx,
                                ArrayRef<int64_t> sizes,
                                SmallVectorImpl<Value> &indices) {
  indices.resize(sizes.size());
  Value cur = flatIdx;
  for (int d = static_cast<int>(sizes.size()) - 1; d >= 0; --d) {
    Value sz = arith::ConstantIndexOp::create(b, loc, sizes[d]);
    indices[d] = arith::RemUIOp::create(b, loc, cur, sz);
    cur = arith::DivUIOp::create(b, loc, cur, sz);
  }
}

enum class ElemBinaryKind { Add, Mul, Sub, Div };

static Value emitBinaryArith(OpBuilder &b, Location loc, ElemBinaryKind kind,
                             Value lhs, Value rhs) {
  Type elemTy = lhs.getType();
  if (isa<FloatType>(elemTy)) {
    switch (kind) {
    case ElemBinaryKind::Add:
      return arith::AddFOp::create(b, loc, lhs, rhs);
    case ElemBinaryKind::Mul:
      return arith::MulFOp::create(b, loc, lhs, rhs);
    case ElemBinaryKind::Sub:
      return arith::SubFOp::create(b, loc, lhs, rhs);
    case ElemBinaryKind::Div:
      return arith::DivFOp::create(b, loc, lhs, rhs);
    }
  }
  switch (kind) {
  case ElemBinaryKind::Add:
    return arith::AddIOp::create(b, loc, lhs, rhs);
  case ElemBinaryKind::Mul:
    return arith::MulIOp::create(b, loc, lhs, rhs);
  case ElemBinaryKind::Sub:
    return arith::SubIOp::create(b, loc, lhs, rhs);
  case ElemBinaryKind::Div:
    return arith::DivSIOp::create(b, loc, lhs, rhs);
  }
  llvm_unreachable("unknown elementwise binary kind");
}

// 逐元素算子 → 手写 gpu.launch kernel：
//   grid = (ceil(N/256), 1, 1), block = (256, 1, 1)
//   thread 按 flat index 覆盖整个张量，反解成多维下标后 load → compute → store
// 返回 false 表示无法处理（动态 shape 等），调用方保留原 op。
static bool lowerElementwiseToGpuKernel(
    Operation *op, ArrayRef<Value> ins,
    llvm::function_ref<Value(OpBuilder &, Location, ArrayRef<Value>)> compute) {
  Location loc = op->getLoc();
  Type resTy = op->getResult(0).getType();
  auto resMemTy = dyn_cast<MemRefType>(tensorTypeToMemrefType(resTy));
  if (!resMemTy || !resMemTy.hasStaticShape())
    return false;
  for (Value v : ins) {
    auto memTy = dyn_cast<MemRefType>(tensorTypeToMemrefType(v.getType()));
    if (!memTy || !memTy.hasStaticShape())
      return false;
  }

  ArrayRef<int64_t> dims = resMemTy.getShape();
  int64_t n = 1;
  for (int64_t d : dims)
    n *= d;

  OpBuilder b(op);
  // host 侧：张量操作数 → memref（type converter 收尾时会把 identity cast 折叠掉）
  SmallVector<Value> memrefs;
  for (Value v : ins)
    memrefs.push_back(
        UnrealizedConversionCastOp::create(
            b, loc, tensorTypeToMemrefType(v.getType()), v)
            .getResult(0));
  // 静态 shape：不需要 dynamicSize 操作数
  Value resMem =
      memref::AllocOp::create(b, loc, resMemTy, ValueRange{}).getResult();

  Value c256 = arith::ConstantIndexOp::create(b, loc, 256);
  Value cN = arith::ConstantIndexOp::create(b, loc, n);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value gridX = ceilDiv(b, loc, cN, c256);

  auto launch = gpu::LaunchOp::create(b, loc, gridX, c1, c1, c256, c1, c1);
  Block &body = launch.getBody().front();
  b.setInsertionPointToStart(&body);

  Value bidX = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value tidX = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value flat = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, bidX, c256), tidX);
  Value inRange = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                        flat, cN);
  auto ifOp = scf::IfOp::create(b, loc, inRange, /*withElse=*/false);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());

  SmallVector<Value> idx;
  emitMultiDimIndices(b, loc, flat, dims, idx);
  SmallVector<Value> loaded;
  for (Value m : memrefs)
    loaded.push_back(memref::LoadOp::create(b, loc, m, idx));
  Value res = compute(b, loc, loaded);
  memref::StoreOp::create(b, loc, res, resMem, idx);

  b.setInsertionPointToEnd(&body);
  gpu::TerminatorOp::create(b, loc);

  // 结果 cast 必须显式定位到 gpu.launch 之后：此时 builder 仍停在 kernel body
  // 内，直接 create 会把 cast 插到 gpu.terminator 之后导致 op 非法
  b.setInsertionPoint(op);
  Value newRes =
      UnrealizedConversionCastOp::create(b, loc, resTy, resMem).getResult(0);
  op->getResult(0).replaceAllUsesWith(newRes);
  op->erase();
  return true;
}

// 逐元素算子分发（add/mul/sub/div 二元；relu/identity 一元；cmpi 比较）
static bool lowerElementwiseTensorOpToGpuKernel(Operation *op) {
  if (auto add = dyn_cast<AddTensorOp>(op)) {
    Value operands[2] = {add.getLhs(), add.getRhs()};
    return lowerElementwiseToGpuKernel(
        op, operands, [](OpBuilder &b, Location loc, ArrayRef<Value> in) {
          return emitBinaryArith(b, loc, ElemBinaryKind::Add, in[0], in[1]);
        });
  }
  if (auto mul = dyn_cast<MulTensorOp>(op)) {
    Value operands[2] = {mul.getLhs(), mul.getRhs()};
    return lowerElementwiseToGpuKernel(
        op, operands, [](OpBuilder &b, Location loc, ArrayRef<Value> in) {
          return emitBinaryArith(b, loc, ElemBinaryKind::Mul, in[0], in[1]);
        });
  }
  if (auto sub = dyn_cast<SubTensorOp>(op)) {
    Value operands[2] = {sub.getLhs(), sub.getRhs()};
    return lowerElementwiseToGpuKernel(
        op, operands, [](OpBuilder &b, Location loc, ArrayRef<Value> in) {
          return emitBinaryArith(b, loc, ElemBinaryKind::Sub, in[0], in[1]);
        });
  }
  if (auto div = dyn_cast<DivTensorOp>(op)) {
    Value operands[2] = {div.getLhs(), div.getRhs()};
    return lowerElementwiseToGpuKernel(
        op, operands, [](OpBuilder &b, Location loc, ArrayRef<Value> in) {
          return emitBinaryArith(b, loc, ElemBinaryKind::Div, in[0], in[1]);
        });
  }
  if (auto relu = dyn_cast<ReluTensorOp>(op)) {
    Value operands[1] = {relu.getX()};
    return lowerElementwiseToGpuKernel(
        op, operands, [](OpBuilder &b, Location loc, ArrayRef<Value> in) -> Value {
          Value v = in[0];
          Type elemTy = v.getType();
          // relu(x) = x < 0 ? 0 : x
          if (isa<FloatType>(elemTy)) {
            Value zero =
                arith::ConstantOp::create(b, loc, b.getFloatAttr(elemTy, 0.0));
            Value lt = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                             v, zero);
            return arith::SelectOp::create(b, loc, lt, zero, v);
          }
          Value zero =
              arith::ConstantOp::create(b, loc, b.getIntegerAttr(elemTy, 0));
          Value lt = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, v,
                                           zero);
          return arith::SelectOp::create(b, loc, lt, zero, v);
        });
  }
  if (auto identity = dyn_cast<IdentityTensorOp>(op)) {
    Value operands[1] = {identity.getIn()};
    return lowerElementwiseToGpuKernel(
        op, operands,
        [](OpBuilder &, Location, ArrayRef<Value> in) { return in[0]; });
  }
  if (auto cmpi = dyn_cast<CmpITensorOp>(op)) {
    int64_t predVal = cmpi.getPredicate();
    Value operands[2] = {cmpi.getLhs(), cmpi.getRhs()};
    return lowerElementwiseToGpuKernel(
        op, operands,
        [predVal](OpBuilder &b, Location loc, ArrayRef<Value> in) -> Value {
          auto pred = static_cast<arith::CmpIPredicate>(predVal);
          return arith::CmpIOp::create(b, loc, pred, in[0], in[1]);
        });
  }
  return false;
}

// gpu.launch body helper：hw_matmul
// grid=(ceil(M/TB), ceil(N/TB), 1), block=(TB, TB, 1)
// 每个 thread 算一个 C[i, j]，内层 K 循环累加乘积
// scf::IfOp builder 自动插 yield；scf::ForOp 有 iter_args 必须手动 yield
static void lowerHwMatmulToGpuKernel(OpBuilder &b, HwMatmulOp op) {
  Location loc = op.getLoc();
  Value A = op.getA(), B = op.getB(), C = op.getC();
  constexpr int64_t TB = 16;
  auto c16 = arith::ConstantIndexOp::create(b, loc, TB);
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto c0 = arith::ConstantIndexOp::create(b, loc, 0);

  Value M = getMemDim(b, loc, A, 2, 0);
  Value K = getMemDim(b, loc, A, 2, 1);
  Value N = getMemDim(b, loc, B, 2, 1);
  Value gridX = ceilDiv(b, loc, M, c16);
  Value gridY = ceilDiv(b, loc, N, c16);

  auto launch = gpu::LaunchOp::create(b, loc, gridX, gridY, c1,
                                      c16, c16, c1);
  Block &body = launch.getBody().front();
  b.setInsertionPointToStart(&body);

  Value bidX = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value bidY = gpu::BlockIdOp::create(b, loc, gpu::Dimension::y);
  Value tidX = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value tidY = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::y);
  Value i = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, bidX, c16), tidX);
  Value j = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, bidY, c16), tidY);

  // scf.if (i < M && j < N) { ... }
  auto iInRange = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                         i, M);
  auto jInRange = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                         j, N);
  auto inRange = arith::AndIOp::create(b, loc, iInRange, jInRange);
  auto ifOp = scf::IfOp::create(b, loc, inRange, /*withElse=*/false);

  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value i32Zero = arith::ConstantIntOp::create(b, loc, 0, 32);
  // scf.for k = 0 to K step 1 iter_args(acc = 0) { acc += A[i,k] * B[k,j]; }
  auto accLoop = scf::ForOp::create(b, loc, c0, K, c1,
                                     ValueRange{i32Zero});
  Block *accBlock = accLoop.getBody();
  b.setInsertionPointToStart(accBlock);
  Value k = accLoop.getInductionVar();
  Value acc = accLoop.getRegionIterArg(0);
  Value ai = memref::LoadOp::create(b, loc, A, ValueRange{i, k});
  Value bj = memref::LoadOp::create(b, loc, B, ValueRange{k, j});
  Value sum = arith::AddIOp::create(
      b, loc, acc, arith::MulIOp::create(b, loc, ai, bj));
  b.setInsertionPointToEnd(accBlock);
  scf::YieldOp::create(b, loc, ValueRange{sum});

  b.setInsertionPointAfter(accLoop);
  memref::StoreOp::create(b, loc, accLoop.getResult(0), C, ValueRange{i, j});

  b.setInsertionPointToEnd(&body);
  gpu::TerminatorOp::create(b, loc);
  op->erase();
}

// GPU 路径 dispatcher：递归 device_exec region，替换 hw_matmul 与逐元素
// 张量算子，其他保留
static void gpuLowerDevice(Region &region) {
  for (Block &block : region) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (auto matmul = dyn_cast<HwMatmulOp>(&op)) {
        OpBuilder b(&op);
        lowerHwMatmulToGpuKernel(b, matmul);
        continue;
      }
      // 逐元素张量算子：add/mul/sub/div/relu/identity/cmpi → gpu.launch
      if (lowerElementwiseTensorOpToGpuKernel(&op))
        continue;
      // 其他 HW 算子（hw_add/hw_mul/hw_div/hw_and/hw_or/hw_xor/hw_load/hw_store）
      // 单标量粒度不值得 GPU kernel，保持原样交给 HW 路径（extern call）。
      for (Region &nested : op.getRegions())
        gpuLowerDevice(nested);
    }
  }
}

static void lowerRegion(Region &region, bool inlineAll, bool gpuTarget,
                        unsigned &count) {
  for (Block &block : region) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (auto dev = dyn_cast<DeviceExecOp>(&op)) {
        if (inlineAll) { inlineDevice(dev); ++count; }
        else if (gpuTarget) { gpuLowerDevice(dev->getRegion(0)); ++count; }
        continue;
      }
      for (Region &nested : op.getRegions())
        lowerRegion(nested, inlineAll, gpuTarget, count);
    }
  }
}

struct LowerDevicePass
    : public PassWrapper<LowerDevicePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDevicePass)
  LowerDevicePass() = default;
  LowerDevicePass(const LowerDevicePass &other) : PassWrapper(other) {}

  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("Deployment target: cpu=inline, gpu=generate gpu.launch "
                     "kernels, otherwise keep the host/device boundary"),
      llvm::cl::init("cpu")};
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Print lowering statistics"),
                       llvm::cl::init(false)};

  StringRef getArgument() const final { return "mydialect-lower-device"; }
  StringRef getDescription() const final {
    return "Lower mydialect.device_exec boundaries according to the "
           "deployment target";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool inlineAll = target == "cpu";
    bool gpuTarget = target == "gpu";
    unsigned count = 0;
    for (Region &r : module->getRegions())
      lowerRegion(r, inlineAll, gpuTarget, count);
    if (verbose)
      llvm::errs() << "[mydialect-lower-device] target=" << target
                   << ", processed " << count << " device_exec ops\n";
  }
};

static PassRegistration<LowerDevicePass> regLowerDevice;

} // anonymous namespace

std::unique_ptr<mlir::Pass> createLowerDevicePass(std::string target,
                                                  bool verbose) {
  auto pass = std::make_unique<LowerDevicePass>();
  pass->target = std::move(target);
  pass->verbose = verbose;
  return pass;
}
} // namespace mydialect
