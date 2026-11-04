// LowerDevicePass.cpp：--mydialect-lower-device
// 异构部署边界落地：
//   target=cpu    : 内联 mydialect.device_exec region，退回 host 执行
//   target=device : 保留 host/device 边界
//   target=gpu    : 保留边界，把 region 内 hw_matmul 替换为手写 gpu.launch
//                   kernel（不接 nvvm/rocdl 降级，用户按需接 MLIR GPU 后端）
#include "MyOptPass.h"
#include "MyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
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

// GPU 路径 dispatcher：递归 device_exec region，替换 hw_matmul，其他保留
static void gpuLowerDevice(Region &region) {
  for (Block &block : region) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (auto matmul = dyn_cast<HwMatmulOp>(&op)) {
        OpBuilder b(&op);
        lowerHwMatmulToGpuKernel(b, matmul);
        continue;
      }
      // 其他 HW 算子（hw_add/hw_mul/hw_div/hw_and/hw_or/hw_xor/hw_load/hw_store）
      // 单标量粒度不值得 GPU kernel，保持原样交给 HW 路径（extern call）。
      // 张量算子 GPU kernel 化需要先把 MyTensorType 转 memref，由
      // PlaceDevice + LowerMyDialectPrint + LowerDevice 协同实现，留作后续迭代。
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
