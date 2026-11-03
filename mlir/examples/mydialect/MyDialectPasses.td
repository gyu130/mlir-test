#ifndef MYDIALECT_PASSES_TD
#define MYDIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerMyDialectPrint : Pass<"lower-mydialect-print", "mlir::ModuleOp"> {
  let summary = "Lower mydialect.print to llvm.call";
  let description = "Lower mydialect.print op into LLVM call to runtime print function";
  let options = [
    Option<"disableRuntimeDecl", "disable-runtime-decl", "bool",
           /*default=*/"false", "Do not emit runtime function declaration">
  ];
}

// ============ LowerHwMulPass：mydialect hw_* 算子族 → SW / HW 双路径 ============
def LowerHwMulPass : Pass<"lower-hw-mul", "mlir::ModuleOp"> {
  let summary = "Lower mydialect hw_* ops to arith/memref (SW) or LLVM/external calls (HW)";
  let description = [{
    Lower the mydialect hardware operator family into one of two paths.
    标量族（hw_mul/hw_add/hw_div/hw_and/hw_or/hw_xor）:
      1. Software path (default): → arith.muli/addi/divsi/andi/ori/xori
      2. Hardware path (use-hardware-path=true): → llvm.call_intrinsic
         "llvm.mydialect.hw.<op>.i32"（RISC-V 自定义指令 / ISS 仿真）
    hw_matmul:
      1. Software path: scf.for 三层循环 + muli/addi 软件模拟
      2. Hardware path: func.call 外部符号 mydialect.hw.matmul
    hw_load/hw_store（DMA）:
      1. Software path: memref.load / memref.store
      2. Hardware path: func.call mydialect.hw.dma.load / mydialect.hw.dma.store
  }];
  let options = [
    Option<"useHardwarePath", "use-hardware-path", "bool",
           /*default=*/"false", "Use LLVM intrinsic path instead of arith.muli">
  ];
}

// ============ OpCountCycleEstimatePass：统计 addi/muli 并估算执行周期 ============
def OpCountCycleEstimatePass : Pass<"mydialect-count-op-cycle-est", "mlir::ModuleOp"> {
  let summary = "Count addi/muli and estimate execution cycles with a HW resource model";
  let description = [{
    Walk the IR and count addi/muli operations, then estimate execution cycles
    with a simple hardware resource model: N operations of one kind are issued
    onto `fu` pipelined functional units with the given latency, costing
    ceil(N / fu) + latency - 1 cycles. The add and mul pipelines run
    concurrently, so the total estimate is the maximum of both pipelines.
  }];
  let options = [
    Option<"addLatency", "add-latency", "int",
           /*default=*/"1", "Latency of addi in cycles">,
    Option<"mulLatency", "mul-latency", "int",
           /*default=*/"4", "Latency of muli in cycles">,
    Option<"addFU", "add-fu", "int",
           /*default=*/"2", "Number of add functional units">,
    Option<"mulFU", "mul-fu", "int",
           /*default=*/"1", "Number of mul functional units">
  ];
}

#endif

