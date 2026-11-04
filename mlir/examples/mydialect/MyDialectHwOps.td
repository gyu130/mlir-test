#ifndef MYDIALECT_HWOPS_TD
#define MYDIALECT_HWOPS_TD

include "mlir/IR/OpBase.td"
include "mydialect.td"

def HwMulOp : Op<MyDialect, "hw_mul", [Pure]> {
  let summary = "Hardware accelerator i32 multiplication primitive";
  let description = [{
    Hardware multiplication operator for target accelerator.
    纯标量 op（i32 -> i32），硬件指令级别的乘法操作。
    两条 lowering 路径：
      1. Software path: lower to arith.muli for CPU JIT execution
      2. Hardware path: lower to custom llvm intrinsic for RISC-V custom instruction / ISS simulation
  }];

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$res);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

// ============ 硬件标量算术/逻辑算子族 ============
// 与 HwMulOp 同构：纯标量 i32×i32→i32，SW/HW 双路径。
//   hw_add  → arith.addi  / llvm.mydialect.hw.add.i32
//   hw_div  → arith.divsi / llvm.mydialect.hw.div.i32（有符号除法）
//   hw_and  → arith.andi  / llvm.mydialect.hw.and.i32（按位与）
//   hw_or   → arith.ori   / llvm.mydialect.hw.or.i32（按位或）
//   hw_xor  → arith.xori  / llvm.mydialect.hw.xor.i32（按位异或）
class HwBinaryOp<string mnemonic, string summaryText, string descText>
    : Op<MyDialect, mnemonic, [Pure]> {
  let summary = summaryText;
  let description = descText;

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$res);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

def HwAddOp : HwBinaryOp<"hw_add",
    "Hardware accelerator i32 addition primitive",
    [{
      Hardware addition operator. SW: arith.addi; HW: llvm.call_intrinsic.
    }]>;
def HwDivOp : HwBinaryOp<"hw_div",
    "Hardware accelerator i32 signed division primitive",
    [{
      Hardware signed division operator. SW: arith.divsi; HW: llvm.call_intrinsic.
    }]>;
def HwAndOp : HwBinaryOp<"hw_and",
    "Hardware accelerator i32 bitwise-and primitive",
    [{
      Hardware bitwise-and operator. SW: arith.andi; HW: llvm.call_intrinsic.
    }]>;
def HwOrOp : HwBinaryOp<"hw_or",
    "Hardware accelerator i32 bitwise-or primitive",
    [{
      Hardware bitwise-or operator. SW: arith.ori; HW: llvm.call_intrinsic.
    }]>;
def HwXorOp : HwBinaryOp<"hw_xor",
    "Hardware accelerator i32 bitwise-xor primitive",
    [{
      Hardware bitwise-xor operator. SW: arith.xori; HW: llvm.call_intrinsic.
    }]>;

// ============ 硬件矩阵单元：hw_matmul ============
// 对标硬件矩阵乘单元（MMA/Systolic Array）：C = A × B，
// A: M×K，B: K×N，C: M×N，三者均为 2-D i32 memref。
//   SW path: scf.for 三层循环 + arith.muli/arith.addi 软件模拟
//   HW path: func.call 到外部符号 mydialect.hw.matmul（设备 kernel 启动边界）
def HwMatmulOp : Op<MyDialect, "hw_matmul", []> {
  let summary = "Hardware matrix-multiply unit (MMA): C = A x B";
  let description = [{
    Hardware matrix multiply primitive targeting an on-chip matrix unit.
    A is MxK, B is KxN, C is MxN; all operands are 2-D i32 memrefs and C is
    accumulated in place (C = A*B, previous C contents overwritten).
    Two lowering paths:
      1. Software path: scf.for loop nest with arith.muli/arith.addi
      2. Hardware path: func.call to external symbol mydialect.hw.matmul
  }];

  let arguments = (ins AnyMemRef:$A, AnyMemRef:$B, AnyMemRef:$C);
  let results = (outs);

  let assemblyFormat =
      "$A `,` $B `,` $C attr-dict `:` type($A) `,` type($B) `,` type($C)";
  let hasVerifier = 1;
}

// ============ 硬件 DMA 搬运：hw_load / hw_store ============
// 硬件 DMA 引擎的读/写原语，语义与 mydialect.load/store 一致
// （索引数 == memref rank），但代表数据经 DMA 通道进出设备本地存储。
//   SW path: memref.load / memref.store（索引经 arith.index_cast 转 index）
//   HW path: func.call 到外部符号 mydialect.hw.dma.load / mydialect.hw.dma.store
def HwLoadOp : Op<MyDialect, "hw_load", []> {
  let summary = "Hardware DMA load primitive";
  let description = [{
    Load one i32 element through the hardware DMA channel.
    Index count must equal the memref rank. SW: memref.load;
    HW: func.call to external symbol mydialect.hw.dma.load.
  }];

  let arguments = (ins AnyMemRef:$memref, Variadic<I32>:$indices);
  let results = (outs I32:$res);

  let assemblyFormat =
      "$memref (`,` $indices^)? attr-dict `:` type($memref) `,` type($res)";
  let hasVerifier = 1;
}

def HwStoreOp : Op<MyDialect, "hw_store", []> {
  let summary = "Hardware DMA store primitive";
  let description = [{
    Store one i32 element through the hardware DMA channel.
    Index count must equal the memref rank. SW: memref.store;
    HW: func.call to external symbol mydialect.hw.dma.store.
  }];

  let arguments = (ins I32:$value, AnyMemRef:$memref, Variadic<I32>:$indices);
  let results = (outs);

  let assemblyFormat =
      "$value `,` $memref (`,` $indices^)? attr-dict `:` type($value) `,` type($memref)";
  let hasVerifier = 1;
}

// ============ 异构部署：host/device 执行边界 ============
// device_exec 把分配到设备（如硬件加速器）的算子包成一个 region，
// 构成 host ↔ device 的流水线边界；region 内产生的、region 外仍要用的值
// 通过 device_yield 提升为该 op 的结果。
def DeviceExecOp : Op<MyDialect, "device_exec", [SingleBlock]> {
  let summary = "Host/device execution boundary for heterogeneous placement";
  let description = [{
    Groups ops placed on a target device into a single-block region.
    Values defined inside the region and used outside are yielded by
    mydialect.device_yield and become results of this op.
    下游 pass 可按 target 选择：
      1. cpu: 内联 region（设备算子退回 host 执行）
      2. device: 保留边界，交给设备后端/kernel 生成
  }];

  let arguments = (ins StrAttr:$device);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let assemblyFormat = "$device attr-dict `:` type($results) $body";
}

def DeviceYieldOp : Op<MyDialect, "device_yield", [
  Pure,
  Terminator,
  ParentOneOf<["DeviceExecOp"]>
]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
  let assemblyFormat = "$values attr-dict `:` type($values)";
}

// ============ 分布式/异构执行原语 ============

// host↔device / device↔device 数据搬运。
// src / dst 必须是 memref；async=true 表示异步拷贝，后续需 device_wait 同步。
// device_copy 不指定源/目的设备（由 HAL 从 memref 的设备描述推断），
// 也不需要显式 buffer 分配（src/dst 由上层 device_alloc 或 host memref 给出）。
def DeviceCopyOp : Op<MyDialect, "device_copy"> {
  let summary = "Cross-device / host-device data transfer";
  let description = [{
    Copies memref content between host and device or between two devices.
    This op does not specify src/dst devices explicitly — the runtime HAL
    infers them from the backing buffer. `async=true` makes the copy
    asynchronous; the op returns before the transfer finishes, and a
    subsequent `device_wait` is required before the dst is read.
  }];
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst, OptionalAttr<BoolAttr>:$async);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `->` type($dst)";
  let hasVerifier = 1;
}

// 同步屏障：等待指定 device 或 "all" 上的所有异步 op（含 device_copy）完成。
// 在 LowerDevicePass 里会 lower 成 extern call @mydialect.hal.wait。
def DeviceWaitOp : Op<MyDialect, "device_wait"> {
  let summary = "Device synchronization barrier";
  let arguments = (ins StrAttr:$device);
  let assemblyFormat = "$device attr-dict";
}

// 设备 buffer 分配：在指定 device 上分配 memref（返回 memref）。
// 动态 shape 时 sizes 为 Variadic<i32>；静态 shape 时 sizes 可为空。
def DeviceAllocOp : Op<MyDialect, "device_alloc"> {
  let summary = "Allocate a buffer on a device";
  let arguments = (ins StrAttr:$device, Variadic<I32>:$sizes);
  let results = (outs AnyMemRef:$buffer);
  let hasVerifier = 1;
  let assemblyFormat = "$device $sizes attr-dict `:` type($buffer)";
}

#endif // MYDIALECT_HWOPS_TD

