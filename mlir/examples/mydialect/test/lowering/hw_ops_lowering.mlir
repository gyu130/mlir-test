// RUN: %mlir-opt --lower-hw-mul %s | %FileCheck --check-prefix=SW %s
// RUN: %mlir-opt --lower-hw-mul="use-hardware-path=true" %s | %FileCheck --check-prefix=HW %s

// 测试 mydialect hw_* 硬件算子族的两条 lowering 路径：
//   标量族 (hw_add/hw_div/hw_and/hw_or/hw_xor)：
//     SW → arith.addi/divsi/andi/ori/xori   HW → llvm.call_intrinsic
//   hw_matmul：
//     SW → scf.for 三层循环软件模拟          HW → func.call mydialect.hw.matmul
//   hw_load/hw_store (DMA)：
//     SW → memref.load / memref.store        HW → func.call mydialect.hw.dma.*

module {
  func.func @scalar_ops(%a : i32, %b : i32) -> i32 {
    %1 = mydialect.hw_add %a, %b : i32
    %2 = mydialect.hw_div %1, %a : i32
    %3 = mydialect.hw_and %2, %b : i32
    %4 = mydialect.hw_or %3, %a : i32
    %5 = mydialect.hw_xor %4, %b : i32
    %6 = mydialect.hw_mul %5, %a : i32
    return %6 : i32
  }

  func.func @matmul(%A : memref<2x3xi32>, %B : memref<3x4xi32>, %C : memref<2x4xi32>) {
    mydialect.hw_matmul %A, %B, %C : memref<2x3xi32>, memref<3x4xi32>, memref<2x4xi32>
    return
  }

  func.func @dma(%buf : memref<2x3xi32>, %i : i32, %j : i32) -> i32 {
    %v = mydialect.hw_load %buf, %i, %j : memref<2x3xi32>, i32
    mydialect.hw_store %v, %buf, %i, %j : i32, memref<2x3xi32>
    return %v : i32
  }
}

// ===== SW 路径：标量族 → arith =====
// SW-LABEL: func.func @scalar_ops
// SW-NOT: mydialect.hw_
// SW: arith.addi
// SW: arith.divsi
// SW-NOT: arith.divui
// SW: arith.andi
// SW: arith.ori
// SW: arith.xori
// SW: arith.muli

// ===== SW 路径：hw_matmul → 清零循环 + K 内层累加循环 =====
// SW-LABEL: func.func @matmul
// SW-NOT: mydialect.hw_matmul
// 先清零 C（两层循环）
// SW: scf.for
// SW-NESTED: scf.for
// SW-NESTED: memref.store %{{.*}}, %arg2
// 再三层累加：i/j 外层，K 最内层
// SW: scf.for
// SW-NESTED: scf.for
// SW-NESTED-NESTED: scf.for
// SW-NESTED-NESTED: memref.load %arg0
// SW-NESTED-NESTED: memref.load %arg1
// SW-NESTED-NESTED: arith.muli
// SW-NESTED-NESTED: memref.load %arg2
// SW-NESTED-NESTED: arith.addi
// SW-NESTED-NESTED: memref.store

// ===== SW 路径：DMA → memref.load/store（索引 index_cast）=====
// SW-LABEL: func.func @dma
// SW-NOT: mydialect.hw_
// SW: arith.index_cast
// SW: memref.load
// SW: arith.index_cast
// SW: memref.store

// ===== HW 路径：模块级 private 声明（插在模块开头，故检查在最前）=====
// HW: func.func private @mydialect.hw.matmul(memref<2x3xi32>, memref<3x4xi32>, memref<2x4xi32>)
// HW: func.func private @mydialect.hw.dma.load(memref<2x3xi32>, i32, i32) -> i32
// HW: func.func private @mydialect.hw.dma.store(i32, memref<2x3xi32>, i32, i32)

// ===== HW 路径：标量族 → llvm.call_intrinsic =====
// HW-LABEL: func.func @scalar_ops
// HW-NOT: mydialect.hw_
// HW: llvm.call_intrinsic "llvm.mydialect.hw.add.i32"
// HW: llvm.call_intrinsic "llvm.mydialect.hw.div.i32"
// HW: llvm.call_intrinsic "llvm.mydialect.hw.and.i32"
// HW: llvm.call_intrinsic "llvm.mydialect.hw.or.i32"
// HW: llvm.call_intrinsic "llvm.mydialect.hw.xor.i32"
// HW: llvm.call_intrinsic "llvm.mydialect.hw.mul.i32"

// ===== HW 路径：hw_matmul → func.call =====
// HW-LABEL: func.func @matmul
// HW-NOT: mydialect.hw_matmul
// HW: call @mydialect.hw.matmul

// ===== HW 路径：DMA → func.call =====
// HW-LABEL: func.func @dma
// HW-NOT: mydialect.hw_load
// HW: call @mydialect.hw.dma.load
// HW: call @mydialect.hw.dma.store
