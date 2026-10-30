# MyDialect 技术方案与实现文档

> 基于 LLVM/MLIR 22.1.8，自定义算子方言（Dialect）端到端编译与执行框架

---

## 目录

1. [项目概述](#1-项目概述)
2. [总体架构](#2-总体架构)
3. [方言定义与类型系统](#3-方言定义与类型系统)
4. [算子（Operation）集合](#4-算子operation集合)
5. [Pass 与转换流水线](#5-pass-与转换流水线)
6. [ONNX → MyDialect 转换](#6-onnx--mydialect-转换)
7. [硬件加速算子 hw_mul 双路径 lowering](#7-硬件加速算子-hw_mul-双路径-lowering)
8. [C 前端解析器（SimpleCParser）](#8-c-前端解析器simplecparser)
9. [并发 JIT 执行引擎](#9-并发-jit-执行引擎)
10. [分析 Pass：算子计数与周期估算](#10-分析-pass算子计数与周期估算)
11. [用例设计](#11-用例设计)
12. [使用到的 MLIR / LLVM 功能清单](#12-使用到的-mlir--llvm-功能清单)
13. [编译与运行](#13-编译与运行)
14. [关键技术决策与经验总结](#14-关键技术决策与经验总结)

---

## 1. 项目概述

MyDialect 是一个在 LLVM/MLIR 框架上从零构建的自定义方言（Dialect）项目，目标是演示并实践 MLIR 全栈开发能力，涵盖：

- **方言定义**：使用 TableGen（ODS）定义自定义类型、操作、属性
- **类型系统**：自定义 `wrapped_i32` 包装类型与 `MyTensorType` 张量类型
- **算子 IR**：标量算术、内存操作、循环、张量算子、硬件加速算子
- **编译流水线**：高层优化 → 常量折叠 → 方言降级（lowering）→ LLVM IR → JIT 执行
- **前端接入**：简化 C 语言解析器、ONNX 模型 IR 转换
- **执行后端**：单线程 JIT 与多线程并发 JIT
- **硬件建模**：硬件乘法算子的软件/硬件双 lowering 路径、执行周期估算

项目位于 `mlir/examples/mydialect/`，最终产物为 `mydriver` 可执行文件与可被 `mlir-opt` 加载的 Pass 库。

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         输入层                                    │
│   ① C 源码 (test_hw.c)   ② MLIR 文本 (.mlir)   ③ ONNX IR        │
└───────┬───────────────────────┬─────────────────────┬───────────┘
        │ SimpleCParser         │ parseSourceFile     │ ONNX stub dialect
        ▼                       ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MyDialect IR (高层)                            │
│   mydialect.addi/muli/subi/divi_s/cmpi  (标量算术)                │
│   mydialect.alloc/load/store         (内存)                       │
│   mydialect.for + yield             (循环，带 iter_args)          │
│   mydialect.add_tensor/mul_tensor/... (张量算子)                  │
│   mydialect.loop + loop_yield        (张量循环，对标 ONNX Loop)   │
│   mydialect.hw_mul                  (硬件加速算子)                │
│   mydialect.print / print_str       (运行时输出)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  Pass 流水线
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  高层优化 Pass                                                    │
│   MyOptPass: 冗余 print 删除 + arith.addi CSE                     │
│   ConstantFoldMyDialectPass: 常量折叠/循环求值/展开/不变式外提      │
│   LowerHwMulPass: hw_mul → arith.muli (SW) 或 llvm intrinsic (HW) │
│   OpCountCycleEstimatePass: 算子计数与周期估算                     │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  LowerMyDialectPrint Pass (核心降级)                              │
│   自定义 TypeConverter:                                            │
│     wrapped_i32 → i32                                             │
│     MyTensorType → memref<...>                                    │
│   算子降级:                                                        │
│     addi/muli/... → arith.*   for → scf.for   loop → scf.while   │
│     alloc/load/store → memref.*   print → llvm.call               │
│     constant_tensor → memref.global + get_global                  │
│     add_tensor/mul_tensor → memref + scf.for 逐元素                │
│     tensor_to_scalar/scalar_to_tensor → memref.load/store         │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  标准降级 Pass                                                     │
│   Canonicalizer → SCFToControlFlow → ConvertToLLVM               │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLVM Dialect → LLVM IR → ExecutionEngine (JIT)                   │
│   注册运行时符号: my_runtime_print_i32, my_runtime_print_string    │
│   invokePacked 执行入口函数                                        │
└─────────────────────────────────────────────────────────────────┘
```

**三条输入通路**：

| 通路 | 入口 | 说明 |
|------|------|------|
| C 前端 | `--parse-c <file.c>` | SimpleCParser 解析简化 C，生成 MyDialect IR 输出到 stdout |
| MLIR 文本 | `--parse-mlir <file.mlir>` 或位置参数 | 直接解析 MLIR，走完整 lowering + JIT |
| ONNX IR | MLIR 文本内含 `onnx.*` op | 先 `--convert-onnx-to-mydialect` 转换，再走 lowering |

---

## 3. 方言定义与类型系统

### 3.1 方言声明

方言在 `mydialect.td` 中通过 TableGen 定义：

```tablegen
def MyDialect : Dialect {
  let name = "mydialect";
  let cppNamespace = "::mydialect";
  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
  }];
}
```

`extraClassDeclaration` 声明自定义类型的 `parseType`/`printType` 钩子，在 `MyDialect.cpp` 中实现。

### 3.2 自定义类型

| 类型 | mnemonic | 参数 | 用途 |
|------|----------|------|------|
| `WrappedI32Type` | `wrapped_i32` | 无 | 包装 i32 的演示类型，降级时擦除为 i32 |
| `MyTensorType` | `tensor` | `shape: ArrayAttr`, `elementType: Type` | 自定义张量类型，对标 builtin tensor，降级为 memref |

**MyTensorType 的汇编格式**（`hasCustomAssemblyFormat = 1`，手写 parse/print）：

```
!mydialect.tensor<[2,3], i32>
```

- mnemonic `tensor` 由 `Dialect::printType` 输出，类型体 `<[dims], elemTy>` 由 `MyTensorType::print/parse` 处理
- `shape` 的 `ArrayAttr` 元素全局统一为 **index 类型** 的 `IntegerAttr`，确保解析类型与推导类型相等

**类型降级规则**（`MyDialectTypeConverter`，`LowerMyDialectPrint.cpp`）：

```
wrapped_i32  ──→  i32
MyTensorType<[dims], elemTy>  ──→  memref<dims x elemTy>
其余类型        ──→  identity（保持原样）
```

`addConversion` 是 LIFO 顺序，因此 identity 兜底必须先注册，具体规则后注册才生效。同时注册 `addSourceMaterialization` / `addTargetMaterialization`，用 `unrealized_conversion_cast` 占位，转换结束后由驱动自动折叠。

---

## 4. 算子（Operation）集合

### 4.1 标量算术算子

| 算子 | 语义 | 特性 |
|------|------|------|
| `mydialect.addi` | i32 加法 | `hasFolder`, `hasCanonicalizer` |
| `mydialect.muli` | i32 乘法 | `hasFolder`, `hasCanonicalizer` |
| `mydialect.subi` | i32 减法 | `hasFolder`, `hasCanonicalizer` |
| `mydialect.divi_s` | 有符号除法 | `hasVerifier`（除零检查）, `hasFolder`, `hasCanonicalizer` |
| `mydialect.cmpi` | 整数比较，结果 i1 | `hasVerifier`（predicate 校验） |

**fold 接口**：`AddI32Op::fold` / `MuliOp::fold` 等在操作数均为常量时直接返回计算结果属性。

**canonicalize 模式**：
- `x + 0 → x`，`0 + x → x`
- `x - 0 → x`
- `x * 1 → x`，`x * 0 → 0`
- `x / 1 → x`

### 4.2 内存算子

| 算子 | 语义 | 降级目标 |
|------|------|----------|
| `mydialect.alloc %size : memref<?xi32>` | 一维数组分配 | `memref.alloca`（size 经 `arith.index_cast` 转 index） |
| `mydialect.load %buf, %idx : memref<?xi32>, i32` | 数组读 | `memref.load` |
| `mydialect.store %val, %buf, %idx` | 数组写 | `memref.store` |

`AllocOp`/`LoadOp`/`StoreOp` 均有 verifier，限制为 1 维 i32 memref。

### 4.3 循环算子

**`mydialect.for`**（标量循环，带 iter_args）：

```
mydialect.for %iv = %lb to %ub step %step iter_args(%arg0 = %init0) -> (i32) : i32 {
  ...
  mydialect.yield %newVal : i32
}
```

- region 块参数：`%iv`（i32）+ iter_args
- 终止符 `mydialect.yield`（`ParentOneOf<["ForOp"]>`）
- 降级目标：`scf.for`（iv 经 `arith.index_cast` 在 i32/index 间转换）

**`mydialect.loop` + `mydialect.loop_yield`**（张量循环，对标 ONNX Loop）：

```
mydialect.loop %maxTrip, %cond, %initArgs : (!mydialect.tensor<...>) -> (!mydialect.tensor<...>) {
  ^bb0(%iter, %cond_in, %carry...):
    ...
    mydialect.loop_yield %newCond, %newCarry... : i1, !mydialect.tensor<...>
}
```

- 设计要素：最大迭代次数 + 初始条件 + 迭代携带张量 + 循环体 + 终止符
- `SingleBlock` trait（不能用 `SingleBlockImplicitTerminator`，因 loop_yield 的 cond 是必需操作数）
- verifier 校验：carry 数量与 initArgs/results 一致，类型一致
- 降级目标：`scf.while`（携带布局 `(i: i64, cond: i1, carries: memref...)`）

### 4.4 张量算子（TensorOps.td）

| 算子 | 语义 | 类型推导 | 降级目标 |
|------|------|----------|----------|
| `add_tensor` | 逐元素加 | `SameOperandsAndResultType` | memref + scf.for |
| `mul_tensor` | 逐元素乘 | `SameOperandsAndResultType` | memref + scf.for |
| `cmpi_tensor` | 逐元素比较（结果 i1 张量） | 手写 `inferReturnTypes` | （ONNX 用） |
| `constant_tensor` | 张量常量 | 手写 `inferReturnTypes` | `memref.global` + `get_global` |
| `identity_tensor` | 恒等 | `SameOperandsAndResultType` | （透传） |
| `reshape_tensor` | 形状重排 | 无（运行时 shape） | （ONNX 用） |
| `transpose_tensor` | 转置 | 手写 `inferReturnTypes` | （ONNX 用） |
| `reduce_sum_tensor` | 归约求和 | 手写 `inferReturnTypes` | （ONNX 用） |

**关键设计**：
- `SameOperandsAndResultType` trait 自动提供 `InferTypeOpInterface` 默认实现
- 结果类型与输入不同的算子（如 `cmpi_tensor` 输出 i1 张量）需手写 `inferReturnTypes`
- `reshape_tensor` 的 shape 是运行时张量，无法静态推导，故不实现 `InferTypeOpInterface`

### 4.5 标量↔张量桥接算子

| 算子 | 语义 | 降级目标 |
|------|------|----------|
| `scalar_to_tensor` | i32 → 0 维 MyTensorType | `memref.alloc` + `memref.store` |
| `tensor_to_scalar` | 0 维 MyTensorType → i32 | `memref.load` |

SimpleCParser 全程用 MyTensorType 做标量载体，而 `hw_mul` 是纯标量 op，这两个 op 桥接两个世界。

### 4.6 输出算子

| 算子 | 语义 | 降级目标 |
|------|------|----------|
| `mydialect.print %val : type` | 打印整数/张量 | `llvm.call @my_runtime_print_i32`（张量逐元素打印，生成嵌套 scf.for） |
| `mydialect.print_str` | 打印字符串 | `llvm.call @my_runtime_print_string` |

### 4.7 硬件加速算子（MyDialectHwOps.td）

| 算子 | 语义 | 降级路径 |
|------|------|----------|
| `mydialect.hw_mul %a, %b : i32` | 硬件 i32 乘法 | SW: `arith.muli`；HW: `llvm.call_intrinsic "llvm.mydialect.hw.mul.i32"` |

---

## 5. Pass 与转换流水线

### 5.1 Pass 清单

| Pass | 命令行参数 | 功能 |
|------|-----------|------|
| `MyOptPass` | `--my-opt` | 冗余 print 删除 + arith.addi 块内 CSE |
| `ConstantFoldMyDialectPass` | `--constant-fold-mydialect` | 算术常量折叠、循环常量求值/展开、循环不变式外提 |
| `LowerHwMulPass` | `--lower-hw-mul` | hw_mul 双路径 lowering（`--use-hardware-path`） |
| `OpCountCycleEstimatePass` | `--mydialect-count-op-cycle-est` | 算子计数 + 硬件周期估算 |
| `LowerMyDialectPrintPass` | `--lower-mydialect-print` | 核心降级：mydialect → scf/memref/arith/llvm |
| `ConvertOnnxToMyDialectPass` | `--convert-onnx-to-mydialect` | ONNX stub → mydialect |

所有 Pass 通过 TableGen `MyDialectPasses.td` / `MyOptPass.td` 声明，并在各自 TU 内静态注册（`static bool _registered = []() { registerXxxPass(); return true; }();`），确保 `mlir-opt` 启动时自动可用。

### 5.2 LowerMyDialectPrint Pass（核心降级）

这是项目最核心的 Pass，完成 mydialect 到标准方言的全面降级。

**ConversionTarget 配置**：
- 所有 mydialect op 标记为 `addIllegalOp`
- `func`/`llvm`/`arith`/`scf`/`memref` 方言标记为合法
- `UnrealizedConversionCastOp` 标记合法（pattern 内手动插入的 cast）
- `func.func` / `func.return` 动态合法性：通过 `typeConverter.isSignatureLegal` 判断，配合 `populateFunctionOpInterfaceTypeConversionPattern` 完成签名重写

**算子降级 Pattern 一览**（均为 `OpConversionPattern`）：

| Pattern | 源 op | 目标 |
|---------|-------|------|
| `WrapOpLowering` | wrap | 透传（identity） |
| `UnwrapOpLowering` | unwrap | 透传（必要时插 unrealized cast） |
| `PrintOpLowering` | print | 标量→`llvm.call`；张量→嵌套 scf.for + memref.load + llvm.call |
| `PrintStrOpLowering` | print_str | `llvm.call @my_runtime_print_string` |
| `AddI32OpLowering` 等 | addi/muli/subi/divi_s | `arith.addi` 等 |
| `CmpiOpLowering` | cmpi | `arith.cmpi`（predicate 字符串→枚举） |
| `ForOpLowering` | for | `scf.for`（inline region，块参数映射） |
| `MyAllocToMemRef` 等 | alloc/load/store | `memref.alloca/load/store` |
| `ConstantTensorOpLowering` | constant_tensor | `memref.global` + `memref.get_global` |
| `AddTensorOpLowering` | add_tensor | memref.alloc + 嵌套 scf.for 逐元素 add |
| `MulTensorOpLowering` | mul_tensor | memref.alloc + 嵌套 scf.for 逐元素 mul |
| `LoopOpLowering` | loop | `scf.while`（before/after 双 region） |
| `TensorToScalarOpLowering` | tensor_to_scalar | `memref.load` |
| `ScalarToTensorOpLowering` | scalar_to_tensor | `memref.alloc` + `memref.store` |

### 5.3 ConstantFoldMyDialectPass

包含 6 个 RewritePattern：

1. `FoldMuliPattern` / `FoldAddiPattern`：两操作数均为常量时直接计算
2. `ConstantFoldMyDialectPattern`：print op 的常量识别（调试用）
3. `ConstantEvaluateForPattern`（benefit=2）：常量边界 + 常量 iter_args + 纯算术循环体 → 解释执行算出结果常量，迭代上限 MAX_EVAL_ITER=16
4. `ConstantUnrollForPattern`（benefit=1）：常量边界循环展开，迭代上限 MAX_UNROLL=16
5. `LoopInvariantHoistPattern`：循环不变式外提

### 5.4 完整驱动流水线（main.cpp）

```
MyOptPass
  → ConstantFoldMyDialectPass
  → LowerHwMulPass          (必须在 LowerMyDialectPrint 之前)
  → OpCountCycleEstimatePass
  → LowerMyDialectPrintPass
  → Canonicalizer
  → SCFToControlFlow
  → ConvertToLLVM
  → ExecutionEngine JIT
```

---

## 6. ONNX → MyDialect 转换

### 6.1 ONNX stub 方言

项目内置一个最小 ONNX 方言（`ONNXOps.td`），使用 MLIR builtin tensor 类型，不依赖外部 onnx-mlir：

| ONNX op | 功能 |
|---------|------|
| `onnx.Add` / `onnx.Mul` | 逐元素加/乘 |
| `onnx.Equal` | 逐元素相等比较 |
| `onnx.Constant` | 张量常量（自定义 assembler，匹配 onnx-mlir 输出格式） |
| `onnx.Identity` | 恒等 |
| `onnx.Reshape` | 形状重排 |
| `onnx.Transpose` | 转置 |
| `onnx.ReduceSum` | 归约求和 |
| `onnx.Loop` + `onnx.Yield` | 循环（对标 ONNX Loop） |
| `onnx.EntryPoint` | 入口标记（`FlatSymbolRefAttr:$func`） |

### 6.2 转换策略

**TypeConverter**：`RankedTensorType` → `MyTensorType`（shape 转为 index IntegerAttr 数组），其余 identity。

**转换分 Phase 执行**（`ConvertOnnxToMyDialectPass::runOnOperation`）：

1. **Phase 1**：`ConvertONNXLoop`（RewritePattern）处理 ONNX Loop，提取 maxTripCount/cond 标量，inline body，转换 yield
2. **Phase 2a**：其余 ONNX op → mydialect op（全部用 `OpRewritePattern` 而非 `OpConversionPattern`，避免 adaptor 自动类型转换导致反向 materialization 残留）
3. **Phase 2b**：手动 walk `LoopOp` body，将 block arg 类型从 tensor `setType` 为 MyTensorType
4. **Phase 2c**：手动 walk `func.func` 签名 + `func.return`，转换类型
5. **Phase 3**：手动消除残留的 `unrealized_conversion_cast`（identity cast 替换 + 套娃 cast 折叠），最多迭代 10 次

**关键决策**：ONNX 转换优先使用 `OpRewritePattern` 而非 `OpConversionPattern`，因为后者的 adaptor 会强制对 operands 做 TypeConverter 转换，当 carry 输入是 func block arg（尚未签名转换）时会导致 adaptor 构造失败。

---

## 7. 硬件加速算子 hw_mul 双路径 lowering

`mydialect.hw_mul` 是硬件乘法原语，支持两条独立 lowering 路径，由 `LowerHwMulPass` 的 `useHardwarePath` 选项控制：

### 7.1 软件路径（默认）

```
mydialect.hw_mul %a, %b : i32   →   arith.muli %a, %b : i32
```

`HwMulToArithMuli` pattern（`OpRewritePattern`）直接替换为 `arith.muli`，后续走标准 arith→LLVM 链路，在 CPU 上 JIT 执行。

### 7.2 硬件路径（`--use-hardware-path=true`）

```
mydialect.hw_mul %a, %b : i32   →   llvm.call_intrinsic "llvm.mydialect.hw.mul.i32"(%a, %b) : i32
```

`HwMulToLLVMIntrinsic` pattern 生成 `llvm.call_intrinsic`，对应 RISC-V 自定义指令或 ISS 仿真。

### 7.3 流水线位置

`LowerHwMulPass` 必须在 `LowerMyDialectPrintPass` **之前**执行：hw_mul 先 lower 成 arith.muli / llvm.intrinsic，然后 LowerMyDialectPrint 处理 SimpleCParser 生成的 tensor_to_scalar / scalar_to_tensor 桥接 op。

---

## 8. C 前端解析器（SimpleCParser）

### 8.1 支持的语法子集

- 函数：`int main() { ... }`
- 变量声明：`int a = expr;`
- 表达式：`+ - * /`，数字，变量，括号，`hw_mul(a, b)`
- 语句：`print(expr);`，`return expr;`，表达式语句

### 8.2 IR 生成策略

全程使用 `MyTensorType`（0 维张量）作为标量载体：
- 数字常量 → `mydialect.constant_tensor`（`DenseElementsAttr` + 0 维 MyTensorType）
- 二元运算 → `mydialect.add_tensor` / `mydialect.mul_tensor`
- `hw_mul(a, b)` → `tensor_to_scalar` → `hw_mul` → `scalar_to_tensor`（桥接标量与张量世界）
- `print` → `mydialect.print`
- `return` → `func.return`

函数返回类型设为 void，末尾显式添加 `func.return` 作为终止符。

---

## 9. 并发 JIT 执行引擎

### 9.1 单模块 JIT（`runSingleJitModule`）

1. `ExecutionEngine::create(module, opts)` 创建引擎
2. `registerSymbols` 回调注册外部符号：`print`、`my_runtime_print_i32`
3. `invokePacked("main", argRef)` 执行入口函数
4. 支持 `jitOptLevel` 参数控制 LLVM 优化级别（`llvm::CodeGenOpt::getLevel`）

### 9.2 多模块并发 JIT（`runConcurrentJit`）

- 输入：多个 MLIR 模块（`ArrayRef<OwningOpRef<ModuleOp>>`）
- 按 `maxThreadNum` 分批，每批创建 `std::thread` 并发执行
- 每个线程 clone 一份模块，独立创建 ExecutionEngine（每个 engine 自带独立 LLVMContext，线程安全）
- 用 `std::promise`/`std::future` 收集结果
- **关键实现细节**：
  - `promises.reserve(batchSize)` 必须先 reserve，否则 emplace_back 重分配搬走 promise，lambda 捕获的指针悬空导致堆损坏
  - `moduleList.push_back(std::move(modOrErr))` 必须移动整个 `OwningOpRef`，不能只 move `*modOrRef`（后者在析构时 erase op 导致悬空）
  - 共享 MLIRContext 需 `enableMultithreading()`，且 DialectRegistry 必须注册 `registerLLVMDialectTranslation` + `registerBuiltinDialectTranslation`

### 9.3 入口函数签名捕获

`captureSignature` 在 lowering 前从原始 `func.func` 捕获参数/结果类型描述（`ArgDesc`），lowering 后按 memref descriptor 展开约定构造 `invokePacked` 的 `void*` 参数数组：

- 标量：1 个槽位
- memref：`(allocated, aligned, offset, sizes..., strides...)` 共 `3 + 2*rank` 个槽位

---

## 10. 分析 Pass：算子计数与周期估算

`OpCountCycleEstimatePass` 遍历 IR 统计 `addi`/`muli` 数量，基于硬件资源模型估算执行周期：

```
pipeline_cycles(N, fu, latency) = ceil(N / fu) + latency - 1
total_cycles = max(add_cycle, mul_cycle)
```

可配置参数：`--add-latency`、`--mul-latency`、`--add-fu`、`--mul-fu`。

统计覆盖 `mydialect.addi`、`arith.addi`、`mydialect.muli`、`arith.muli`、`mydialect.hw_mul`（hw_mul 在流水线中先 lower 成 arith.muli，需同时统计）。

---

## 11. 用例设计

### 11.1 C 前端用例

**`test_hw.c`** — 硬件乘法端到端：

```c
int main() {
  int a = 10;
  int b = 20;
  int c = hw_mul(a, b);
  print(c);
}
```

运行：`mydriver --parse-c test_hw.c > test_hw.mlir && mydriver test_hw.mlir`

预期输出：`[my-runtime-print] value = 200`

### 11.2 硬件双路径 lowering 用例

**`test/lowering/hw_mul_lowering.mlir`**：

```mlir
func.func @hw_test(%a : i32, %b : i32) -> i32 {
  %c = mydialect.hw_mul %a, %b : i32
  return %c : i32
}
```

- SW 路径：`mlir-opt --lower-hw-mul` → 检查出现 `arith.muli`
- HW 路径：`mlir-opt --lower-hw-mul="use-hardware-path=true"` → 检查出现 `llvm.call_intrinsic "llvm.mydialect.hw.mul.i32"`

### 11.3 ONNX Loop 转换用例

**`test/onnx/loop.mlir`**：

```mlir
func.func @test_loop(%arg0: tensor<2x3xi32>, %init: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %m = onnx.Constant dense<10> : tensor<i64>
  %c = onnx.Constant dense<true> : tensor<i1>
  %0:1 = "onnx.Loop"(%m, %c, %init) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>, %carry: tensor<2x3xi32>):
      %1 = "onnx.Add"(%carry, %arg0) : ...
      "onnx.Yield"(%cond_in, %1) : ...
  }) : ...
  return %0 : tensor<2x3xi32>
}
```

运行：`mlir-opt --convert-onnx-to-mydialect` → 检查 `mydialect.loop` / `mydialect.loop_yield` 出现，`onnx.` 消失。

### 11.4 标量循环与常量折叠用例

**`loop.mlir`** / `loop-iter.mlir`：测试 `mydialect.for` 的 iter_args、常量循环求值与展开。

**`fold.mlir`**：测试 addi/muli 的常量折叠与 canonicalize（x+0→x, x*1→x 等）。

### 11.5 内存操作用例

**`mybuf.mlir`**：测试 `mydialect.alloc`/`load`/`store` 降级到 memref。

### 11.6 张量算子用例

**`tensor.mlir`** / `tensor-ops.mlir`：测试 `add_tensor`/`mul_tensor`/`constant_tensor` 的降级（嵌套 scf.for 逐元素计算）。

### 11.7 并发 JIT 用例

```bash
mydriver --concurrent-jit --jit-mlir-files=a.mlir,b.mlir,c.mlir --jit-threads=4 --jit-opt=2
```

多个模块并发 JIT 执行，每个模块独立 ExecutionEngine。

### 11.8 高层优化用例

**`test/myopt/redundant_print.mlir`**：连续相同 print 删除。
**`test/myopt/addi_cse.mlir`**：块内相同 addi 公共子表达式消除。

### 11.9 周期估算用例

**`test/analysis/op_count_cycle.mlir`**：含 addi/muli 的模块，运行 `--mydialect-count-op-cycle-est` 输出周期报告。

---

## 12. 使用到的 MLIR / LLVM 功能清单

### 12.1 MLIR 方言定义（ODS / TableGen）

| 功能 | 用途 |
|------|------|
| `Dialect` def | 声明 mydialect / onnx 方言 |
| `TypeDef` | 定义 `WrappedI32Type`、`MyTensorType` 自定义类型 |
| `Op<Dialect, mnemonic>` | 定义所有操作 |
| `arguments` / `results` / `regions` | 声明操作数、结果、region |
| `assemblyFormat` | 声明式汇编格式 |
| `hasCustomAssemblyFormat = 1` | 手写 parse/print（PrintOp、ForOp、ONNXConstantOp、MyTensorType） |
| `hasVerifier = 1` | 手写 verify 方法 |
| `hasFolder = 1` | 实现 fold 接口 |
| `hasCanonicalizer = 1` | 注册 canonicalization pattern |
| Traits: `Pure`, `Terminator`, `ParentOneOf`, `SingleBlock`, `SameOperandsAndResultType`, `SameTypeOperands` | 操作语义约束 |
| `DeclareOpInterfaceMethods<InferTypeOpInterface>` | 声明类型推导接口 |
| `AttrParameter` / `TypeParameter` | 类型参数声明 |
| `OptionalAttr` / `DefaultValuedAttr` | 可选/默认属性 |
| `Pass` / `Option` | Pass 声明与选项 |

### 12.2 MLIR IR 与类型系统

| 功能 | 用途 |
|------|------|
| `MLIRContext` / `DialectRegistry` | 上下文与方言注册 |
| `OpBuilder` / `ImplicitLocOpBuilder` | IR 构造 |
| `Operation` / `Value` / `Block` / `Region` | IR 核心抽象 |
| `IntegerType` / `IndexType` / `FunctionType` | 内置类型 |
| `MemRefType` / `RankedTensorType` | 结构化类型 |
| `ArrayAttr` / `IntegerAttr` / `ElementsAttr` / `DenseElementsAttr` / `FlatSymbolRefAttr` | 属性系统 |
| `TypeConverter` + `addConversion` / `addSourceMaterialization` / `addTargetMaterialization` | 类型转换框架 |
| `UnrealizedConversionCastOp` | 类型擦除占位 |
| `SymbolTable` | 符号查重（memref.global 命名） |

### 12.3 MLIR 模式匹配与重写

| 功能 | 用途 |
|------|------|
| `OpRewritePattern` / `OpConversionPattern` | 重写模式基类 |
| `PatternRewriter` / `ConversionPatternRewriter` | 重写器 |
| `RewritePatternSet` + `applyPatternsGreedily` / `applyPatternsAndFoldGreedily` | 贪婪重写驱动 |
| `GreedyRewriteConfig` | 重写配置（迭代次数、遍历顺序） |
| `matchPattern` + `m_ConstantInt` | 常量匹配 |
| `IRMapping` + `clone` | 循环展开时克隆操作 |
| `inlineBlockBefore` / `inlineRegionBefore` | region/块内联 |
| `convertRegionTypes` | region 块参数类型转换 |

### 12.4 MLIR 方言转换框架

| 功能 | 用途 |
|------|------|
| `ConversionTarget` + `addIllegalOp` / `addLegalDialect` / `addDynamicallyLegalOp` | 合法性判定 |
| `applyPartialConversion` | 部分转换 |
| `populateFunctionOpInterfaceTypeConversionPattern` / `populateReturnOpTypeConversionPattern` | func 签名转换 |
| `DialectConversion` 调试：`--debug-only=dialect-conversion` | 转换调试 |

### 12.5 MLIR Pass 框架

| 功能 | 用途 |
|------|------|
| `PassWrapper` / `OperationPass<ModuleOp>` | Pass 基类 |
| `PassManager` | Pass 流水线管理 |
| `PassRegistration` / 静态注册 | Pass 注册 |
| `getDependentDialects` | 声明依赖方言 |
| `signalPassFailure` | Pass 失败标记 |
| TableGen `GEN_PASS_DECL` / `GEN_PASS_DEF` / `GEN_PASS_REGISTRATION` | Pass 代码生成 |

### 12.6 标准方言

| 方言 | 用途 |
|------|------|
| `arith` | 标量算术（addi/muli/subi/divsi/cmpi/constant/index_cast） |
| `scf` | 结构化控制流（for/while/yield/condition） |
| `memref` | 内存抽象（alloc/alloca/load/store/global/get_global） |
| `func` | 函数抽象（func/return） |
| `llvm` | LLVM IR 方言（func/call/call_intrinsic/void/pointer） |
| `tensor` | 张量（extract，ONNX 转换用） |
| `cf` | 控制流（SCFToControlFlow 目标） |
| `complex` | 复数（tensor 方言级联加载，需注册 ToLLVM interface） |
| `affine` | 仿射（注册以备扩展） |

### 12.7 MLIR 到 LLVM 的翻译与 JIT

| 功能 | 用途 |
|------|------|
| `ConvertToLLVMPass` | 统一 LLVM 降级 |
| `registerConvertArithToLLVMInterface` / `FuncToLLVM` / `ControlFlowToLLVM` / `MemRefToLLVM` / `UBToLLVM` / `ComplexToLLVM` | 各方言到 LLVM 的转换接口注册 |
| `registerLLVMDialectTranslation` / `registerBuiltinDialectTranslation` | LLVM IR 翻译接口注册 |
| `SCFToControlFlowPass` | scf → cf |
| `CanonicalizerPass` | 规范化（折叠 cast 等） |
| `ExecutionEngine` | JIT 执行引擎 |
| `ExecutionEngineOptions` + `makeOptimizingTransformer` | JIT 配置 |
| `invokePacked` | 打包参数调用 |
| `registerSymbols` + `MangleAndInterner` + `ExecutorSymbolDef` | 外部符号注册 |
| `OwningOpRef<ModuleOp>` | 模块所有权管理 |
| `translateModuleToLLVMIR`（EE 内部） | MLIR → LLVM IR 翻译 |

### 12.8 LLVM 功能

| 功能 | 用途 |
|------|------|
| `llvm::cl` 命令行解析 | `opt`/`list`/`Positional`/`CommaSeparated`/`ZeroOrMore` |
| `llvm::InitializeNativeTarget` 系列 | 目标初始化 |
| `llvm::orc::LLJIT`（EE 内部） | ORC JIT |
| `llvm::CodeGenOpt::getLevel` | 优化级别转换（enum class） |
| `llvm::SmallVector` / `DenseMap` / `StringSwitch` | ADT 容器 |
| `llvm::MemoryBuffer` | 文件读取 |
| `llvm::raw_ostream` (outs/errs) | 输出 |
| `llvm::TypeSwitch` | 类型分发 |

### 12.9 C++ 并发

| 功能 | 用途 |
|------|------|
| `std::thread` / `std::promise` / `std::future` | 并发 JIT |
| `std::mutex` | 运行时 IO 互斥 |
| `std::vector::reserve` | 防止重分配导致指针悬空 |

---

## 13. 编译与运行

### 13.1 构建

项目集成在 LLVM/MLIR 构建树中，TableGen 生成以下文件：

- `mydialect.h.inc` / `mydialect.cpp.inc` — 方言定义
- `mydialect-opdefs.h.inc` / `mydialect-opdefs.cpp.inc` — 操作定义
- `mydialect.typedefs.h.inc` / `mydialect.typedefs.cpp.inc` — 类型定义
- `MyDialectPasses.h.inc` / `MyOptPass.h.inc` — Pass 定义
- `MyDialectHwOps.h.inc` / `MyDialectHwOps.cpp.inc` — 硬件操作定义
- `onnx.h.inc` / `onnx.cpp.inc` / `onnx-opdefs.*.inc` — ONNX stub 方言

产物：
- `MyDialectLib`：方言库（链接到 mlir-opt，提供所有 Pass）
- `mydriver`：独立驱动工具

### 13.2 运行示例

```bash
# 1. C 前端生成 MLIR
mydriver --parse-c test_hw.c > test_hw.mlir

# 2. 端到端 lowering + JIT 执行
mydriver test_hw.mlir

# 3. 指定入口函数和参数
mydriver loop.mlir 10 main

# 4. 硬件 intrinsic 路径
mydriver --hw-lower test_hw.mlir

# 5. 并发 JIT
mydriver --concurrent-jit --jit-mlir-files=a.mlir,b.mlir --jit-threads=4

# 6. 用 mlir-opt 单独跑 Pass
mlir-opt --lower-hw-mul test.mlir
mlir-opt --convert-onnx-to-mydialect onnx_loop.mlir
mlir-opt --lower-mydialect-print tensor.mlir
mlir-opt --mydialect-count-op-cycle-est --add-fu=2 --mul-fu=1 test.mlir
```

---

## 14. 关键技术决策与经验总结

### 14.1 类型系统设计

- **MyTensorType 的 shape 用 index IntegerAttr**：全局统一，避免解析类型与推导类型的 ArrayAttr 元素类型不一致（i64 vs index）导致 `InferTypeOpInterface` 回读报 "inferred types incompatible"
- **自定义类型的 print/parse 只处理类型体**：mnemonic 由 `Dialect::printType` 输出后委托成员 print，与 `async::ValueType` 模式一致

### 14.2 方言转换

- **TypeConverter::addConversion 是 LIFO**：identity 兜底必须先注册，具体规则后注册
- **OpConversionPattern 的 adaptor 不会自动转换 operand 类型**：未转换的上游 op 产出值仍是原类型，需 pattern 内自行处理（`ensureConverted` helper）
- **`UnrealizedConversionCastOp` 必须标记合法**：否则 pattern 内手动创建的 cast 会被合法性检查回滚
- **带 region 的 op 转换后必须 convertRegionTypes**：否则 region 内 op 操作数类型不匹配
- **scf::WhileOp 必须显式传 before/after 回调**：否则 create 生成空 region，getBeforeBody() 解引用崩溃

### 14.3 ONNX 转换

- **优先 OpRewritePattern 而非 OpConversionPattern**：避免 adaptor 自动类型转换导致反向 materialization 残留与 cast 套娃
- **Loop 转换需立即创建新 op 并 inline region**：RewritePattern 延迟执行机制会导致 cast 套娃、操作丢失

### 14.4 汇编格式

- **assemblyFormat 必须包含 attr-dict**：否则触发解析错误
- **ODS 不支持内联 verifier 代码块**：需用 `let hasVerifier = 1` 声明
- **`SingleBlockImplicitTerminator` 要求终止符支持零操作数默认构造**：若终止符有必需操作数则改用 `SingleBlock` + 显式 verifier 校验

### 14.5 并发 JIT

- **每个 ExecutionEngine 自带独立 LLVMContext 是安全的**：共享 MLIRContext 需 enableMultithreading
- **必须注册 DialectTranslation 接口**：否则 translateModuleToLLVMIR 返回 null
- **promise 必须先 reserve 再取地址**：vector 重分配会搬走 promise 导致悬空
- **OwningOpRef 必须整体移动**：不能只 move `*ref`（operator* 按值返回句柄）

### 14.6 运行时与符号

- **ExecutionEngine::registerSymbols 用回调形式**：`SymbolMap(MangleAndInterner)`，符号名须与 IR 引用一致
- **LLVM 22 的 CodeGenOptLevel 是 enum class**：用 `llvm::CodeGenOpt::getLevel(int)` 转换

---

*文档基于项目源码（mlir/examples/mydialect/）整理，覆盖方言定义、类型系统、算子集合、Pass 流水线、ONNX 转换、硬件 lowering、C 前端、并发 JIT、周期估算等全部模块。*
