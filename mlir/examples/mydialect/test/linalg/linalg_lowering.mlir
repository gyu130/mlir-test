// RUN: %mlir-opt --lower-mydialect-print %s | %FileCheck %s
// RUN: %mlir-opt --mydialect-fuse-elementwise --lower-mydialect-print %s | %FileCheck %s --check-prefix=FUSED
//
// 逐元素算子与 matmul 的 lowering 目标为 Linalg 结构化操作（buffer 语义）：
//   add/mul/sub/div/cmpi/relu/exp → linalg.generic（全 parallel 迭代）
//   matmul_tensor                 → linalg.fill + linalg.matmul
// 不再手写循环 nest，后续由 --convert-linalg-to-loops 统一降回 scf.for。

module {
  func.func @add(%a : !mydialect.tensor<[4],i32>,
                 %b : !mydialect.tensor<[4],i32>) -> !mydialect.tensor<[4],i32> {
    %0 = mydialect.add_tensor %a, %b : !mydialect.tensor<[4],i32>
    return %0 : !mydialect.tensor<[4],i32>
  }

  func.func @mm(%A : !mydialect.tensor<[8,16],i32>,
                %B : !mydialect.tensor<[16,4],i32>) -> !mydialect.tensor<[8,4],i32> {
    %0 = mydialect.matmul_tensor %A, %B
        : !mydialect.tensor<[8,16],i32>, !mydialect.tensor<[16,4],i32> -> !mydialect.tensor<[8,4],i32>
    return %0 : !mydialect.tensor<[8,4],i32>
  }

  func.func @relu(%x : !mydialect.tensor<[4],f32>) -> !mydialect.tensor<[4],f32> {
    %0 = mydialect.relu_tensor %x : !mydialect.tensor<[4],f32>
    return %0 : !mydialect.tensor<[4],f32>
  }

  func.func @fused_chain(%a : !mydialect.tensor<[4],i32>,
                         %b : !mydialect.tensor<[4],i32>,
                         %c : !mydialect.tensor<[4],i32>) -> !mydialect.tensor<[4],i32> {
    %0 = mydialect.mul_tensor %a, %b : !mydialect.tensor<[4],i32>
    %1 = mydialect.add_tensor %0, %c : !mydialect.tensor<[4],i32>
    return %1 : !mydialect.tensor<[4],i32>
  }
}

// ----- 逐元素 add -----
// CHECK-LABEL: func.func @add
// CHECK: linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%{{.*}}, %{{.*}} : memref<4xi32>, memref<4xi32>) outs(%{{.*}} : memref<4xi32>)
// CHECK: arith.addi
// CHECK: linalg.yield
// CHECK-NOT: scf.for
// CHECK: return

// ----- matmul -----
// CHECK-LABEL: func.func @mm
// CHECK: linalg.fill ins(%{{.*}} : i32) outs(%{{.*}} : memref<8x4xi32>)
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : memref<8x16xi32>, memref<16x4xi32>) outs(%{{.*}} : memref<8x4xi32>)
// CHECK-NOT: scf.for
// CHECK: return

// ----- relu -----
// CHECK-LABEL: func.func @relu
// CHECK: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%{{.*}} : memref<4xf32>) outs(%{{.*}} : memref<4xf32>)
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: linalg.yield

// ----- 融合后：(a*b)+c 只发射一个 linalg.generic -----
// FUSED-LABEL: func.func @fused_chain
// FUSED: linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<4xi32>, memref<4xi32>, memref<4xi32>) outs(%{{.*}} : memref<4xi32>)
// FUSED: arith.muli
// FUSED: arith.addi
// FUSED: linalg.yield
// FUSED-NOT: linalg.generic
