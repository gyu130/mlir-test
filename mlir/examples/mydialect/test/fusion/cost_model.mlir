// RUN: %mlir-opt --mydialect-fuse-elementwise="fusion-max-dup=2" %s | %FileCheck %s
//
// 跨组复制代价模型：producer 被融合组外的 return 使用，复制它会把表达式
// 复制进 body（原 op 仍需保留）。cost model 判定「复制成本 < 省下的中间
// 张量一次 store+load（默认 4 拍）」是否成立：
//   - add 子树成本 1 拍 < 4 → 允许复制（mul 只在 body 外保留一份）
//   - mul 子树成本 4 拍，不 < 4 → 拒绝复制，add 也不融合（无收益）
module {
  func.func @dup_add(%a : !mydialect.tensor<[4],i32>,
                     %b : !mydialect.tensor<[4],i32>,
                     %c : !mydialect.tensor<[4],i32>)
      -> (!mydialect.tensor<[4],i32>, !mydialect.tensor<[4],i32>) {
    %0 = mydialect.add_tensor %a, %b : !mydialect.tensor<[4],i32>
    %1 = mydialect.add_tensor %0, %c : !mydialect.tensor<[4],i32>
    return %1, %0 : !mydialect.tensor<[4],i32>, !mydialect.tensor<[4],i32>
  }

  func.func @dup_mul(%a : !mydialect.tensor<[4],i32>,
                     %b : !mydialect.tensor<[4],i32>,
                     %c : !mydialect.tensor<[4],i32>)
      -> (!mydialect.tensor<[4],i32>, !mydialect.tensor<[4],i32>) {
    %0 = mydialect.mul_tensor %a, %b : !mydialect.tensor<[4],i32>
    %1 = mydialect.add_tensor %0, %c : !mydialect.tensor<[4],i32>
    return %1, %0 : !mydialect.tensor<[4],i32>, !mydialect.tensor<[4],i32>
  }
}

// add 被复制：body 内 a+b 计算一次后与 c 相加；原 add 仍为 return 保留
// CHECK-LABEL: func.func @dup_add
// CHECK: mydialect.add_tensor %arg0, %arg1
// CHECK: mydialect.fused_elementwise %arg0, %arg1, %arg2
// CHECK: %[[A:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK: %[[B:.*]] = arith.addi %[[A]], %{{.*}} : i32
// CHECK: mydialect.fused_yield %[[B]] : i32

// mul 复制成本过高被拒绝：不加融合体，mul/add 原样保留
// CHECK-LABEL: func.func @dup_mul
// CHECK: mydialect.mul_tensor %arg0, %arg1
// CHECK: mydialect.add_tensor
// CHECK-NOT: mydialect.fused_elementwise
