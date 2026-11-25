// RUN: %mlir-opt --lower-mydialect-print --mydialect-linalg-opt --convert-linalg-to-loops %s | %FileCheck %s --check-prefix=DEFAULT
// RUN: %mlir-opt --lower-mydialect-print "--mydialect-linalg-opt=tile-size=4" --convert-linalg-to-loops %s | %FileCheck %s --check-prefix=TILE4
// 首维静态长度 <= tile-size 时分块无收益，pass 跳过；显式指定更小
// tile-size 时由 linalg::tileLinalgOp 分块生效（操作数换成 subview）
module {
  func.func @main(%a : !mydialect.tensor<[8],i32>,
                  %b : !mydialect.tensor<[8],i32>) -> !mydialect.tensor<[8],i32> {
    %0 = mydialect.add_tensor %a, %b : <[8], i32>
    return %0 : !mydialect.tensor<[8], i32>
  }
}

// DEFAULT-LABEL: func.func @main
// DEFAULT: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// DEFAULT-NOT: memref.subview
// DEFAULT: return %alloc

// TILE4-LABEL: func.func @main
// TILE4: scf.for %[[IO:.*]] = %c0 to %c8 step %c4 {
// TILE4-NEXT: memref.subview %arg0[%[[IO]]] [4] [1]
// TILE4: scf.for %{{.*}} = %c0 to %c4 step %c1 {
// TILE4: return %alloc
