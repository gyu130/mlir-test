// RUN: %mlir-opt --lower-mydialect-print --mydialect-tile %s | %FileCheck %s --check-prefix=DEFAULT
// RUN: %mlir-opt --lower-mydialect-print "--mydialect-tile=tile-size=4" %s | %FileCheck %s --check-prefix=TILE4
// 上界 <= tile-size 时不分块；显式指定更小 tile-size 时分块生效
module {
  func.func @main(%a : !mydialect.tensor<[8],i32>,
                  %b : !mydialect.tensor<[8],i32>) -> !mydialect.tensor<[8],i32> {
    %0 = mydialect.add_tensor %a, %b : <[8], i32>
    return %0 : !mydialect.tensor<[8], i32>
  }
}

// DEFAULT-LABEL: func.func @main
// DEFAULT: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// DEFAULT-NOT: affine.min
// DEFAULT: return %alloc

// TILE4-LABEL: func.func @main
// TILE4: scf.for %[[IO:.*]] = %c0 to %c8 step %c4 {
// TILE4-NEXT: affine.min #map(%[[IO]])[%c8]
// TILE4-NEXT: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 {
// TILE4: return %alloc
