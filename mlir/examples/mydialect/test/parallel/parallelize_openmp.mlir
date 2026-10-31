// RUN: %mlir-opt --lower-mydialect-print --mydialect-parallelize --scf-forall-to-parallel --convert-scf-to-openmp %s | %FileCheck %s
// OpenMP 后端：scf.forall → scf.parallel → omp.parallel / omp.wsloop / omp.loop_nest
module {
  func.func @main(%a : !mydialect.tensor<[64],i32>,
                  %b : !mydialect.tensor<[64],i32>) -> !mydialect.tensor<[64],i32> {
    %0 = mydialect.add_tensor %a, %b : <[64], i32>
    return %0 : !mydialect.tensor<[64], i32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: omp.parallel {
// CHECK-NEXT: omp.wsloop {
// CHECK: omp.loop_nest
// CHECK: memref.store
// CHECK: omp.yield
// CHECK: omp.terminator
