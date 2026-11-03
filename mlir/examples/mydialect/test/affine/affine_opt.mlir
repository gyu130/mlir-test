// RUN: %mlir-opt --mydialect-affine-opt="fusion=true tiling=false vectorize=false" %s | %FileCheck %s --check-prefix=FUSE
// RUN: %mlir-opt --mydialect-affine-opt="fusion=false tiling=true vectorize=false tile-size=4" %s | %FileCheck %s --check-prefix=TILE
// RUN: %mlir-opt --mydialect-affine-opt="fusion=false tiling=false vectorize=true vector-width=4" %s | %FileCheck %s --check-prefix=VEC
// RUN: %mlir-opt --mydialect-affine-opt="fusion=false tiling=false vectorize=true vector-width=3" %s | %FileCheck %s --check-prefix=VEC3
//
// affine.for 上的三类变换：融合 / 分块 / 向量化，各自可独立开关。

func.func @fuse(%a: memref<16xi32>, %b: memref<16xi32>) {
  %buf = memref.alloc() : memref<16xi32>
  %out = memref.alloc() : memref<16xi32>
  affine.for %i = 0 to 16 {
    %v = memref.load %a[%i] : memref<16xi32>
    memref.store %v, %buf[%i] : memref<16xi32>
  }
  affine.for %j = 0 to 16 {
    %w = memref.load %b[%j] : memref<16xi32>
    memref.store %w, %out[%j] : memref<16xi32>
  }
  return
}

func.func @tile(%a: memref<16x16xi32>, %b: memref<16x16xi32>) {
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      %v = memref.load %a[%i, %j] : memref<16x16xi32>
      memref.store %v, %b[%i, %j] : memref<16x16xi32>
    }
  }
  return
}

func.func @vec(%a: memref<8xf32>, %b: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    %v = memref.load %a[%i] : memref<8xf32>
    %w = memref.load %b[%i] : memref<8xf32>
    %s = arith.addf %v, %w : f32
    memref.store %s, %b[%i] : memref<8xf32>
  }
  return
}

// FUSE-LABEL: func.func @fuse
// FUSE: affine.for %[[IV:.*]] = 0 to 16 {
// FUSE-NEXT: %[[V:.*]] = memref.load %arg0[%[[IV]]] : memref<16xi32>
// FUSE-NEXT: memref.store %[[V]], %{{.*}}[%[[IV]]] : memref<16xi32>
// FUSE-NEXT: %[[W:.*]] = memref.load %arg1[%[[IV]]] : memref<16xi32>
// FUSE-NEXT: memref.store %[[W]], %{{.*}}[%[[IV]]] : memref<16xi32>
// FUSE-NOT: affine.for
// FUSE-LABEL: func.func @tile

// TILE-LABEL: func.func @tile
// TILE: affine.for %[[TI:.*]] = 0 to 16 step 4 {
// TILE-NEXT: affine.for %[[TJ:.*]] = 0 to 16 step 4 {
// TILE-NEXT: affine.for %[[I:.*]] = #{{.*}}(%[[TI]]) to #{{.*}}(%[[TI]]) {
// TILE-NEXT: affine.for %[[J:.*]] = #{{.*}}(%[[TJ]]) to #{{.*}}(%[[TJ]]) {
// TILE: memref.load %arg0[%[[I]], %[[J]]] : memref<16x16xi32>

// VEC-LABEL: func.func @vec
// VEC: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// VEC: affine.for %[[IV:.*]] = 0 to 8 step 4 {
// VEC-NEXT: %[[A:.*]] = vector.transfer_read %arg0[%[[IV]]], %[[PAD]] : memref<8xf32>, vector<4xf32>
// VEC-NEXT: %[[B:.*]] = vector.transfer_read %arg1[%[[IV]]], %[[PAD]] : memref<8xf32>, vector<4xf32>
// VEC-NEXT: %[[S:.*]] = arith.addf %[[A]], %[[B]] : vector<4xf32>
// VEC-NEXT: vector.transfer_write %[[S]], %arg1[%[[IV]]] : vector<4xf32>, memref<8xf32>
// VEC-NOT: memref.load

// VEC3-LABEL: func.func @vec
// VEC3-NOT: vector.transfer_read
// VEC3: affine.for
// VEC3: memref.load
