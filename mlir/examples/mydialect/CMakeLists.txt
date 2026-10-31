
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

#include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS mydialect.td)
set(ADDITIONAL_TABLEGEN_DEPENDS TensorOps.td MyDialectHwOps.td)
mlir_tablegen(mydialect.h.inc -gen-dialect-decls -dialect=mydialect DEPENDS ${ADDITIONAL_TABLEGEN_DEPENDS})
mlir_tablegen(mydialect.cpp.inc -gen-dialect-defs -dialect=mydialect DEPENDS ${ADDITIONAL_TABLEGEN_DEPENDS})
mlir_tablegen(mydialect-opdefs.h.inc -gen-op-decls DEPENDS ${ADDITIONAL_TABLEGEN_DEPENDS})
mlir_tablegen(mydialect-opdefs.cpp.inc -gen-op-defs DEPENDS ${ADDITIONAL_TABLEGEN_DEPENDS})
mlir_tablegen(mydialect.typedefs.h.inc -gen-typedef-decls --typedefs-dialect=mydialect DEPENDS ${ADDITIONAL_TABLEGEN_DEPENDS})
mlir_tablegen(mydialect.typedefs.cpp.inc -gen-typedef-defs --typedefs-dialect=mydialect DEPENDS ${ADDITIONAL_TABLEGEN_DEPENDS})
add_public_tablegen_target(MyDialectTableGen)

# ======================= ONNX stub dialect TableGen ========================
# 生成 onnx.h.inc / onnx.cpp.inc / onnx-opdefs.h.inc / onnx-opdefs.cpp.inc
# 用于 ONNX → MyDialect 转换演示，使用 MLIR builtin tensor 类型，无外部依赖
set(LLVM_TARGET_DEFINITIONS ONNXOps.td)
mlir_tablegen(onnx.h.inc -gen-dialect-decls -dialect=onnx)
mlir_tablegen(onnx.cpp.inc -gen-dialect-defs -dialect=onnx)
mlir_tablegen(onnx-opdefs.h.inc -gen-op-decls)
mlir_tablegen(onnx-opdefs.cpp.inc -gen-op-defs)
add_public_tablegen_target(OnnxStubTableGen)
set_property(TARGET OnnxStubTableGen PROPERTY FOLDER "MLIR/Tablegen")

unset(LLVM_TARGET_DEFINITIONS)
set(LLVM_TARGET_DEFINITIONS MyDialectPasses.td)
mlir_tablegen(MyDialectPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRMyDialectPassesIncGen)

unset(LLVM_TARGET_DEFINITIONS)
set(LLVM_TARGET_DEFINITIONS MyOptPass.td)
mlir_tablegen(MyOptPass.h.inc -gen-pass-decls)
add_public_tablegen_target(MyOptPassTableGen)

unset(LLVM_TARGET_DEFINITIONS)
set(LLVM_TARGET_DEFINITIONS MyDialectHwOps.td)
mlir_tablegen(MyDialectHwOps.h.inc -gen-op-decls)
mlir_tablegen(MyDialectHwOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MyDialectHwOpsIncGen)

# Dialect库，MyDialect + pass实现
add_library(MyDialectLib
  MyDialect.cpp
  pass.cpp
  LowerMyDialectPrint.cpp
  MyOptPass.cpp
  FusionPass.cpp
  ONNXToMyDialect.cpp
  mydialect-ops.cpp
  SimpleCParser.cpp
  LowerHwMulPass.cpp
  OpCountCycleEstimate.cpp
  ConcurrentJit.cpp
)
add_dependencies(MyDialectLib MyDialectTableGen OnnxStubTableGen MLIRMyDialectPassesIncGen MyOptPassTableGen MyDialectHwOpsIncGen)
llvm_process_sources(MyDialectLib PARTIAL_SOURCES_INTENDED)
target_compile_options(MyDialectLib PRIVATE -fno-rtti)

target_link_libraries(MyDialectLib
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRDialect
    MLIRLLVMDialect
    MLIRPass
    MLIRTransforms
)


# 独立驱动工具 mydriver，仅 main.cpp
add_executable(mydriver
  main.cpp
)
llvm_process_sources(mydriver PARTIAL_SOURCES_INTENDED)
target_compile_options(mydriver PRIVATE -fno-rtti)
target_link_libraries(mydriver
  PRIVATE
    MyDialectLib
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRDialect
    MLIRExecutionEngine
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRPass
    MLIRConvertToLLVMPass
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRMemRefToLLVM
    MLIRUBToLLVM
    MLIRTensorDialect
    MLIRComplexToLLVM
    MLIRBuiltinToLLVMIRTranslation
    LLVMX86AsmParser
)

if (MLIR_INCLUDE_TESTS)
  add_subdirectory(test)
endif()


