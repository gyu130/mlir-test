
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
mlir_tablegen(mydialect.h.inc -gen-dialect-decls)
mlir_tablegen(mydialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(mydialect-opdefs.h.inc -gen-op-decls)
mlir_tablegen(mydialect-opdefs.cpp.inc -gen-op-defs)
add_public_tablegen_target(MyDialectTableGen)

# Dialect库，MyDialect + pass实现
add_library(MyDialectLib
  MyDialect.cpp
  pass.cpp
)
add_dependencies(MyDialectLib MyDialectTableGen)
llvm_process_sources(MyDialectLib PARTIAL_SOURCES_INTENDED)
target_compile_options(MyDialectLib PRIVATE -fno-rtti)

target_link_libraries(MyDialectLib
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
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
    MLIRBuiltinToLLVMIRTranslation
    LLVMX86AsmParser
)

