import lit.formats
import os

config.name = "MyDialect"
config.test_format = lit.formats.ShTest(True)

BUILD_ROOT = "/data1/windy01/mlir-test/llvm-project-22.1.8.src/build_mlir"

# 占位符，习惯上用大写，和LLVM项目风格对齐
config.substitutions.append(("%mlir-opt", os.path.join(BUILD_ROOT, "bin/mlir-opt")))
config.substitutions.append(("%FileCheck", os.path.join(BUILD_ROOT, "bin/FileCheck")))

config.suffixes = [".mlir"]
# 开启额外输出，方便调试
config.test_source_root = os.path.dirname(__file__)

