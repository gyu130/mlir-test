set(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TEST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")

configure_lit_site_cfg(
  ${TEST_SOURCE_DIR}/lit.cfg.py
  ${TEST_BUILD_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${TEST_SOURCE_DIR}/lit.cfg.py
  SITE_CONFIG ${TEST_BUILD_DIR}/lit.site.cfg.py
)

add_lit_testsuite(check-mydialect
  "Run MyDialect regression tests"
  ${TEST_SOURCE_DIR}
  DEPENDS mlir-opt MyDialectLib mydriver FileCheck
)
set_target_properties(check-mydialect PROPERTIES FOLDER "Tests")

