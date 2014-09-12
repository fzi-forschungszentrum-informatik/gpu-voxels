# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# To be used by other modules
SET(ICL_CORE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE INTERNAL "")
SET(ICL_CORE_IDL_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")
SET(ICL_CORE_IDE_FOLDER "icl_core")

ICMAKER_REGISTER_PACKAGE(icl_core)

ADD_SUBDIRECTORY (src/icl_core)
ADD_SUBDIRECTORY (src/icl_core_config)
ADD_SUBDIRECTORY (src/icl_core_logging)
ADD_SUBDIRECTORY (src/icl_core_plugin)
ADD_SUBDIRECTORY (src/icl_core_thread)
ADD_SUBDIRECTORY (src/icl_core_testsuite)
ADD_SUBDIRECTORY (src/icl_core_dispatch)
ADD_SUBDIRECTORY (src/icl_core_crypt)

ADD_SUBDIRECTORY (src/uls)

ICMAKER_CONFIGURE_PACKAGE()

IF (BUILD_TESTS)
  ADD_SUBDIRECTORY (src/test/test_icl_core_config)
  ADD_SUBDIRECTORY (src/test/test_icl_core_logging)
  ADD_SUBDIRECTORY (src/ts/icl_core)
  ADD_SUBDIRECTORY (src/ts/icl_core_config)
  ADD_SUBDIRECTORY (src/ts/icl_core_thread)
  ADD_SUBDIRECTORY (src/ts/icl_core_crypt)
ENDIF (BUILD_TESTS)

