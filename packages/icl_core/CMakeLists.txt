# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# Include icmaker if not already available.
#
# When building in an ic_workspace based on icmaker, icmaker is usually globally
# available.
#
# When using catkin (for example in ROS workspaces), icmaker is not globally
# available and thus has to be found and included in every package. Also catkin
# packages usually define a project per package.
#
#
IF(NOT DEFINED ICMAKER_DIRECTORY)
  find_package(icmaker REQUIRED)
  INCLUDE(UseFullIcMaker REQUIRED)
  project(icl_core)
  catkin_package(
    INCLUDE_DIRS src
    LIBRARIES icl_core icl_core_config icl_core_logging icl_core_thread
  )
ENDIF()


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
ADD_SUBDIRECTORY (src/icl_core_performance_monitor)

ADD_SUBDIRECTORY (src/uls)

ICMAKER_CONFIGURE_PACKAGE()

IF (BUILD_TESTS)
  ADD_SUBDIRECTORY (src/test/test_icl_core)
  ADD_SUBDIRECTORY (src/test/test_icl_core_config)
  ADD_SUBDIRECTORY (src/test/test_icl_core_logging)
  ADD_SUBDIRECTORY (src/ts/icl_core)
  ADD_SUBDIRECTORY (src/ts/icl_core_config)
  ADD_SUBDIRECTORY (src/ts/icl_core_thread)
  ADD_SUBDIRECTORY (src/ts/icl_core_crypt)
ENDIF (BUILD_TESTS)
