# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# ======================================
# CMakeLists file to demonstrate how to use GPU Voxels in your own project:
# ======================================

cmake_minimum_required (VERSION 2.8)
project (gvl_linkage_test)

# First we have to find our dependencies:
FIND_PACKAGE(icl_core REQUIRED )
FIND_PACKAGE(gpu_voxels REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system REQUIRED)

# Also we have to set some Environment definitions required for our base libs:
IF(WIN32)
  add_definitions("-D_SYSTEM_WIN32_")
ENDIF(WIN32)

IF(UNIX)
  IF(NOT APPLE)
    add_definitions("-D_SYSTEM_LINUX_" "-D_SYSTEM_POSIX_")
  ELSE()
    add_definitions("-D_SYSTEM_DARWIN_" "-D_SYSTEM_POSIX_")
  ENDIF(NOT APPLE)
ENDIF(UNIX)


# This is a quirk and should be removed in upcoming versions
# If you built GPU Voxels without ROS support, remove this.
FIND_PACKAGE(orocos_kdl REQUIRED)

# A little debug info:
MESSAGE(STATUS "GVL include dirs are: ${gpu_voxels_INCLUDE_DIRS}")


# Create a library that uses GPU Voxels:
add_library (gvl_linkage_test_lib gvl_linkage_test_lib.cpp)

target_include_directories (gvl_linkage_test_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${gpu_voxels_INCLUDE_DIRS}
    PUBLIC ${orocos_kdl_INCLUDE_DIRS} # this should be removed in upcoming versions.
)

# Add an executable that calls the lib:
add_executable (gvl_linkage_test gvl_linkage_test.cpp)

# Link the executable to the library.
# We currently also have to link against Boost and icl_core...
target_link_libraries (gvl_linkage_test
    LINK_PUBLIC gvl_linkage_test_lib
    LINK_PUBLIC ${Boost_SYSTEM_LIBRARY}
    LINK_PUBLIC ${icl_core_LIBRARIES}
    LINK_PUBLIC ${gpu_voxels_LIBRARIES}

)
