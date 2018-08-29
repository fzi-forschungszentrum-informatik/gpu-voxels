# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# ======================================
# CMakeLists file to demonstrate how to use GPU Voxels as a collision checker in OMPL:
# ======================================

cmake_minimum_required (VERSION 2.8)
SET(CMAKE_CXX_STANDARD 11)

project (gvl_ompl_planning)

# First we have to find our dependencies:
FIND_PACKAGE(icl_core REQUIRED )
FIND_PACKAGE(gpu_voxels REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(ompl REQUIRED)


# This is a quirk and should be removed in upcoming versions
# If you built GPU Voxels without ROS support, remove this.
FIND_PACKAGE(orocos_kdl REQUIRED)

# A little debug info:
MESSAGE(STATUS "GVL include dirs are: ${gpu_voxels_INCLUDE_DIRS}")
MESSAGE(STATUS "OMPL include dirs are: ${OMPL_INCLUDE_DIRS}")

# Also we have to inherit some Environment definitions required for our base libs:
add_definitions(
  ${icl_core_DEFINITIONS}
  ${gpu_voxels_DEFINITIONS}
)


# Create a library that uses GPU Voxels:
add_library (gvl_ompl_planner_helper gvl_ompl_planner_helper.cpp)

target_include_directories (gvl_ompl_planner_helper
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${gpu_voxels_INCLUDE_DIRS}
    PUBLIC ${icl_core_INCLUDE_DIRS}
    PUBLIC ${CUDA_INCLUDE_DIRS}
    PUBLIC ${OMPL_INCLUDE_DIRS}
    PUBLIC ${orocos_kdl_INCLUDE_DIRS} # this should be removed in upcoming versions.
)

# Add an executable that calls the lib:
add_executable (gvl_ompl_planner gvl_ompl_planner.cpp)

# Link the executable to the library.
# We currently also have to link against Boost and icl_core...
target_link_libraries (gvl_ompl_planner
    LINK_PUBLIC gvl_ompl_planner_helper
    LINK_PUBLIC ${Boost_SYSTEM_LIBRARY}
    LINK_PUBLIC ${icl_core_LIBRARIES}
    LINK_PUBLIC ${gpu_voxels_LIBRARIES}
    LINK_PUBLIC ${OMPL_LIBRARIES}
    LINK_PUBLIC ${CUDA_LIBRARIES}
)

