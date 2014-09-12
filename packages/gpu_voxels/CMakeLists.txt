# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# --- To be used by other modules ---
SET(GPU_VOXELS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE INTERNAL "")
SET(GPU_VOXELS_IDL_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")
SET(GPU_VOXELS_IDE_FOLDER "gpu_voxels")

# --- Global options ---

FIND_PACKAGE(CUDA)

# Enable position independent code
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# require GPU CUDA capabilities >= 2.0
SET(ICMAKER_CUDA_CPPDEFINES -arch=sm_20)


# --- Package management ---
ICMAKER_REGISTER_PACKAGE(gpu_voxels)

ADD_SUBDIRECTORY(src/gpu_voxels)
ADD_SUBDIRECTORY(src/gpu_visualization)

ICMAKER_CONFIGURE_PACKAGE()

# --- Additional stuff ---
ADD_SUBDIRECTORY(src/examples)

IF(BUILD_TESTS)
ENDIF(BUILD_TESTS)

