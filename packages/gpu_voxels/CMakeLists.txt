# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

SET(CMAKE_CXX_STANDARD 11) # comment out to deactivate C++11 when using ROS indigo to avoid incompatibilities

# --- To be used by other modules ---
SET(GPU_VOXELS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE INTERNAL "")
SET(GPU_VOXELS_IDL_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")
SET(GPU_VOXELS_IDE_FOLDER "gpu_voxels")

# --- Global options ---

SET(CUDA_USE_STATIC_CUDA_RUNTIME OFF) # fix for cuda 9.1 thrust cub addArgument runtime segfault
FIND_PACKAGE(CUDA)

FIND_PACKAGE(Boost COMPONENTS system thread filesystem date_time unit_test_framework chrono)

# LibRT is needed for Boost Interprocess on POSIX systems
IF(UNIX AND NOT APPLE)
  FIND_PACKAGE(LibRt)
ENDIF(UNIX AND NOT APPLE)

# Dependencies for visualization
FIND_PACKAGE(GLEW)
FIND_PACKAGE(GLM)
FIND_PACKAGE(OpenGL)
FIND_PACKAGE(GLUT)

# Dependencies for ROS interconnect
FIND_PACKAGE(ROS)

# Dependencies for URDF loading (also requires ROS for kdl/frames.hpp)
FIND_PACKAGE(urdfdom)
FIND_PACKAGE(orocos_kdl)
FIND_PACKAGE(kdl_parser)

# Dependencies for Benchmarking against Octomap
FIND_PACKAGE(Octomap)

# Dependencies for PCL interfaces
FIND_PACKAGE(PCL)

FIND_PACKAGE(pcl_ros)

# Dependencies for Kinect support
FIND_PACKAGE(OpenNi)
#FIND_PACKAGE(OpenNi2)

# ICL Package management
ICMAKER_REGISTER_PACKAGE(gpu_voxels)

MESSAGE(STATUS "--------------------------------------------------------------------------")
MESSAGE(STATUS "------------------------ GPU Voxels configuration ------------------------")
MESSAGE(STATUS "  ")

IF(CUDA_FOUND)
  MESSAGE(STATUS "[OK]      Building GPU-Voxels. Found CUDA.")
ELSE(CUDA_FOUND)
  MESSAGE(STATUS "[ERROR]   Could not find CUDA. GPU-Voxels can not be built!")
ENDIF(CUDA_FOUND)

IF(GLEW_FOUND AND GLM_FOUND AND OPENGL_FOUND AND GLUT_FOUND)
  MESSAGE(STATUS "[OK]      Building GPU-Voxels with visualization. GLEW, GLM, OpenGL and GLUT were found.")
ELSE(GLEW_FOUND AND GLM_FOUND AND OPENGL_FOUND AND GLUT_FOUND)
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without visualization because the following packages are missing:") 
  IF(NOT GLEW_FOUND)
    MESSAGE(STATUS "    Glew")
  ENDIF(NOT GLEW_FOUND)
  IF(NOT GLM_FOUND)
    MESSAGE(STATUS "    GLM")
  ENDIF(NOT GLM_FOUND)
  IF(NOT OPENGL_FOUND)
    MESSAGE(STATUS "    OpenGL")
  ENDIF(NOT OPENGL_FOUND)
  IF(NOT GLUT_FOUND)
    MESSAGE(STATUS "    Glut")
  ENDIF(NOT GLUT_FOUND)
ENDIF(GLEW_FOUND AND GLM_FOUND AND OPENGL_FOUND AND GLUT_FOUND)

IF(ROS_FOUND)
  MESSAGE(STATUS "[OK]      Building GPU-Voxels with ROS connections. ROS was found.")
ELSE(ROS_FOUND)
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without ROS connections. ROS not found.")
ENDIF(ROS_FOUND)

IF(urdfdom_FOUND AND orocos_kdl_FOUND AND kdl_parser_FOUND)
  MESSAGE(STATUS "[OK]      Building GPU-Voxels with URDF support. urdfdom, orocos and kdl_parser and ROS-kdl were found.")
ELSE(urdfdom_FOUND AND orocos_kdl_FOUND AND kdl_parser_FOUND)
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without URDF support. Could not find the following packages:")
  IF(NOT urdfdom_FOUND)
    MESSAGE(STATUS "    urdfdom")
  ENDIF(NOT urdfdom_FOUND)
  IF(NOT orocos_kdl_FOUND)
    MESSAGE(STATUS "    orocos_kdl")
  ENDIF(NOT orocos_kdl_FOUND)
  IF(NOT kdl_parser_FOUND)
    MESSAGE(STATUS "    kdl_parser")
  ENDIF(NOT kdl_parser_FOUND)
  IF(ROS_FOUND)
    MESSAGE(STATUS "!!! to fix this, source /opt/ros/YOUR_ROS_DISTRO/setup.bash before building !!!")
  ENDIF(ROS_FOUND)
ENDIF(urdfdom_FOUND AND orocos_kdl_FOUND AND kdl_parser_FOUND)

#IF(OPENNI2_FOUND)
#  MESSAGE(STATUS "[OK]      Building GPU-Voxels with Kinect support. OpenNI2 was found.")
#ELSE(OPENNI2_FOUND)
#  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without Kinect support. OpenNI2 not found.")
#ENDIF(OPENNI2_FOUND)

IF(OPENNI_FOUND)
  MESSAGE(STATUS "[OK]      Building GPU-Voxels with Kinect support. OpenNI was found.")
ELSE(OPENNI_FOUND)
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without Kinect support. OpenNI not found.")
ENDIF(OPENNI_FOUND)

IF(Octomap_FOUND)
  MESSAGE(STATUS "[OK]      Building GPU-Voxels with Octomap support. Octomap found.")
ELSEIF(Octomap_FOUND)
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without Octomap support. Octomap not found.")
ENDIF(Octomap_FOUND)

IF(PCL_FOUND)
  # needed because of problems with libvtk6 and PCL on Jetson
  list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
  MESSAGE(STATUS "[OK]      Building GPU-Voxels with PCL interfaces. PCL found.")
ELSE(PCL_FOUND)
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without PCL interfaces. PCL not found.")
ENDIF(PCL_FOUND)

MESSAGE(STATUS "  ")
MESSAGE(STATUS "-------------------- END of GPU Voxels configuration ---------------------")
MESSAGE(STATUS "--------------------------------------------------------------------------")


###############################################################################

# Change these lines to increase performance if your GPU's compute capability is higher
# GPU-Voxels requires GPU CUDA capabilities >= 2.0
#SET(ICMAKER_CUDA_COMPUTE_VERSION 20)
SET(ICMAKER_CUDA_COMPUTE_VERSION 35)

SET(ICMAKER_CUDA_ARCH compute_${ICMAKER_CUDA_COMPUTE_VERSION})
SET(ICMAKER_CUDA_CODE compute_${ICMAKER_CUDA_COMPUTE_VERSION})

#Usage example for specifying PTX for compute_35 and cubin for sm_51:
# SET(ICMAKER_CUDA_CODE compute_35,sm_51)

## Uncomment this definition if you want to avoid PTX JIT compilation at first runtime. 
## Having both a compute_ and a sm_ code target will include both PTX and CUBIN in the final binary.
## IMPORTANT: Check your device's compute capability. the sm_xy argument needs to match your device to have any effect!
## Version table: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
## See also: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#extended-notation
#SET(ICMAKER_CUDA_CODE compute_${ICMAKER_CUDA_COMPUTE_VERSION},sm_${ICMAKER_CUDA_COMPUTE_VERSION})

SET(ICMAKER_CUDA_PTXAS_VERBOSE "") # "--resource-usage") #nvcc outputs register and memory usage data for each kernel
SET(ICMAKER_CUDA_WALL "-Xcompiler=-Wall")
SET(ICMAKER_CUDA_MAXREGS "--maxrregcount=63") # set to 31 to compile for JetsonTX1; N*blocksize must be smaller than max registers per block
SET(ICMAKER_CUDA_DEBUG "-lineinfo") # "-lineinfo") #"-g -G") # -G disables optimization. use -lineinfo for profiling

# Enable position independent code
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

IF( (CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo") )
 MESSAGE(WARNING "Building with THRUST_DEBUG option!")
 SET(ICMAKER_CUDA_THRUST_DEBUG "-DTHRUST_DEBUG" )
ELSE()
 SET(ICMAKER_CUDA_THRUST_DEBUG "" )
ENDIF()

SET(ICMAKER_CUDA_CPPDEFINES "${ICMAKER_CUDA_DEBUG} ${ICMAKER_CUDA_MAXREGS} ${ICMAKER_CUDA_PTXAS_VERBOSE} -gencode=arch=${ICMAKER_CUDA_ARCH},code=[${ICMAKER_CUDA_CODE}] ${ICMAKER_CUDA_THRUST_DEBUG} ${ICMAKER_CUDA_WALL}")

MESSAGE("ICMAKER_CUDA_CPPDEFINES: ${ICMAKER_CUDA_CPPDEFINES}")

###############################################################################
# Remove vtk definitions (Macro copied from PCL)
# This is used for CUDA targets, because nvcc does not like VTK 6+ definitions
# style.
macro(REMOVE_VTK_DEFINITIONS)
    get_directory_property(_dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(_vtk_definitions)
    foreach(_item ${_dir_defs})
        if(_item MATCHES "vtk*")
            list(APPEND _vtk_definitions -D${_item})
        endif()
    endforeach()
    remove_definitions(${_vtk_definitions})
endmacro(REMOVE_VTK_DEFINITIONS)


###############################################################################

REMOVE_VTK_DEFINITIONS()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

ADD_SUBDIRECTORY(src/gpu_voxels)
ADD_SUBDIRECTORY(src/gpu_visualization)

ICMAKER_CONFIGURE_PACKAGE()

###############################################################################
# Build examples
# ADD_SUBDIRECTORY(src/examples)

IF(BUILD_TESTS)
ENDIF(BUILD_TESTS)

