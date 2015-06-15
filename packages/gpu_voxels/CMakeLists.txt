# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# --- To be used by other modules ---
SET(GPU_VOXELS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE INTERNAL "")
SET(GPU_VOXELS_IDL_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")
SET(GPU_VOXELS_IDE_FOLDER "gpu_voxels")

# --- Global options ---

FIND_PACKAGE(CUDA)

# LibRT is needed for Boost Interprocess on POSIX systems
IF(UNIX AND NOT APPLE)
  FIND_PACKAGE(LibRt)
ENDIF(UNIX AND NOT APPLE)

# Dependencies for visualization
FIND_PACKAGE(GLEW)
FIND_PACKAGE(GLM)
FIND_PACKAGE(OpenGL)
FIND_PACKAGE(GLUT)

# Dependencies for URDF loading
FIND_PACKAGE(urdfdom)
FIND_PACKAGE(orocos_kdl)
FIND_PACKAGE(kdl_parser)

# Dependencied for Benchmarking against Octomap
FIND_PACKAGE(Octomap)

# Dependencies for PCL interfaces
FIND_PACKAGE(PCL)

# Dependencies for ROS interconnect
FIND_PACKAGE(ROS)

# Dependencies for Kinect support
FIND_PACKAGE(OpenNi)
#FIND_PACKAGE(OpenNi2)

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
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without visualization. Could not find GLEW, GLM, OpenGL or GLUT.")
ENDIF(GLEW_FOUND AND GLM_FOUND AND OPENGL_FOUND AND GLUT_FOUND)

IF(urdfdom_FOUND AND orocos_kdl_FOUND AND kdl_parser_FOUND)
  MESSAGE(STATUS "[OK]      Building GPU-Voxels with URDF support. urdfdom, orocos and kdl_parser were found.")
ELSE(urdfdom_FOUND AND orocos_kdl_FOUND AND kdl_parser_FOUND)
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without URDF support. Could not find urdfdom or orocos or kdl_parser. ")
ENDIF(urdfdom_FOUND AND orocos_kdl_FOUND AND kdl_parser_FOUND)

IF(ROS_FOUND)
  MESSAGE(STATUS "[OK]      Building GPU-Voxels with ROS connections. ROS was found.")
ELSE(ROS_FOUND)
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without ROS connections. ROS not found.")
ENDIF(ROS_FOUND)

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
  MESSAGE(STATUS "[OK]      Building GPU-Voxels with PCL interfaces. PCL found.")
ELSEIF(PCL_FOUND)
  MESSAGE(STATUS "[WARNING] Building GPU-Voxels without PCL interfaces. PCL not found.")
ENDIF(PCL_FOUND)

MESSAGE(STATUS "  ")
MESSAGE(STATUS "-------------------- END of GPU Voxels configuration ---------------------")
MESSAGE(STATUS "--------------------------------------------------------------------------")



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

