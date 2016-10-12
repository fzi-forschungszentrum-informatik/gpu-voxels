# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-11-07
#
# Different versions of Ubuntu and ROS provide different find scripts
# for Eigen3, setting different variables.  To hide this mess from the
# user, the "always_find_eigen3" macro unifies the different versions
# and defines the following variables:
#
#  - Eigen3_FOUND         True if Eigen3 was found. Duh.
#  - Eigen3_INCLUDE_DIRS  Required include directories.
#  - Eigen3_VERSION       The version, if the find script provides it.
#  - Eigen3_DEFINITIONS   Additional preprocessor defines provided by
#                         some versions of the find script.
# ----------------------------------------------------------------------

# Try all possible versions
find_package(Eigen3 QUIET)
find_package(Eigen QUIET)

# And now unify the variables
if (Eigen3_FOUND)
  # Just in case
  set(Eigen3_INCLUDE_DIRS ${Eigen3_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIR})
elseif (EIGEN3_FOUND)
  set(Eigen3_FOUND TRUE)
  set(Eigen3_INCLUDE_DIRS ${Eigen3_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})
  set(Eigen3_VERSION ${EIGEN3_VERSION})
  set(Eigen3_DEFINITIONS ${EIGEN3_DEFINITIONS})
elseif (Eigen_FOUND)
  set(Eigen3_FOUND TRUE)
  set(Eigen3_INCLUDE_DIRS ${Eigen3_INCLUDE_DIRS} ${Eigen_INCLUDE_DIR} ${Eigen_INCLUDE_DIRS})
  set(Eigen3_VERSION ${Eigen_VERSION})
  set(Eigen3_DEFINITIONS ${Eigen_DEFINITIONS})
endif ()

if (Eigen3_FOUND)
  message(STATUS "Found Eigen3 version ${Eigen3_VERSION}")
else ()
  message(WARNING "Could not find Eigen or Eigen3")
endif ()
