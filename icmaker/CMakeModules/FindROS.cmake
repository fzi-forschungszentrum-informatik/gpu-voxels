# - Try to find ROS
# Once done, this will define
#
#  ROS_FOUND - system has ROS
#  ROS_INCLUDE_DIRS - the ROS include directories
#  ROS_LIBRARIES - link these to use ROS

IF(ROS_FOUND)
  # in cache already
  SET( ROS_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

# Ros installation directory
IF(NOT ROS_ROOT)
  IF(EXISTS "/opt/ros/indigo/include/ros/time.h")
    set(ROS_ROOT /opt/ros/indigo)
  ELSEIF(EXISTS "/opt/ros/hydro/include/ros/time.h")
    set(ROS_ROOT /opt/ros/hydro)
  ELSEIF(EXISTS "/opt/ros/groovy/include/ros/time.h")
    set(ROS_ROOT /opt/ros/groovy)
  ELSE()
    set(ROS_ROOT /opt/ros/fuerte)
  ENDIF()
ENDIF()

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(ROS_PKGCONF ssl)

# Include dir
find_path(ROS_INCLUDE_DIR
  NAMES ros/time.h
  PATHS ${ROS_PKGCONF_INCLUDE_DIRS} "${ROS_ROOT}/include"
)

SET (ros_libraries rostime roslib roscpp roscpp_serialization rosconsole tf)
FOREACH (ros_library ${ros_libraries})
  find_library(ROS_${ros_library}_LIBRARY
    NAMES ${ros_library}
    PATHS ${ROS_PKGCONF_LIBRARY_DIRS} "${ROS_ROOT}/lib"
  )
  #MESSAGE(STATUS "Appending ${ROS_${ros_library}_LIBRARY}")
  LIST (APPEND ROS_CORE_LIBRARIES ${ROS_${ros_library}_LIBRARY})
ENDFOREACH()

# sensor_msgs is not available in all ros builds
find_library(ROS_sensor_msgs_LIBRARY
  NAMES sensor_msgs
  PATHS ${ROS_PKGCONF_LIBRARY_DIRS} "${ROS_ROOT}/lib"
)
IF(ROS_sensor_msgs_LIBRARY)
  #MESSAGE(STATUS "Appending ${ROS_sensor_msgs_LIBRARY}")
  LIST (APPEND ROS_CORE_LIBRARIES ${ROS_sensor_msgs_LIBRARY})
ENDIF()

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(ROS_PROCESS_INCLUDES ROS_INCLUDE_DIR)
set(ROS_PROCESS_LIBS ROS_CORE_LIBRARIES)
libfind_process(ROS)

PRINT_LIBRARY_STATUS(ROS
  DETAILS "[${ROS_LIBRARIES}][${ROS_INCLUDE_DIRS}]"
  COMPONENTS "${ROS_FIND_COMPONENTS}"
)

### Handle components

IF(ROS_FIND_COMPONENTS)

  ##############################################
  #
  # ROS stack configuration. Adjust below vvvv
  #
  ##############################################

  # ROS bullet stack
  SET( ROS_BULLET_STACK bullet )
  SET( ROS_BULLET_HEADER_NAMES LinearMath/btVector3.h )
  SET( ROS_BULLET_LIBRARY_NAMES LinearMath )

  # ROS tf stack
  SET( ROS_TF_STACK geometry/tf )
  SET( ROS_TF_HEADER_NAMES tf/tf.h tf/tfMessage.h tf/transform_broadcaster.h tf/FrameGraph.h)
  SET( ROS_TF_INCLUDE_DIR_SUGGESTIONS
    ${ROS_ROOT}/stacks/geometry/tf/include
    ${ROS_ROOT}/stacks/geometry/tf/msg_gen/cpp/include
    ${ROS_ROOT}/share/geometry/tf/include
    ${ROS_ROOT}/share/geometry/tf/msg_gen/cpp/include
    ${ROS_ROOT}/stacks/geometry/tf/srv_gen/cpp/include
    ${ROS_ROOT}/share/geometry/tf/srv_gen/cpp/include
    ${ROS_ROOT}/include
    ${ROS_ROOT}/include/tf )

  # ROS tf2_ros stack
  SET( ROS_TF2_ROS_HEADER_NAMES tf2_ros/transform_broadcaster.h)
  SET( ROS_TF2_ROS_LIBRARY_NAMES tf2 tf2_ros)

  # ROS actionlib stack
  SET( ROS_ACTIONLIB_HEADER_NAMES actionlib/action_definition.h)

  # ROS eigen_conversions stack
  SET( ROS_EIGEN_CONVERSIONS_HEADER_NAMES eigen_conversions/eigen_msg.h)

  # ROS image_proc stack
  SET( ROS_IMAGE_PROC_STACK image_pipeline/image_proc )
  SET( ROS_IMAGE_PROC_HEADER_NAMES image_proc/processor.h )
  SET( ROS_IMAGE_PROC_LIBRARY_NAMES image_proc )

  # ROS cv_bridge stack (needed for ROS < groovy)
  SET( ROS_CV_BRIDGE_STACK vision_opencv/cv_bridge )

  # ROS image_geometry stack (needed for ROS < groovy)
  SET( ROS_IMAGE_GEOMETRY_STACK vision_opencv/image_geometry )
  SET( ROS_IMAGE_GEOMETRY_HEADER_NAMES image_geometry/pinhole_camera_model.h )

  # ROS image_transport stack (needed for ROS < groovy)
  SET( ROS_IMAGE_TRANSPORT_STACK image_common/image_transport )

  # ROS pcl_ros stack (needed for ROS < groovy)
  SET( ROS_PCL_ROS_STACK perception_pcl/pcl_ros )
  SET( ROS_PCL_ROS_HEADER_NAMES pcl_ros/point_cloud.h )
  SET( ROS_PCL_ROS_LIBRARY_NAMES pcl_ros_filters pcl_ros_io pcl_ros_tests pcl_ros_tf)

  # ROS interactive_markers stack
  SET( ROS_INTERACTIVE_MARKERS_HEADER_NAMES interactive_markers/interactive_marker_server.h )

  # ROS message filters
  SET( ROS_MESSAGE_FILTERS_HEADER_NAMES message_filters/subscriber.h)
  SET( ROS_MESSAGE_FILTERS_LIBRARY_NAMES message_filters)

  # ROS gps common
  SET( ROS_GPS_COMMON_INCLUDE_DIR_SUGGESTIONS ${ROS_ROOT}/stacks/gps_umd/gps_common/msg_gen/cpp/include ${ROS_ROOT}/include)
  SET( ROS_GPS_COMMON_HEADER_NAMES gps_common/GPSFix.h)
  SET( ROS_GPS_COMMON_HEADER_ONLY TRUE)

  # ROS pcl_conversions
  SET( ROS_PCL_CONVERSIONS_HEADER_ONLY TRUE)

  ##############################################
  #
  # ROS stack configuration. Adjust above ^^^^
  #
  ##############################################

  FOREACH( component ${ROS_FIND_COMPONENTS} )
    STRING( TOUPPER ${component} _COMPONENT )
    IF (NOT ROS_${_COMPONENT}_STACK)
      SET( ROS_${_COMPONENT}_STACK ${component} )
    ENDIF()
    IF (NOT ROS_${_COMPONENT}_INCLUDE_DIR_SUGGESTIONS)
      SET( ROS_${_COMPONENT}_INCLUDE_DIR_SUGGESTIONS
        ${ROS_ROOT}/stacks/${ROS_${_COMPONENT}_STACK}/include
        ${ROS_ROOT}/share/${ROS_${_COMPONENT}_STACK}/include
        ${ROS_ROOT}/include )
    ENDIF()
    IF (NOT ROS_${_COMPONENT}_HEADER_NAMES)
      SET( ROS_${_COMPONENT}_HEADER_NAMES ${component}/${component}.h )
    ENDIF()
    IF (NOT ROS_${_COMPONENT}_LIB_DIR_SUGGESTION)
      SET( ROS_${_COMPONENT}_LIB_DIR_SUGGESTION
        ${ROS_ROOT}/stacks/${ROS_${_COMPONENT}_STACK}/lib
        ${ROS_ROOT}/share/${ROS_${_COMPONENT}_STACK}/lib
        ${ROS_ROOT}/lib )
    ENDIF()
    IF (NOT ROS_${_COMPONENT}_LIBRARY_NAMES)
      SET( ROS_${_COMPONENT}_LIBRARY_NAMES ${component} )
    ENDIF()

    FOREACH( header_name ${ROS_${_COMPONENT}_HEADER_NAMES} )
      #MESSAGE(STATUS "Searching for ${header_name} in ${ROS_${_COMPONENT}_INCLUDE_DIR_SUGGESTIONS}")
      FIND_PATH(ROS_${_COMPONENT}_INCLUDE_DIR_${header_name}
        NAMES ${header_name}
        PATHS ${ROS_${_COMPONENT}_INCLUDE_DIR_SUGGESTIONS}
      )
      #MESSAGE(STATUS "Appending ${ROS_${_COMPONENT}_INCLUDE_DIR_${header_name}}")
      LIST(APPEND ROS_${_COMPONENT}_INCLUDES ${ROS_${_COMPONENT}_INCLUDE_DIR_${header_name}})
    ENDFOREACH()

    FOREACH( lib_name ${ROS_${_COMPONENT}_LIBRARY_NAMES} )
      #MESSAGE(STATUS "Searching for ${lib_name} in ${ROS_${_COMPONENT}_LIB_DIR_SUGGESTION}")
      FIND_LIBRARY(ROS_${_COMPONENT}_LIBRARY_${lib_name}
        NAMES ${lib_name}
        PATHS ${ROS_PKGCONF_LIBRARY_DIRS} ${ROS_${_COMPONENT}_LIB_DIR_SUGGESTION}
      )
      LIST(APPEND ROS_${_COMPONENT}_LIBRARIES ${ROS_${_COMPONENT}_LIBRARY_${lib_name}})
    ENDFOREACH()

    SET(ROS_${_COMPONENT}_PROCESS_INCLUDES ROS_${_COMPONENT}_INCLUDES)
    IF (NOT ROS_${_COMPONENT}_HEADER_ONLY)
      SET(ROS_${_COMPONENT}_PROCESS_LIBS ROS_${_COMPONENT}_LIBRARIES)
    ENDIF()
    libfind_process(ROS_${_COMPONENT})
  ENDFOREACH()
ENDIF()
