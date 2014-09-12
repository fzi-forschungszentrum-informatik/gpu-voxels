# - Try to find CoreFoundation
# Once done, this will define
#
#  CoreFoundation_FOUND - system has CoreFoundation
#  CoreFoundation_FRAMEWORKS - the CoreFoundation frameworks

INCLUDE(CMakeFindFrameworks)

CMAKE_FIND_FRAMEWORKS(CoreFoundation)
IF (CoreFoundation_FRAMEWORKS)
  SET (CoreFoundation_INCLUDE_DIRS CoreFoundation_FRAMEWORK_INCLUDES)
  SET (CoreFoundation_LIBRARIES "-framework CoreFoundation" CACHE STRING "CoreFoundation library")
ENDIF ()

# handle the QUIETLY and REQUIRED arguments and set CoreFoundation_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CoreFoundation DEFAULT_MSG CoreFoundation_LIBRARIES CoreFoundation_INCLUDE_DIRS)

MARK_AS_ADVANCED(CoreFoundation_INCLUDE_DIRS CoreFoundation_LIBRARIES )
