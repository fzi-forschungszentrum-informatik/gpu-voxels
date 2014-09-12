# - Try to find YAML
# Once done this will define
#  YAML_FOUND - System has CarMaker
#  YAML_INCLUDE_DIRS - The CarMaker include directories
#  YAML_LIBRARIES - The libraries needed to use CarMaker
#  YAML_DEFINITIONS - Compiler switches required for using CarMaker

SET( YAML_FOUND FALSE )

find_package(PkgConfig)

find_path( YAML_INCLUDE_DIR yaml-cpp/yaml.h /usr/include/ )
find_library( YAML_LIB yaml-cpp HINTS /usr/lib/i386-linux-gnu/ )

include(FindPackageHandleStandardArgs)
INCLUDE(PrintLibraryStatus)

SET( YAML_LIBRARIES 
  ${YAML_LIB}
)

SET( YAML_INCLUDE_DIRS
  ${YAML_INCLUDE_DIR}
)
   

# handle the QUIETLY and REQUIRED arguments and set YAML_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args( YAML DEFAULT_MSG
                                   YAML_LIBRARIES YAML_INCLUDE_DIRS )

mark_as_advanced(YAML_INCLUDE_DIRS YAML_LIBRARIES )

IF( DEFINED PRINT_LIBRARY_STATUS )
  PRINT_LIBRARY_STATUS(YAML
    DETAILS "[${YAML_LIBRARIES}][${YAML_INCLUDE_DIRS}]"
  )
ENDIF (DEFINED PRINT_LIBRARY_STATUS )

