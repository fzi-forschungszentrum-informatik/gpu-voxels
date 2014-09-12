# - Try to find Alut
# Once done this will define
#  Alut_FOUND - System has CarMaker
#  Alut_INCLUDE_DIRS - The CarMaker include directories
#  Alut_LIBRARIES - The libraries needed to use CarMaker
#  Alut_DEFINITIONS - Compiler switches required for using CarMaker

SET( Alut_FOUND FALSE )

find_package(PkgConfig)

find_path( Alut_INCLUDE_DIR AL/alut.h HINTS "/usr/include" )
find_library( Alut_LIB alut HINTS "/usr/lib/" )

include(FindPackageHandleStandardArgs)
INCLUDE(PrintLibraryStatus)

SET( Alut_LIBRARIES 
  ${Alut_LIB}
)

SET( Alut_INCLUDE_DIRS
  ${Alut_INCLUDE_DIR}
)
   

# handle the QUIETLY and REQUIRED arguments and set CarMakerApo_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args( Alut DEFAULT_MSG
                                   Alut_LIBRARIES Alut_INCLUDE_DIRS )

mark_as_advanced(Alut_INCLUDE_DIRS Alut_LIBRARIES )

IF( DEFINED PRINT_LIBRARY_STATUS )
  PRINT_LIBRARY_STATUS(Alut
    DETAILS "[${Alut_LIBRARIES}][${Alut_INCLUDE_DIRS}]"
  )
ELSE( DEFINED PRINT_LIBRARY_STATUS )
  MESSAGE( "Alut: ${Alut_INCLUDE_DIRS}" )
  MESSAGE( "Alut: ${Alut_LIBRARIES}" ) 
ENDIF (DEFINED PRINT_LIBRARY_STATUS )
