# - Try to find CarMaker
# Once done this will define
#  CarMaker_FOUND - System has CarMaker
#  CarMaker_INCLUDE_DIRS - The CarMaker include directories
#  CarMaker_LIBRARIES - The libraries needed to use CarMaker
#  CarMaker_BIN_DIR - The bin dir of carmaker (used for pre-build steps)

IF( CarMaker_FOUND )
  SET( CarMaker_FIND_QUIETLY TRUE )
ENDIF( CarMaker_FOUND )

INCLUDE(PrintLibraryStatus)
INCLUDE(LibFindMacros)
FIND_PACKAGE( LibUSB )

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(CarMaker_PKGCONF carmaker)

# Include dir
find_path(CarMaker_INCLUDE_DIR
  NAMES CarMaker.h
  PATHS ${CarMaker_PKGCONF_INCLUDE_DIRS} "/opt/ipg/hil/linux/include"
)

find_path(CarMaker_BIN_DIR
  NAMES CreateCarMakerAppInfo
  PATHS ${CarMaker_PKGCONF_INCLUDE_DIRS} "/opt/ipg/hil/linux/bin"
)

# Libraries
find_library(CarMaker_CM_LIBRARY
  NAMES carmaker
  PATHS ${CarMaker_PKGCONF_LIBRARY_DIRS} "/opt/ipg/hil/linux/lib"
)
find_library(CarMaker_C_LIBRARY
  NAMES car
  PATHS ${CarMaker_PKGCONF_LIBRARY_DIRS} "/opt/ipg/hil/linux/lib"
)
find_library(CarMaker_D_LIBRARY
  NAMES ipgdriver
  PATHS ${CarMaker_PKGCONF_LIBRARY_DIRS} "/opt/ipg/hil/linux/lib"
)
find_library(CarMaker_R_LIBRARY
  NAMES ipgroad
  PATHS ${CarMaker_PKGCONF_LIBRARY_DIRS} "/opt/ipg/hil/linux/lib"
)
find_library(CarMaker_T_LIBRARY
  NAMES tametire
  PATHS ${CarMaker_PKGCONF_LIBRARY_DIRS} "/opt/ipg/hil/linux/lib"
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.

set(CarMaker_PROCESS_INCLUDES CarMaker_INCLUDE_DIR LibUSB_INCLUDE_DIRS )
set(CarMaker_PROCESS_LIBS CarMaker_C_LIBRARY CarMaker_CM_LIBRARY CarMaker_D_LIBRARY CarMaker_R_LIBRARY CarMaker_T_LIBRARY LibUSB_LIBRARIES )
libfind_process(CarMaker)

PRINT_LIBRARY_STATUS(CarMaker
  DETAILS "[${CarMaker_LIBRARIES}][${CarMaker_INCLUDE_DIRS}]"
)

if( LibUSB_FOUND )
  if(CarMaker_FOUND)
    set(CarMaker_DEFINITIONS "-D_IC_BUILDER_CARMAKER_")
  endif()
elseif()
  SET( CarMaker_FOUND ) #Unsets CarMaker_FOUND
endif()

