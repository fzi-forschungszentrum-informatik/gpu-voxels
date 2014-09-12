# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

# - Try to find FlyCapture2
# Once done, this will define
#
#  FlyCapture2_FOUND - system has FlyCapture2
#  FlyCapture2_INCLUDE_DIRS - the FlyCapture2 include directories
#  FlyCapture2_LIBRARIES - link these to use FlyCapture2

IF( FlyCapture2_FOUND )
   # in cache already
   SET( FlyCapture2_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

# Include dir
find_path(FlyCapture2_INCLUDE_DIR
  NAMES flycapture/Camera.h
)

# Finally the library itself
find_library(FlyCapture2_LIBRARY
  NAMES flycapture
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(FlyCapture2_PROCESS_INCLUDES FlyCapture2_INCLUDE_DIR)
set(FlyCapture2_PROCESS_LIBS FlyCapture2_LIBRARY)
libfind_process(FlyCapture2)

PRINT_LIBRARY_STATUS(FlyCapture2
  DETAILS "[${FlyCapture2_LIBRARIES}][${FlyCapture2_INCLUDE_DIRS}]"
)
