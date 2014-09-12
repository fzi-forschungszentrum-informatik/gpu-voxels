# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

# - Try to find libspeechd-dev
# Once done this will define
#  Speechd_FOUND - System has libspeechd
#  Speechd_INCLUDE_DIRS - The libspeechd include directories
#  Speechd_LIBRARIES - link these to use libspeechd

include(PrintLibraryStatus)
include(LibFindMacros)


find_path(speechd_INCLUDE_DIR
  NAMES libspeechd.h
  PATHS "/usr/include"
)

find_library(speechd_LIBRARY
  NAMES libspeechd.so
  PATHS "/usr/lib" "/usr/local/lib"
)


set(speechd_PROCESS_INCLUDES speechd_INCLUDE_DIR)
set(speechd_PROCESS_LIBS speechd_LIBRARY)
libfind_process(speechd)


PRINT_LIBRARY_STATUS(speechd
  DETAILS "[${speechd_LIBRARIES}][${speechd_INCLUDE_DIRS}]"
)

