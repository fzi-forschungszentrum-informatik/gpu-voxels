# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

# - Try to find libcrypt
# Once done, this will define
#
#  Crypt_FOUND - system has Ltdl
#  Crypt_INCLUDE_DIR - the Ltdl include directories
#  Crypt_LIBRARY - link these to use Crypt

include(LibFindMacros)

if ( Crypt_LIBRARY AND Crypt_INCLUDE_DIR )
   # in cache already
   SET( Crypt_FIND_QUIETLY TRUE )
endif ()

# Use pkg-config to get hints about paths
# libfind_pkg_check_modules(Ltdl_PKGCONF ltdl)

# Include dir
find_path(Crypt_INCLUDE_DIR
  NAMES crypt.h
  PATHS ${Crypt_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(Crypt_LIBRARY
  NAMES crypt
  PATHS ${Crypt_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(Crypt_PROCESS_INCLUDES Crypt_INCLUDE_DIR)
set(Crypt_PROCESS_LIBS Crypt_LIBRARY)
libfind_process(Crypt)
