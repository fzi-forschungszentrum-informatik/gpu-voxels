# - Try to find Boost include dirs and libraries
#
# Please see the Documentation for Boost in the CMake Manual for details
# This module only forwards to the one included in cmake for compatibility
# reasons.

# This call is kept for compatibility of this module with CMake 2.6.2, which
# only knows about Boost < 1.37.
# Note: Do _not_ add new Boost versions here, we're trying to get rid
# of this module in kdelibs, but thats only possible if there's a CMake-included
# version that finds all modules that this file finds.
# Instead add a similar call with newer version numbers to the CMakeLists.txt
# in your project before calling find_package(Boost)
#
#  Copyright (c) 2009      Andreas Pakulat <apaku@gmx.de>
#
#  Redistribution AND use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.

include(PrintLibraryStatus)

MACRO (_Boost_ADJUST_INC_VARS basename)
  IF(Boost_INCLUDE_DIR)
    SET(Boost_${basename}_INCLUDE_DIRS ${Boost_INCLUDE_DIR} CACHE FILEPATH "The Boost ${basename} include directory")
    SET(Boost_${basename}_DEFINITIONS -D_IC_BUILDER_BOOST_ -DBOOST_ALL_NO_LIB)
    IF(BUILD_SHARED_LIBS)
      SET(Boost_${basename}_DEFINITIONS ${Boost_${basename}_DEFINITIONS} -DBOOST_DYN_LINK)
    ENDIF(BUILD_SHARED_LIBS)
  ENDIF(Boost_INCLUDE_DIR)
  # Make variables changeble to the advanced user
  MARK_AS_ADVANCED(
      Boost_${basename}_INCLUDE_DIRS
  )
ENDMACRO (_Boost_ADJUST_INC_VARS)

IF(Boost_FOUND)
   # in cache already
   SET(Boost_FIND_QUIETLY TRUE)
ENDIF()

set(Boost_ADDITIONAL_VERSIONS ${Boost_ADDITIONAL_VERSIONS} "1.48" "1.48.0" "1.47" "1.47.0" "1.45" "1.44" "1.43" )

IF(BUILD_SHARED_LIBS)
  set(Boost_USE_STATIC_LIBS OFF CACHE INTERNAL "")
ELSE()
  set(Boost_USE_STATIC_LIBS ON CACHE INTERNAL "")
  set(Boost_USE_STATIC_RUNTIME ON CACHE INTERNAL "")
ENDIF()

include(${CMAKE_ROOT}/Modules/FindBoost.cmake)

# Manually add the component include dirs
# The FindBoost.cmake module does not provide
# the components together with their include dirs
_Boost_ADJUST_INC_VARS(DATE_TIME)
_Boost_ADJUST_INC_VARS(FILESYSTEM)
_Boost_ADJUST_INC_VARS(GRAPH)
_Boost_ADJUST_INC_VARS(IOSTREAMS)
_Boost_ADJUST_INC_VARS(MATH_C99)
_Boost_ADJUST_INC_VARS(MATH_C99F)
_Boost_ADJUST_INC_VARS(MATH_C99L)
_Boost_ADJUST_INC_VARS(MATH_TR1)
_Boost_ADJUST_INC_VARS(MATH_TR1F)
_Boost_ADJUST_INC_VARS(MATH_TR1L)
_Boost_ADJUST_INC_VARS(PRG_EXEC_MONITOR)
_Boost_ADJUST_INC_VARS(PROGRAM_OPTIONS)
_Boost_ADJUST_INC_VARS(PYTHON)
_Boost_ADJUST_INC_VARS(RANDOM)
_Boost_ADJUST_INC_VARS(REGEX)
_Boost_ADJUST_INC_VARS(SERIALIZATION)
_Boost_ADJUST_INC_VARS(SIGNALS_DIRS)
_Boost_ADJUST_INC_VARS(SYSTEM)
_Boost_ADJUST_INC_VARS(THREAD)
_Boost_ADJUST_INC_VARS(UNIT_TEST_FRAMEWORK)
_Boost_ADJUST_INC_VARS(WAVE)
_Boost_ADJUST_INC_VARS(WSERIALIZATION)

PRINT_LIBRARY_STATUS(Boost
  VERSION "${Boost_VERSION}"
  DETAILS "[${Boost_LIBRARY_DIRS}][${Boost_INCLUDE_DIRS}]"
  COMPONENTS "${Boost_FIND_COMPONENTS}"
)
