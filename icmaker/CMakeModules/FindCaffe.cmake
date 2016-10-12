#- Try to find the deep learning framework caffe
# Once done, this will define
#
#  CAFFE_FOUND - system has Caffe
#  CAFFE_INCLUDE_DIRS - the Caffe include directories
#  CAFFE_LIBRARIES - link these to use Caffe

include(PrintLibraryStatus)
include(LibFindMacros)

find_path(CAFFE_INCLUDE_DIRS NAMES caffe/caffe.hpp
  PATHS $ENV{CAFFE_ROOT}/distribute/include
)

find_library(CAFFE_LIBRARIES NAMES libcaffe.so
  PATHS $ENV{CAFFE_ROOT}/distribute/lib
)
set(CAFFE_PROCESS_LIBS CAFFE_LIBRARIES)

if(CAFFE_INCLUDE_DIRS AND CAFFE_LIBRARIES)
  SET(CAFFE_FOUND TRUE CACHE BOOL "" FORCE)
else(CAFFE_INCLUDE_DIRS AND CAFFE_LIBRARIES)
  SET(CAFFE_INCLUDE_DIRS "caffe-include-NOTFOUND" CACHE PATH "caffe include path")
endif(CAFFE_INCLUDE_DIRS AND CAFFE_LIBRARIES)

#libfind_process(CAFFE)

PRINT_LIBRARY_STATUS(CAFFE
  DETAILS "[${CAFFE_LIBRARIES}][${CAFFE_INCLUDE_DIRS}]"
)

