# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find SVM.  Once done, this will define:
#  SVM_FOUND:          System has SVM
#  SVM_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  SVM_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  SVM_DEFINITIONS:    Preprocessor definitions.
#  SVM_LIBRARIES:      only the libraries (w/o the '-l')
#  SVM_LDFLAGS:        all required linker flags
#  SVM_LDFLAGS_OTHER:  all other linker flags
#  SVM_CFLAGS:         all required cflags
#  SVM_CFLAGS_OTHER:   the other compiler flags
#  SVM_VERSION:        version of the module
#  SVM_PREFIX:         prefix-directory of the module
#  SVM_INCLUDEDIR:     include-dir of the module
#  SVM_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(SVM svm
  HEADERS svm.h
  LIBRARIES svm
  HEADER_PATHS /usr/include/libsvm
  )
