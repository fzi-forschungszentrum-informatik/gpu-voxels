# - Try to find QCustomPlot
# Once done, this will define
#
#  QCustomPlot_FOUND - system has QCustomPlot
#  QCustomPlot_INCLUDE_DIRS - the QCustomPlot include directories
#  QCustomPlot_LIBRARIES - link these to use QCustomPlot

IF( QCustomPlot_FOUND )
   # in cache already
   SET( QCustomPlot_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(QCustomPlot_PKGCONF libqcustomplot)

# Include dir
find_path(QCustomPlot_INCLUDE_DIR
  NAMES qcustomplot.h qcustomplot/qcustomplot.h
  PATHS ${QCustomPlot_PKGCONF_INCLUDE_DIRS} "/usr/include"
)

# Finally the library itself
find_library(QCustomPlot_LIBRARY
  NAMES qcustomplot
  PATHS ${QCustomPlot_PKGCONF_LIBRARY_DIRS} "/usr/lib"
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(QCustomPlot_PROCESS_INCLUDES QCustomPlot_INCLUDE_DIR)
set(QCustomPlot_PROCESS_LIBS QCustomPlot_LIBRARY)
libfind_process(QCustomPlot)

PRINT_LIBRARY_STATUS(QCustomPlot
  DETAILS "[${QCustomPlot_LIBRARIES}][${QCustomPlot_INCLUDE_DIRS}]"
)

