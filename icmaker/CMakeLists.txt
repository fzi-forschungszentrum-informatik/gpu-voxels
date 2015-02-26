# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
# ----------------------------------------------------------------------------
#  Root CMake file for IcMaker
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_ICMAKER_ROOT>
#
#
#   - OCT-2010: Initial version <schamm@fzi.de>
#               Taken from OpenCV
#
# ----------------------------------------------------------------------------

SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)
# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
IF(UNIX)
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /lib /usr/lib)
ENDIF()
# it _must_ go before PROJECT(IcMaker) in order to work
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "Install path prefix, prepended onto install directories (default: CMAKE_BINARY_DIR)." FORCE)
    ELSE()
        SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/export" CACHE PATH "Install path prefix, prepended onto install directories (default: ../export)." FORCE)
    ENDIF()
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

SET(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Configs" FORCE)
SET(CMAKE_C_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
SET(CMAKE_CXX_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
SET(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
SET(CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
SET(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
SET(CMAKE_VERBOSE OFF CACHE BOOL "Verbose mode")

IF(CMAKE_VERBOSE OR CMAKE_EXTRA_GENERATOR MATCHES "Eclipse CDT.*")
    SET(CMAKE_VERBOSE_MAKEFILE ON)
ENDIF()

IF(ICMAKER_PROJECT_NAME)
  SET(ICMAKER_PROJECT_NAME ${ICMAKER_PROJECT_NAME} CACHE INTERNAL "")
ELSE()
  SET(ICMAKER_PROJECT_NAME IcMaker CACHE INTERNAL "")
ENDIF()
PROJECT(${ICMAKER_PROJECT_NAME})

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

IF(MSVC)
    SET(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
ENDIF()

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
IF(POLICY CMP0003)
    CMAKE_POLICY(SET CMP0003 NEW)
ENDIF()
IF(POLICY CMP0015)
    CMAKE_POLICY(SET CMP0015 NEW)
ENDIF()

# ----------------------------------------------------------------------------
#  Current version number:
# ----------------------------------------------------------------------------
SET(ICMAKER_VERSION "1.0.0")

STRING(REGEX MATCHALL "[0-9]" ICMAKER_VERSION_PARTS "${ICMAKER_VERSION}")

LIST(GET ICMAKER_VERSION_PARTS 0 ICMAKER_VERSION_MAJOR)
LIST(GET ICMAKER_VERSION_PARTS 1 ICMAKER_VERSION_MINOR)
LIST(GET ICMAKER_VERSION_PARTS 2 ICMAKER_VERSION_PATCH)

SET(ICMAKER_SOVERSION "${ICMAKER_VERSION_MAJOR}.${ICMAKER_VERSION_MINOR}")

IF(WIN32)
    # Postfix of DLLs:
    SET(ICMAKER_DLLVERSION "${ICMAKER_VERSION_MAJOR}${ICMAKER_VERSION_MINOR}${ICMAKER_VERSION_PATCH}")
    SET(ICMAKER_DEBUG_POSTFIX d)
ELSE()
    # Postfix of so's:
    #SET(ICMAKER_DLLVERSION "${ICMAKER_VERSION_MAJOR}${ICMAKER_VERSION_MINOR}${ICMAKER_VERSION_PATCH}")
    SET(ICMAKER_DLLVERSION "")
    SET(ICMAKER_DEBUG_POSTFIX)
ENDIF()


# ----------------------------------------------------------------------------
# Build static or dynamic libs?
# ----------------------------------------------------------------------------
# Default: dynamic libraries:
SET(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")
IF(BUILD_SHARED_LIBS)
    SET(ICMAKER_BUILD_SHARED_LIB 1)
ELSE(BUILD_SHARED_LIBS)
    SET(ICMAKER_BUILD_SHARED_LIB 0)
	ADD_DEFINITIONS(-D_IC_STATIC_)

        # set /MT for windows static build according to http://www.cmake.org/Wiki/CMake_FAQ#How_can_I_build_my_MSVC_application_with_a_static_runtime.3F
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /MTd")
        SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
        SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} /MT")

ENDIF(BUILD_SHARED_LIBS)

# ----------------------------------------------------------------------------
#  Variables for icmaker.h.cmake
# ----------------------------------------------------------------------------
SET(PACKAGE "icmaker")
SET(PACKAGE_NAME "icmaker")
SET(PACKAGE_STRING "${PACKAGE} ${ICMAKER_VERSION}")
SET(PACKAGE_TARNAME "${PACKAGE}")
SET(PACKAGE_VERSION "${ICMAKER_VERSION}")

# Configure IcMaker:
# ===================================================
SET(ICMAKER_VERBOSE OFF CACHE BOOL "Verbose IcMaker status messages")
SET(ICMAKER_USE_DEPRECATED_MACROS OFF CACHE BOOL "Allow usage of deprecated macros")
SET(ICMAKER_USE_TCMALLOC OFF CACHE BOOL "Link everything against tcmalloc from Google's perftools")

# Detect GNU version:
# ===================================================
IF(CMAKE_COMPILER_IS_GNUCXX)
    EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                    OUTPUT_VARIABLE CMAKE_ICMAKER_GCC_VERSION_FULL
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Typical output in CMAKE_ICMAKER_GCC_VERSION_FULL: "c+//0 (whatever) 4.2.3 (...)"
    #  Look for the version number
    STRING(REGEX MATCH "[0-9].[0-9].[0-9]" CMAKE_GCC_REGEX_VERSION "${CMAKE_ICMAKER_GCC_VERSION_FULL}")

    # Split the three parts:
    STRING(REGEX MATCHALL "[0-9]" CMAKE_ICMAKER_GCC_VERSIONS "${CMAKE_GCC_REGEX_VERSION}")

    LIST(GET CMAKE_ICMAKER_GCC_VERSIONS 0 CMAKE_ICMAKER_GCC_VERSION_MAJOR)
    LIST(GET CMAKE_ICMAKER_GCC_VERSIONS 1 CMAKE_ICMAKER_GCC_VERSION_MINOR)

    SET(CMAKE_ICMAKER_GCC_VERSION ${CMAKE_ICMAKER_GCC_VERSION_MAJOR}${CMAKE_ICMAKER_GCC_VERSION_MINOR})
    MATH(EXPR CMAKE_ICMAKER_GCC_VERSION_NUM "${CMAKE_ICMAKER_GCC_VERSION_MAJOR}*100 + ${CMAKE_ICMAKER_GCC_VERSION_MINOR}")
    MESSAGE(STATUS "Detected version of GNU GCC: ${CMAKE_ICMAKER_GCC_VERSION} (${CMAKE_ICMAKER_GCC_VERSION_NUM})")

    IF(WIN32)
        EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
                  OUTPUT_VARIABLE CMAKE_ICMAKER_GCC_TARGET_MACHINE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
        IF(CMAKE_ICMAKER_GCC_TARGET_MACHINE MATCHES "64")
            SET(MINGW64 1)
        ENDIF()
    ENDIF()
ENDIF()

# Load additional CMake modules
# ===================================================
IF(ICMAKER_DIRECTORY)
MESSAGE(STATUS "icmaker dir is ${ICMAKER_DIRECTORY}")
    SET(CMAKE_MODULE_PATH "${ICMAKER_DIRECTORY}/CMakeModules/;${CMAKE_MODULE_PATH}")
ELSE()
    SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/;${CMAKE_MODULE_PATH}")
ENDIF()

# Build/install (or not) some apps:
# ===================================================
SET(BUILD_EXAMPLES ON CACHE BOOL "Build all examples")

# Build tests:
# ===================================================
SET(BUILD_TESTS ON CACHE BOOL "Build tests")
ENABLE_TESTING() # Always enable testing, so that at least the "test" target is available, even if the tests are not built!

IF(ICMAKER_DIRECTORY)
    INCLUDE(${ICMAKER_DIRECTORY}/IcMakerMacros.cmake REQUIRED)
ELSE()
    INCLUDE(IcMakerMacros.cmake REQUIRED)
ENDIF()

IF(UNIX)
#    IF(ICMAKER_DIRECTORY)
#        INCLUDE(${ICMAKER_DIRECTORY}/IcMakerFindPkgConfig.cmake OPTIONAL)
#    ELSE()
#        INCLUDE(IcMakerFindPkgConfig.cmake OPTIONAL)
#    ENDIF()
    INCLUDE(CheckFunctionExists)
    INCLUDE(CheckIncludeFile)
ENDIF()

# Enable or disable CUDA (disabled by default because of compiler compatibility
# ===================================================
SET(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA - if off, will cause CUDA find script to not run.")

# Set compiler flags
# ===================================================
IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")
    SET(USE_OMIT_FRAME_POINTER ON CACHE BOOL "Enable -fomit-frame-pointer for GCC")
    SET(USE_PERMISSIVE OFF CACHE BOOL "Enable -fpermissive for GCC")
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES arm*)
        # We can use only -O2 because the -O3 causes gcc crash
        SET(USE_O2 ON CACHE BOOL "Enable -O2 for GCC")
    ENDIF()

    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
        SET(X86_64 1)
    ENDIF()
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
        SET(X86_64 1)
    ENDIF()

    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686*)
        SET(X86 1)
    ENDIF()
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES i386*)
        SET(X86 1)
    ENDIF()
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86*)
        SET(X86 1)
    ENDIF()

    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES powerpc*)
        SET(ENABLE_POWERPC ON CACHE BOOL "Enable PowerPC for GCC")
    endif ()

    IF(X86 OR X86_64)
        # enable everything, since the available set of instructions is checked at runtime
        SET(ENABLE_SSE ON CACHE BOOL "Enable SSE for GCC")
        SET(ENABLE_SSE2 ON CACHE BOOL "Enable SSE2 for GCC")
        SET(ENABLE_SSE3 OFF CACHE BOOL "Enable SSE3 for GCC")
        SET(ENABLE_SSSE3 OFF CACHE BOOL "Enable SSSE3 for GCC")
        #SET(ENABLE_SSE4_1 OFF CACHE BOOL "Enable SSE4.1 for GCC")
    ENDIF()
ENDIF()

# ----------------------------------------------------------------------------
# IcMaker Definitions
# ----------------------------------------------------------------------------

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# IcMaker operating system defines
# ===================================================
IF(WIN32)
    ADD_DEFINITIONS(-D_SYSTEM_WIN32_ -D_USE_MATH_DEFINES)
ENDIF(WIN32)

IF(UNIX)
  IF(NOT APPLE)
    ADD_DEFINITIONS(-D_SYSTEM_LINUX_ -D_SYSTEM_POSIX_)
  ELSE()
    ADD_DEFINITIONS(-D_SYSTEM_DARWIN_ -D_SYSTEM_POSIX_)
  ENDIF(NOT APPLE)
ENDIF(UNIX)

ADD_DEFINITIONS(-D_SYSTEM_IDENTIFIER_=${CMAKE_SYSTEM_PROCESSOR}.${CMAKE_SYSTEM_NAME})

# ----------------------------------------------------------------------------
#           Set the maximum level of warnings:
# ----------------------------------------------------------------------------
# May be set to true for development
SET(ICMAKER_WARNINGS_ARE_ERRORS OFF CACHE BOOL "Treat warnings as errors")

SET(EXTRA_C_FLAGS "")
SET(EXTRA_C_FLAGS_RELEASE "")
SET(EXTRA_C_FLAGS_RELWITHDEBINFO "")
SET(EXTRA_C_FLAGS_DEBUG "")
SET(EXTRA_EXE_LINKER_FLAGS "")
SET(EXTRA_EXE_LINKER_FLAGS_RELEASE "")
SET(EXTRA_EXE_LINKER_FLAGS_RELWITHDEBINFO "")
SET(EXTRA_EXE_LINKER_FLAGS_DEBUG "")

IF(MSVC)
    SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /D _CRT_SECURE_NO_DEPRECATE	/D _CRT_NONSTDC_NO_DEPRECATE /D _SCL_SECURE_NO_WARNINGS /D WIN32_LEAN_AND_MEAN /D _WIN32_WINNT=0x0501 /D NOMINMAX")
    # 64-bit portability warnings, in MSVC8
    IF(MSVC80)
        SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /Wp64")
    ENDIF()
    #IF(MSVC90)
    #    SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /D _BIND_TO_CURRENT_CRT_VERSION=1 /D _BIND_TO_CURRENT_VCLIBS_VERSION=1")
    #ENDIF()

    SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /nologo")
	SET(EXTRA_EXE_LINKER_FLAGS "${EXTRA_EXE_LINKER_FLAGS} /nologo")
	SET(EXTRA_EXE_LINKER_FLAGS_DEBUG "${EXTRA_EXE_LINKER_FLAGS_DEBUG} /DEBUG")

    # Remove unreferenced functions: function level linking
#    SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /Gy")
#    SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /Zi")
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
    # High level of warnings.
    SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall")

    IF(USE_CXX11)
      SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -std=c++11")
    ENDIF()

    # The -Wno-long-long is required in 64bit systems when including sytem headers.
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
      SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    ENDIF()
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
      SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    ENDIF()

    IF(ICMAKER_WARNINGS_ARE_ERRORS)
        SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Werror")
    ENDIF()

    IF(X86)
        IF(NOT MINGW64)
            IF(NOT X86_64)
                IF(NOT APPLE)
                    SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -march=i686")
                ENDIF()
            ENDIF()
        ENDIF()
    ENDIF()

    # Other optimizations
    IF(USE_OMIT_FRAME_POINTER)
       SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fomit-frame-pointer")
       SET(EXTRA_C_FLAGS_RELWITHDEBINFO "${EXTRA_C_FLAGS_RELWITHDEBINFO} -fomit-frame-pointer")
    ENDIF()
    IF(USE_PERMISSIVE)
       SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fpermissive")
       SET(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG} -fpermissive")
       SET(EXTRA_C_FLAGS_RELWITHDEBINFO "${EXTRA_C_FLAGS_RELWITHDEBINFO} -fpermissive")
    ENDIF()
    IF(USE_O2)
       SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O2")
       SET(EXTRA_C_FLAGS_RELWITHDEBINFO "${EXTRA_C_FLAGS_RELWITHDEBINFO} -O2")
    ENDIF()
    IF(USE_FAST_MATH)
       SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -ffast-math")
       SET(EXTRA_C_FLAGS_RELWITHDEBINFO "${EXTRA_C_FLAGS_RELWITHDEBINFO} -ffast-math")
    ENDIF()
    IF(ENABLE_POWERPC)
       SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
       SET(EXTRA_C_FLAGS_RELWITHDEBINFO "${EXTRA_C_FLAGS_RELWITHDEBINFO} -mcpu=G3 -mtune=G5")
    ENDIF()
    IF(ENABLE_SSE)
       SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse")
    ENDIF()
    IF(ENABLE_SSE2)
       SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse2")
    ENDIF()
    # SSE3 and further should be disabled under MingW because it generates compiler errors
    IF(NOT MINGW)
       IF(ENABLE_SSE3)
          SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
       ENDIF()
       IF(${CMAKE_ICMAKER_GCC_VERSION_NUM} GREATER 402)
          SET(HAVE_GCC43_OR_NEWER 1)
       ENDIF()
       IF(HAVE_GCC43_OR_NEWER OR APPLE)
          IF(ENABLE_SSSE3)
             SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mssse3")
             SET(EXTRA_C_FLAGS_RELWITHDEBINFO "${EXTRA_C_FLAGS_RELWITHDEBINFO} -mssse3")
          ENDIF()
          #IF(ENABLE_SSE4_1)
          #   SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.1")
          #ENDIF()
       ENDIF()
    ENDIF()

    IF(X86 OR X86_64)
        IF(NOT APPLE)
            IF(CMAKE_SIZEOF_VOID_P EQUAL 4)
                SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mfpmath=387")
                SET(EXTRA_C_FLAGS_RELWITHDEBINFO "${EXTRA_C_FLAGS_RELWITHDEBINFO} -mfpmath=387")
            ENDIF()
        ENDIF()
    ENDIF()

    # Profiling?
    IF(ENABLE_PROFILING)
        SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -pg -g")
        SET(EXTRA_C_FLAGS_RELWITHDEBINFO "${EXTRA_C_FLAGS_RELWITHDEBINFO} -pg -g")
    ELSE()
        # Remove unreferenced functions: function level linking
        IF(NOT APPLE)
            SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
        ENDIF()
    ENDIF()

    SET(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG} -O0 -DDEBUG -D_DEBUG")
ENDIF()

# Extra link libs if the user selects building static libs:
IF(NOT BUILD_SHARED_LIBS)
    IF(CMAKE_COMPILER_IS_GNUCXX)
        SET(ICMAKER_LINKER_LIBS ${ICMAKER_LINKER_LIBS} stdc++)
    ENDIF()
ENDIF()

# Extra flags when using the Eclipse generator.
IF (${CMAKE_EXTRA_GENERATOR} MATCHES "Eclipse CDT.*")
    IF(CMAKE_COMPILER_IS_GNUCC)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fmessage-length=0")
    ENDIF(CMAKE_COMPILER_IS_GNUCC)
    IF(CMAKE_COMPILER_IS_GNUCXX)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0")
    ENDIF(CMAKE_COMPILER_IS_GNUCXX)
ENDIF ()

# Advanced Eclipse configuration for use with Klocwork Insight
# ============================================================
IF (${CMAKE_EXTRA_GENERATOR} MATCHES "Eclipse CDT.*")
    IF(ICMAKER_USE_KLOCWORK)
        SET(ICMAKER_USE_KLOCWORK ${ICMAKER_USE_KLOCWORK} CACHE BOOL "")

        SET(CMAKE_MAKE_PROGRAM "kwinject -u ${CMAKE_BUILD_TOOL}")
    ELSE()
        SET(ICMAKER_USE_KLOCWORK OFF CACHE BOOL "")
    ENDIF()
ENDIF()


# Add user supplied extra options (optimization, etc...)
# ==========================================================
SET(ICMAKER_EXTRA_C_FLAGS "" CACHE STRING "Extra compiler options")
SET(ICMAKER_EXTRA_C_FLAGS_RELEASE "" CACHE STRING "Extra compiler options for Release build")
SET(ICMAKER_EXTRA_C_FLAGS_RELWITHDEBINFO "" CACHE STRING "Extra compiler options for RelWithDebInfo build")
SET(ICMAKER_EXTRA_C_FLAGS_DEBUG "-D_IC_DEBUG_" CACHE STRING "Extra compiler options for Debug build")
SET(ICMAKER_EXTRA_EXE_LINKER_FLAGS "" CACHE STRING "Extra linker flags" FORCE)
SET(ICMAKER_EXTRA_EXE_LINKER_FLAGS_RELEASE "" CACHE STRING "Extra linker flags for Release build" FORCE)
SET(ICMAKER_EXTRA_EXE_LINKER_FLAGS_RELWITHDEBINFO "" CACHE STRING "Extra linker flags for RelWithDebInfo build" FORCE)
SET(ICMAKER_EXTRA_EXE_LINKER_FLAGS_DEBUG "" CACHE STRING "Extra linker flags for Debug build" FORCE)
SET(ICMAKER_DEPRECATED_STYLE "YES" CACHE BOOL "Build functions with deprecated coding style")
SET(ICMAKER_ENABLE_BASE_TYPES "YES" CACHE BOOL "Build with the icl_core base types declared")
SET(ICMAKER_DEPRECATED_BASE_TYPES "NO" CACHE BOOL "Build with the icl_core base types declared as deprecated")

# Set compiler and linker options,
# ==========================================================
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS} ${ICMAKER_EXTRA_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS} ${ICMAKER_EXTRA_C_FLAGS}")
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} ${ICMAKER_EXTRA_C_FLAGS_RELEASE}")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} ${ICMAKER_EXTRA_C_FLAGS_RELEASE}")
SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ${ICMAKER_EXTRA_C_FLAGS_DEBUG}")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ${ICMAKER_EXTRA_C_FLAGS_DEBUG}")
SET(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${EXTRA_C_FLAGS_RELWITHDEBINFO} ${ICMAKER_EXTRA_C_FLAGS_RELWITHDEBINFO}")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${EXTRA_C_FLAGS_RELWITHDEBINFO} ${ICMAKER_EXTRA_C_FLAGS_RELWITHDEBINFO}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS} ${ICMAKER_EXTRA_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWUTHDEBINFO} ${EXTRA_EXE_LINKER_FLAGS_RELWITHDEBINFO} ${ICMAKER_EXTRA_EXE_LINKER_FLAGS_RELWITHDEBINFO}")
SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE} ${ICMAKER_EXTRA_EXE_LINKER_FLAGS_RELEASE}")
SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG} ${ICMAKER_EXTRA_EXE_LINKER_FLAGS_DEBUG}")

# Set cxx flags for OS X Mavericks 10.9 (Darwin version 13.x.x).
# The default compiler recursive template depth is set to 128 in 10.9
# which is not sufficient to compile some parts of boost library
IF (UNIX AND APPLE)
  EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  IF (DARWIN_VERSION GREATER 12)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=512")
    MESSAGE(STATUS "OS X Mavericks 10.9 detected. Setting -ftemplate-depth=512")
  ENDIF()
ENDIF()

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Debug:
IF(${CMAKE_GENERATOR} MATCHES " Makefiles")
    IF(NOT CMAKE_BUILD_TYPE)
        SET(CMAKE_BUILD_TYPE Debug CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
    ENDIF(NOT CMAKE_BUILD_TYPE)
ENDIF()

IF("${CMAKE_CONFIGURE_LDFLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_CONFIGURE_LDFLAGS}")
ENDIF("${CMAKE_CONFIGURE_LDFLAGS}")

# Compile deprecated global and member functions (using the obsolete
# coding style) only if this option is set.
IF(ICMAKER_DEPRECATED_STYLE)
    ADD_DEFINITIONS(-D_IC_BUILDER_DEPRECATED_STYLE_)
ENDIF()
# icl_core base type handling
IF(ICMAKER_ENABLE_BASE_TYPES)
    ADD_DEFINITIONS(-D_IC_BUILDER_ENABLE_BASE_TYPES_)
    IF(ICMAKER_DEPRECATED_BASE_TYPES AND ICMAKER_DEPRECATED_STYLE)
        ADD_DEFINITIONS(-D_IC_BUILDER_DEPRECATED_BASE_TYPES_)
    ENDIF()
ENDIF()

# ----------------------------------------------------------------------------
#                       PROCESS SUBDIRECTORIES:
# ----------------------------------------------------------------------------
# Save libs and executables in the same place
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )

# ----------------------------------------------------------------------------
#   Search for base libraries:
# ----------------------------------------------------------------------------

MESSAGE(STATUS "")
MESSAGE(STATUS "Library configuration for icmaker ${ICMAKER_VERSION} =====================================")
MESSAGE(STATUS "")
MESSAGE(STATUS "Checking for libraries not found in previous checks: ")
MESSAGE(STATUS "")
INCLUDE(PrintLibraryStatus)

SET(ICMAKER_USE_PERCEPTION_PACKAGES ON CACHE BOOL "Use perception packages")
SET(ICMAKER_USE_PARALLEL_PACKAGES ON CACHE BOOL "Use parallel packages")
SET(ICMAKER_USE_MAPS_PACKAGES ON CACHE BOOL "Use maps packages")
SET(ICMAKER_USE_HARDWARE_PACKAGES ON CACHE BOOL "Use hardware packages")
SET(ICMAKER_USE_VISUALIZATION_PACKAGES ON CACHE BOOL "Use visualization packages")
SET(ICMAKER_USE_COMMUNICATION_PACKAGES ON CACHE BOOL "Use communication packages")
SET(ICMAKER_USE_OTHER_PACKAGES ON CACHE BOOL "Use other packages")
SET(ICMAKER_USE_DEPRECATED_PACKAGES ON CACHE BOOL "Use deprecated packages")
SET(ICMAKER_USE_EXTRA_PACKAGES ON CACHE BOOL "Use extra packages")

IF(ICMAKER_DIRECTORY)
    INCLUDE(${ICMAKER_DIRECTORY}/CorePackages.cmake REQUIRED)
ELSE()
    INCLUDE(CorePackages.cmake REQUIRED)
ENDIF()

IF(WIN32)
  SET(ICMAKER_DOC_INSTALL_PATH doc)
ELSE()
  SET(ICMAKER_DOC_INSTALL_PATH share/icmaker/doc)
ENDIF()

# LaTeX
# ===================================================
SET(BUILD_LATEX_DOCS OFF CACHE BOOL "Build LaTeX Documentation")

# Doxygen
# ===================================================

IF(DOXYGEN_FOUND)
    SET(BUILD_DOXYGEN_DOCS ON CACHE BOOL "Generate HTML docs using Doxygen")
ENDIF()

IF(BUILD_LATEX_DOCS)
  # INCLUDE(OpenCVFindLATEX.cmake REQUIRED)
  #
  # IF(PDFLATEX_COMPILER)
  #   MESSAGE(STATUS "PDF LaTeX found!")
  # ENDIF()
ENDIF()

MESSAGE(STATUS "")
MESSAGE(STATUS "Documentation: ")

IF(BUILD_LATEX_DOCS AND PDFLATEX_COMPILER)
MESSAGE(STATUS "  Build PDF ... yes")
ELSE()
MESSAGE(STATUS "  Build PDF ... no")
ENDIF()

IF(BUILD_DOXYGEN_DOCS AND DOXYGEN_FOUND)
MESSAGE(STATUS "  Doxygen HTMLs ... yes")
ELSE()
MESSAGE(STATUS "  Doxygen HTMLs ... no")
ENDIF()

# Referenced in IcWorkspace's doc/doxyfile.in.  Can be overridden by
# packages if a different project name is desired, by setting the
# variable in a package's local CMakeLists.txt file.
SET(IC_WORKSPACE_DOXYGEN_PROJECT_NAME "IcWorkspace" CACHE INTERNAL "")

# Referenced in IcWorkspace's doc/doxyfile.in.  Can be set or appended
# to by packages if Doxygen configuration options should be
# overridden.
SET(IC_WORKSPACE_DOXYGEN_CONFIG_OVERRIDES "IcWorkspace" CACHE INTERNAL "")

# ----------------------------------------------------------------------------
#   Summary:
# ----------------------------------------------------------------------------
MESSAGE(STATUS "")
MESSAGE(STATUS "Compile configuration for icmaker ${ICMAKER_VERSION} =====================================")
MESSAGE(STATUS "")
MESSAGE(STATUS "    Built as dynamic libs?:        ${BUILD_SHARED_LIBS}")
MESSAGE(STATUS "    Compiler:                      ${CMAKE_COMPILER}")
MESSAGE(STATUS "    C++ flags (Release):           ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
MESSAGE(STATUS "    C++ flags (Debug):             ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
MESSAGE(STATUS "    C++ flags (RelWithDebInfo):    ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
IF(WIN32)
MESSAGE(STATUS "    Linker flags (Release):        ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
MESSAGE(STATUS "    Linker flags (Debug):          ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
MESSAGE(STATUS "    Linker flags (RelWithDebInfo): ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}")
ELSE()
MESSAGE(STATUS "    Linker flags (Release):        ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
MESSAGE(STATUS "    Linker flags (Debug):          ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
MESSAGE(STATUS "    Linker flags (RelWithDebInfo): ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}")
ENDIF()
MESSAGE(STATUS "")
MESSAGE(STATUS "    Build type:                    ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "")
MESSAGE(STATUS "    Install path:                  ${CMAKE_INSTALL_PREFIX}")
MESSAGE(STATUS "")
IF(ICMAKER_USE_TCMALLOC AND Tcmalloc_FOUND)
  MESSAGE(STATUS "    Linking against tcmalloc:      ${Tcmalloc_LIBRARY}")
  MESSAGE(STATUS "")
ENDIF()
MESSAGE(STATUS "---- Scanning packages -------------------------------------------------------------")
MESSAGE(STATUS "")
