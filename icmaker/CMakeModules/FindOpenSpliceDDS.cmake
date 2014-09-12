# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# - Try to find OpenSpliceDDS
# Once done, this will define
#
#  OpenSpliceDDS_FOUND - system has OpenSpliceDDS
#  OpenSpliceDDS_INCLUDE_DIRS - the OpenSpliceDDS include directories
#  OpenSpliceDDS_LIBRARIES - link these to use OpenSpliceDDS

include(PrintLibraryStatus)
include(LibFindMacros)

IF ("${OSPL_HOME}" STREQUAL "")
  SET(OSPL_HOME $ENV{OSPL_HOME})
endif ()

IF (NOT "${OSPL_HOME}" STREQUAL "")
  STRING(REGEX REPLACE "\\\\" "/" OSPL_HOME ${OSPL_HOME})
    
  SET(OpenSpliceDDS_SEARCH_INCLUDE_PATHS ${OSPL_HOME}/include/dcps/C++/SACPP;${OSPL_HOME}/include/dcps/C/SAC;${OSPL_HOME}/include;${OSPL_HOME}/include/sys)
  SET(OpenSpliceDDS_SEARCH_LIB_PATHS ${OSPL_HOME}/lib)
ENDIF ()

SET(OpenSpliceDDS_SEARCH_LIBS ddsdatabase;ddsgapi;ddssacpp;ddsos)	

find_path(
  OpenSpliceDDS_INCLUDE_DIR
  NAMES ccpp_dds_dcps.h
  PATHS ${OpenSpliceDDS_SEARCH_INCLUDE_PATHS})
	
find_library(
  OpenSpliceDDS_LIBRARY_ddsos
  NAMES ddsos
  PATHS ${OpenSpliceDDS_SEARCH_LIB_PATHS})
find_library(
  OpenSpliceDDS_LIBRARY_ddsdatabase
  NAMES ddsdatabase
  PATHS ${OpenSpliceDDS_SEARCH_LIB_PATHS})
find_library(
  OpenSpliceDDS_LIBRARY_dcpsgapi
  NAMES dcpsgapi
  PATHS ${OpenSpliceDDS_SEARCH_LIB_PATHS})
find_library(
  OpenSpliceDDS_LIBRARY_dcpssacpp
  NAMES dcpssacpp
  PATHS ${OpenSpliceDDS_SEARCH_LIB_PATHS})
find_library(
  OpenSpliceDDS_LIBRARY_cmxml
  NAMES cmxml
  PATHS ${OpenSpliceDDS_SEARCH_LIB_PATHS})

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(OpenSpliceDDS_PROCESS_INCLUDES OpenSpliceDDS_SEARCH_INCLUDE_PATHS)
set(OpenSpliceDDS_PROCESS_LIBS
    OpenSpliceDDS_LIBRARY_ddsos
    OpenSpliceDDS_LIBRARY_ddsdatabase
    OpenSpliceDDS_LIBRARY_dcpsgapi
    OpenSpliceDDS_LIBRARY_dcpssacpp
    OpenSpliceDDS_LIBRARY_cmxml)
libfind_process(OpenSpliceDDS)

PRINT_LIBRARY_STATUS(OpenSpliceDDS
  DETAILS "[${OpenSpliceDDS_LIBRARIES}][${OpenSpliceDDS_INCLUDE_DIRS}]"
)

if (OpenSpliceDDS_FOUND)

    set(OpenSpliceDDS_RUNTIME_HINTS HINTS ${OSPL_HOST_HOME}/bin ${OSPL_HOME}/bin)
    
    find_program(IDLPP_EXECUTABLE "idlpp" ${OpenSpliceDDS_RUNTIME_HINTS})
    if (IDLPP_EXECUTABLE)
        MACRO(ICMAKER_ADD_DDS_IDL)
            PARSE_ARGUMENTS(IDLPP
            "IMPORT_EXPORT_DEFINE;IMPORT_EXPORT_HEADER"
            ""
            ${ARGN}
            )

            FOREACH (CURRENT_FILE ${IDLPP_DEFAULT_ARGS})
                GET_FILENAME_COMPONENT(outfilename ${CURRENT_FILE} NAME_WE)
                GET_FILENAME_COMPONENT(infile ${CURRENT_FILE} ABSOLUTE)
                GET_FILENAME_COMPONENT(idl_path ${infile} PATH)
                
                SET(hdr
                    ${CMAKE_CURRENT_BINARY_DIR}/ccpp_${outfilename}.h
                    ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}.h
                    ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}Dcps.h
                    ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}Dcps_impl.h
                    ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}SplDcps.h)
                SET(src
                    ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}Dcps.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}Dcps_impl.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}SplDcps.cpp)
                
                SET(IDLPP_FLAGS "-I${idl_path}")
                FOREACH (inc_dir ${${icmaker_target}_MACRO_EXPORT_IDL_DIRS})
                    LIST(APPEND IDLPP_FLAGS "-I${inc_dir}")
                ENDFOREACH ()
                
                if (WIN32)
                    if (NOT "${IDLPP_IMPORT_EXPORT_DEFINE}" STREQUAL "" AND NOT "${IDLPP_IMPORT_EXPORT_HEADER}" STREQUAL "")
                        LIST(APPEND IDLPP_FLAGS -P ${IDLPP_IMPORT_EXPORT_DEFINE},${IDLPP_IMPORT_EXPORT_HEADER})
                    elseif (NOT "${IDLPP_IMPORT_EXPORT_DEFINE}" STREQUAL "")
                        LIST(APPEND IDLPP_FLAGS -P ${IDLPP_IMPORT_EXPORT_DEFINE})
                    endif ()
                endif ()

                ADD_CUSTOM_COMMAND(
                    OUTPUT ${hdr} ${src}
                    COMMAND ${IDLPP_EXECUTABLE}
                    ARGS -S -l cpp -d ${CMAKE_CURRENT_BINARY_DIR} ${IDLPP_FLAGS} ${infile}
                    MAIN_DEPENDENCY ${infile}
                    COMMENT "Compiling OpenSpliceDDS IDL file ${CURRENT_FILE}")
                
                LIST(APPEND ${icmaker_target}_GENERATED_HEADERS ${hdr})
                LIST(APPEND ${icmaker_target}_GENERATED_SOURCES ${src})
                
                SOURCE_GROUP("DDS IDL" FILES ${CURRENT_FILE})
            ENDFOREACH (CURRENT_FILE)
        ENDMACRO(ICMAKER_ADD_DDS_IDL)
    endif (IDLPP_EXECUTABLE)
    
else (OpenSpliceDDS_FOUND)
	MACRO(ICMAKER_ADD_DDS_IDL)
	ENDMACRO(ICMAKER_ADD_DDS_IDL)
endif (OpenSpliceDDS_FOUND)
