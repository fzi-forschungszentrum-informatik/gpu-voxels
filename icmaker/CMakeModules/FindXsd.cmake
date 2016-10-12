# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# - Try to find code synthesis xsd (http://www.codesynthesis.com/products/xsd/)
# Once done, this will define
#
#  XSD_FOUND - system has xsd
#  XSD_INCLUDE_DIRS - the xsd include directories
#  XSD_LIBRARIES - link these to use xsd

include(PrintLibraryStatus)
include(LibFindMacros)

SET(XSD_FOUND FALSE)

find_path(XSD_INCLUDE_DIR NAMES version.hxx
  PATHS /usr/include/xsd/cxx
)

find_library(XSD_LIBRARIES NAMES libxerces-c.so
  PATHS /usr/lib
)

set(XSD_PROCESS_LIBS ${XSD_LIBRARIES})
libfind_process(XSD)

if(XSD_INCLUDE_DIR AND XSD_LIBRARIES)
  SET(XSD_FOUND TRUE)
else(XSD_INCLUDE_DIR AND XSD_LIBRARIES)
  SET(XSD_INCLUDE_DIR "xsd-include-NOTFOUND" CACHE PATH "xsd include path")
endif(XSD_INCLUDE_DIR AND XSD_LIBRARIES)

PRINT_LIBRARY_STATUS(XSD
  DETAILS "[${XSD_LIBRARIES}][${XSD_INCLUDE_DIRS}]"
)

if(XSD_FOUND)

  if(WIN32)
    SET(__XSD_NAME xsd.exe)
  else(WIN32)
    SET(__XSD_NAME xsdcxx)
  endif(WIN32)

  set(XSD_RUNTIME_HINTS HINTS ${XSD_DIR}/bin /usr/bin /usr/local/bin)
  find_program(XSD_EXECUTABLE NAMES ${__XSD_NAME} PATHS ${XSD_RUNTIME_HINTS})

  if(XSD_EXECUTABLE)
    MACRO(ICMAKER_GENERATE_XSD_CLASSES)
      PARSE_ARGUMENTS(XSDPP
        "ROOT_ELEMENT;NAMESPACES;GENERATE_OSTREAM;GENERATE_DOXYGEN;GENERATE_COMPARISON;GENERATE_DEFAULT_CTOR;GENERATE_FROM_BASE_CTOR"
        ""
        ${ARGN}
      )

      # Set xsd root element, if parameter was specified
      IF(XSDPP_ROOT_ELEMENT)
        SET(XSD_ROOT_ELEMENT "--root-element" ${XSDPP_ROOT_ELEMENT})
      ELSE(XSDPP_ROOT_ELEMENT)
        SET(XSD_ROOT_ELEMENT "--root-element-all")
      ENDIF(XSDPP_ROOT_ELEMENT)

      # Set suffix for generated source files
      SET(HEADER_SUFFIX "_xsd.h")
      SET(SOURCE_SUFFIX "_xsd.cpp")

      # Set arguments for xsd command
      SET(XSD_ARGS "cxx-tree")
      LIST(APPEND XSD_ARGS "--hxx-suffix" ${HEADER_SUFFIX})
      LIST(APPEND XSD_ARGS "--cxx-suffix" ${SOURCE_SUFFIX})
      LIST(APPEND XSD_ARGS ${XSD_ROOT_ELEMENT})
      LIST(APPEND XSD_ARGS "--generate-serialization")
      LIST(APPEND XSD_ARGS "--generate-polymorphic")

      # Set optional arguments
      IF(XSDPP_GENERATE_OSTREAM)
        LIST(APPEND XSD_ARGS "--generate-ostream")
      ENDIF(XSDPP_GENERATE_OSTREAM)
      IF(XSDPP_GENERATE_DOXYGEN)
        LIST(APPEND XSD_ARGS "--generate-doxygen")
      ENDIF(XSDPP_GENERATE_DOXYGEN)
      IF(XSDPP_GENERATE_COMPARISON)
        LIST(APPEND XSD_ARGS "--generate-comparison")
      ENDIF(XSDPP_GENERATE_COMPARISON)
      IF(XSDPP_GENERATE_DEFAULT_CTOR)
        LIST(APPEND XSD_ARGS "--generate-default-ctor")
      ENDIF(XSDPP_GENERATE_DEFAULT_CTOR)
      IF(XSDPP_GENERATE_FROM_BASE_CTOR)
        LIST(APPEND XSD_ARGS "--generate-from-base-ctor")
      ENDIF(XSDPP_GENERATE_FROM_BASE_CTOR)

      # Add arguments for namespace mapping (xns to cns)
      FOREACH(NAMESPACE ${XSDPP_NAMESPACES})
        #MESSAGE(STATUS "NAMESPACE ${NAMESPACE}")
        LIST(APPEND XSD_ARGS "--namespace-map" ${NAMESPACE})
      ENDFOREACH(NAMESPACE)

      FOREACH(xsd_file ${XSDPP_DEFAULT_ARGS})
        GET_FILENAME_COMPONENT(outfilename ${xsd_file} NAME_WE)
        GET_FILENAME_COMPONENT(infile ${xsd_file} ABSOLUTE)
        GET_FILENAME_COMPONENT(xsd_path ${infile} PATH)

        SET(hdr ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}${HEADER_SUFFIX})
        SET(src ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}${SOURCE_SUFFIX})

        #MESSAGE(STATUS "execute command: ${XSD_EXECUTABLE} ${XSD_ARGS} --output-dir ${CMAKE_CURRENT_BINARY_DIR} ${infile}")
        ADD_CUSTOM_COMMAND(OUTPUT ${src} ${hdr}
                           COMMAND ${XSD_EXECUTABLE} ${XSD_ARGS} "--output-dir" ${CMAKE_CURRENT_BINARY_DIR} ${infile}
                           DEPENDS ${infile}
                           COMMENT "Generating sources from XSD file ${infile}"
        )

        LIST(APPEND ${icmaker_target}_GENERATED_HEADERS ${hdr})
        LIST(APPEND ${icmaker_target}_GENERATED_SOURCES ${src})
      ENDFOREACH(xsd_file)

    ENDMACRO(ICMAKER_GENERATE_XSD_CLASSES)
  endif(XSD_EXECUTABLE)

else(XSD_FOUND)
  MACRO(ICMAKER_GENERATE_XSD_CLASSES)
    MESSAGE(STATUS "XSD not found -> ICMAKER_GENERATE_XSD_CLASSES MACRO is empty! Please check if the following packages are installed on your system: xsdcxx libxerces-c-dev.")
  ENDMACRO(ICMAKER_GENERATE_XSD_CLASSES)
endif(XSD_FOUND)
