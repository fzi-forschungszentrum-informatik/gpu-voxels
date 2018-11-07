# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# Copyright (c) 2018, FZI Forschungszentrum Informatik
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
#    conditions and the following disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- END LICENSE BLOCK ------------------------------------------------

# Handles generation of MCA description files.

# We use an evil hack here: The path the description builder executable is
# set as a global cmake variable during build of the the binary. This allows
# us there to handle special situations like cross compiling. 
# 
# See CMakeLists in mcal_kernel/src/description_builder for information
# about cmake variables according to the behaviour of this.
# In short, if we compile for the same system all should be fine. And a
# special export file will generated. 
# If we cross compile you must provide an export file from a native build in
# the variable MCA_DESCRIPTION_BUILDER_CMAKE_FILE. If you do this
# the native binary will be used for generation. If something is wrong
# and we are cross compiling and you definitely want to create the binary
# use MCA_DESCRIPTION_BUILDER_OVERRIDE_CROSS_COMPILATION. Note that it is
# really unlikely that you want this.
#

IF (MCA_DESCRIPTION_BUILDER_EXECUTABLE)
  MACRO (MCA_ADD_HEADER_DESCRIPTIONS)
    FOREACH (CURRENT_FILE ${ARGN})
      #message ("header descr ${CURRENT_FILE}")
      GET_FILENAME_COMPONENT(outfilename ${CURRENT_FILE} NAME_WE)
      GET_FILENAME_COMPONENT(infilename ${CURRENT_FILE} NAME)
      GET_FILENAME_COMPONENT(infile ${CURRENT_FILE} ABSOLUTE)
      SET (outfile ${CMAKE_CURRENT_BINARY_DIR}/descr_h_${outfilename}.cpp)
      
      ADD_CUSTOM_COMMAND(
        OUTPUT "${outfile}"
        COMMAND "${MCA_DESCRIPTION_BUILDER_EXECUTABLE}"
              ARGS "${infile}" "${outfile}"
              DEPENDS "${infile}" "${MCA_DESCRIPTION_BUILDER_EXECUTABLE}" # depends on the 'processor'
              COMMENT "Generating descriptions for ${infilename}")
      LIST(APPEND ${icmaker_target}_GENERATED_SOURCES "${outfile}")
    ENDFOREACH ()
  ENDMACRO ()

  MACRO (MCA_ADD_SOURCE_DESCRIPTIONS)
    FOREACH (CURRENT_FILE ${ARGN})
      #message ("source descr ${CURRENT_FILE}")
      GET_FILENAME_COMPONENT(outfilename ${CURRENT_FILE} NAME_WE)
      GET_FILENAME_COMPONENT(infilename ${CURRENT_FILE} NAME)
      GET_FILENAME_COMPONENT(infile ${CURRENT_FILE} ABSOLUTE)
      SET (outfile ${CMAKE_CURRENT_BINARY_DIR}/descr_C_${outfilename}.cpp)
      
      ADD_CUSTOM_COMMAND(
        OUTPUT "${outfile}"
        COMMAND "${MCA_DESCRIPTION_BUILDER_EXECUTABLE}"
              ARGS "${infile}" "${outfile}"
              DEPENDS "${infile}" "${MCA_DESCRIPTION_BUILDER_EXECUTABLE}" # depends on the 'processor'
              COMMENT "Generating descriptions for ${infilename}")
      LIST(APPEND ${icmaker_target}_GENERATED_SOURCES "${outfile}")
    ENDFOREACH ()
  ENDMACRO ()

  MACRO (MCA_ADD_TEMPLATE_DESCRIPTIONS)
    FOREACH (CURRENT_FILE ${ARGN})
      #message ("template descr ${CURRENT_FILE}")
      GET_FILENAME_COMPONENT(outfilename ${CURRENT_FILE} NAME_WE)
      GET_FILENAME_COMPONENT(infilename ${CURRENT_FILE} NAME)
      GET_FILENAME_COMPONENT(infile ${CURRENT_FILE} ABSOLUTE)
      SET (outfile ${CMAKE_CURRENT_BINARY_DIR}/descr_h_${outfilename}.hpp)
      
      ADD_CUSTOM_COMMAND(
        OUTPUT "${outfile}"
        COMMAND "${MCA_DESCRIPTION_BUILDER_EXECUTABLE}"
              ARGS "${infile}" "${outfile}"
              DEPENDS "${infile}" "${MCA_DESCRIPTION_BUILDER_EXECUTABLE}" # depends on the 'processor'
              COMMENT "Generating template descriptions for ${infilename}")
      LIST(APPEND ${icmaker_target}_GENERATED_HEADERS "${outfile}")
    ENDFOREACH ()
  ENDMACRO ()
ELSE ()
  MESSAGE(STATUS "Warning: mca description builder not found.")
  MACRO (MCA_ADD_HEADER_DESCRIPTIONS)
    MESSAGE(STATUS "Error:   mca description builder not found, but used.")
  ENDMACRO ()
  MACRO (MCA_ADD_SOURCE_DESCRIPTIONS)
    MESSAGE(STATUS "Error:   mca description builder not found, but used.")
  ENDMACRO ()
  MACRO (MCA_ADD_TEMPLATE_DESCRIPTIONS)
    MESSAGE(STATUS "Error:   mca description builder not found, but used.")
  ENDMACRO ()
ENDIF ()

MACRO(MCA_GENERATE_DESCRIPTIONS)
  SET (search_files ${${icmaker_target}_SOURCES} ${${icmaker_target}_HEADERS})
  FOREACH (current_file ${search_files})

    GET_FILENAME_COMPONENT(abs_file ${current_file} ABSOLUTE)

    IF (EXISTS ${abs_file})

      FILE (READ ${abs_file} contents)

      GET_FILENAME_COMPONENT (file_ext ${current_file} EXT)
      
      IF ((file_ext STREQUAL ".h") OR (file_ext STREQUAL ".hpp"))
        STRING (REGEX MATCH "[^A-Za-z0-9]_?DESCR[^A-Za-z0-9]" match "${contents}")
        IF (match)
          MCA_ADD_HEADER_DESCRIPTIONS ("${current_file}")
        ENDIF ()
      ELSEIF (file_ext STREQUAL ".cpp")
        STRING (REGEX MATCH "[^A-Za-z0-9]_?DESCR[^A-Za-z0-9]" match "${contents}")
        IF (match)
          MCA_ADD_SOURCE_DESCRIPTIONS ("${current_file}")
        ENDIF ()
      ENDIF ()
    ENDIF ()
  ENDFOREACH ()
ENDMACRO()
