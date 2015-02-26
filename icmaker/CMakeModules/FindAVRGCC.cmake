# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Soeren Bohn <bohn@fzi.de>
# \date    2014-10-27
#
# Try to find the avr-gcc compiler.
#
# Defines the following variables:
#  avrgcc_FOUND            Compiler found
#  AVRGCC_CXX_COMPILER     Path to avr-g++ compiler
#  AVRGCC_C_COMPILER       Path to avr-cc compiler
#  AVRGCC_OBJCOPY          Path to avr-libc objcopy command
#  AVRGCC_AVRDUDE          Path to avrdude
#  ICMAKER_AVR_OUTPUT_DIR  Path to hex file storage
#
# Configurable variables:
#  ICMAKER_AVR_AVRDUDE_PROGRAMMER      use this programmer, default: avrisp2
#  ICMAKER_AVR_AVRDUDE_PORT            use this port, default: /dev/ttyUSB0
#  ICMAKER_AVR_AVRDUDE_EXTRA_OPTIONS   additional options for avrdude
#
# Exposes the following macros:
#  ICMAKER_BUILD_AVR_HEX   Build an avr targets as hex.
#          All generated programs will be stored in the hex
#          subdirectory of the build directory and can be
#          programmed directly using avrdude (e.g.
#            avrdude avrdude -p m8 -c avrisp2 -U flash:w:hex/target.hex -P /dev/ttyUSB0
#          ). See avrdude command reference for details.
#          This macro tooks the following addiotional parameters:
#            MCU <mcu_type (e.g. atmega8)>
#               sets the -mmcu commandline argument, required
#            F_CPU <crystal clock (e.g. 4000000 for 4 MHz)>
#               defines the clockspeed preprocessor makro
#            OPTIMIZE <one of 0,1,2,3,s>
#               sets the optimizer level, default: -Os
#            EEPROM
#               generate eep file
#          The build currently ignores all global defines and
#          only obeys the ICMAKER_LOCAL_CPPDEFINES macro. Also
#          the icmaker dependency management is ignored for now. So one
#          must include all needed files for a working program and it
#          exists no macro for library creation.
#          When the avrdude program is found another target
#          "<target_name>-program" is defined which uploads the
#          program with an attached programmer.
#
#  ICMAKER_AVR_FUSE      Define needed fuse settings
#          Store the needed fuse settings along with the target.
#          When avrdude was found this generates an additional target
#          "<target_name>-fuse" which calls avrdude to fuse the device.
#          This macro tooks the following parameters: H, L each followed
#          by a hexadecimal number to call avrdude.
#
#
#
#
# Example (will create a hex file avr_test_uc.hex suitable to program an
#          atmega8 controller running at 4 MHz):
#  ICMAKER_SET("avr_test_uc" IDE_FOLDER ${MCAP_AVR_PROJECT})
#
#  ICMAKER_ADD_SOURCES(
#    uc_prog.cpp
#    UART.cpp
#    plain_c_file.c
#  )
#
#  ICMAKER_ADD_HEADERS(
#    UART.hpp
#  )
#
#  ICMAKER_LOCAL_CPPDEFINES(-funsigned-char -funsigned-bitfields
#                           -fpack-struct -fshort-enums
#                           -Wall -Wstrict-prototypes -std=gnu99)
#
#  ICMAKER_AVR_FUSE(L DE H D9)
#
#  ICMAKER_BUILD_AVR_HEX(MCU atmega8 F_CPU 4000000 OPTIMIZE 3)
#
#----------------------------------------------------------------------


find_program(AVRGCC_CXX_COMPILER avr-g++)
find_program(AVRGCC_C_COMPILER avr-gcc)
find_program(AVRGCC_OBJCOPY avr-objcopy)
find_program(AVRGCC_AVRDUDE avrdude)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(avrgcc
                                  FOUND_VAR avrgcc_FOUND
                                  REQUIRED_VARS AVRGCC_CXX_COMPILER AVRGCC_C_COMPILER AVRGCC_OBJCOPY)


if(avrgcc_FOUND)
    set(ICMAKER_AVR_OUTPUT_DIR ${CMAKE_BINARY_DIR}/hex)
    file(MAKE_DIRECTORY ${ICMAKER_AVR_OUTPUT_DIR})

    if(NOT ("${AVRGCC_AVRDUDE}" STREQUAL "AVRGCC_AVRDUDE-NOTFOUND"))
        set(ICMAKER_AVR_AVRDUDE_PROGRAMMER "avrisp2" CACHE STRING "avrdude: used programmer")
        set(ICMAKER_AVR_AVRDUDE_PORT "/dev/ttyUSB0" CACHE STRING "avrdude: used port")
        set(ICMAKER_AVR_AVRDUDE_EXTRA_OPTIONS "" CACHE STRING "avrdude: add extra options")
    endif()

    macro(ICMAKER_BUILD_AVR_HEX)
        # parse options
        parse_arguments(ICMAKER_AVR_BUILD_OPT "MCU;F_CPU;OPTIMIZE" "EEPROM" ${ARGN})
        if(ICMAKER_AVR_BUILD_OPT_MCU)
            set(${icmaker_target}_MCU "-mmcu=${ICMAKER_AVR_BUILD_OPT_MCU}")
        else()
            if(NOT ("${${icmaker_target}_MACRO_DEFINITIONS}" MATCHES "-mmcu"))
                message(STATUS "Error:   ${icmaker_target} -- no mcu type given")
            endif()
        endif()
        if(ICMAKER_AVR_BUILD_OPT_F_CPU)
            set(${icmaker_target}_F_CPU "-DF_CPU=${ICMAKER_AVR_BUILD_OPT_F_CPU}")
        endif()
        if(ICMAKER_AVR_BUILD_OPT_OPTIMIZE)
            if("${ICMAKER_AVR_BUILD_OPT_OPTIMIZE}" MATCHES "^(0|1|2|3|s)$")
                set(${icmaker_target}_OPTIMIZE "-O${ICMAKER_AVR_BUILD_OPT_OPTIMIZE}")
            else()
                message(STATUS "Warning: ${icmaker_target} -- unknown optimizer setting '${ICMAKER_AVR_BUILD_OPT_OPTIMIZE}', defaulting to -Os")
                set(${icmaker_target}_OPTIMIZE "-Os")
            endif()
        else()
            set(${icmaker_target}_OPTIMIZE "-Os")
        endif()

        # ensure headers are added
        add_custom_target(${icmaker_target}_header_target SOURCES ${${icmaker_target}_HEADERS})

        # code generation
        foreach(_project_file ${${icmaker_target}_SOURCES})
            get_filename_component(_extension "${_project_file}" EXT)
            string(SUBSTRING "${_extension}" 1 -1 _extension)
            list(FIND CMAKE_C_SOURCE_FILE_EXTENSIONS "${_extension}" _cc)
            set(_compiler "${AVRGCC_CXX_COMPILER}")
            if(NOT (_cc STREQUAL "-1"))
                set(_compiler "${AVRGCC_C_COMPILER}")
            endif()

            add_custom_command(OUTPUT ${_project_file}.o
                               COMMAND ${_compiler}
                               ARGS -c ${${icmaker_target}_MCU}
                                    ${${icmaker_target}_OPTIMIZE}
                                    ${${icmaker_target}_F_CPU}
                                    ${${icmaker_target}_MACRO_DEFINITIONS}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/${_project_file}
                                    -o ${_project_file}.o
                               MAIN_DEPENDENCY ${_project_file}
                               DEPENDS ${${icmaker_target}_HEADERS}
                               COMMENT "Building AVR object ${_project_file}.o" VERBATIM)
            list(APPEND ${icmaker_target}_AVR_COMMANDS "${_project_file}.o")
            unset(_compiler)
            unset(_cc)
        endforeach()
        # elf target
        add_custom_command(OUTPUT ${icmaker_target}.elf
                           COMMAND ${AVRGCC_C_COMPILER}
                           ARGS -g ${${icmaker_target}_MCU} ${${icmaker_target}_OPTIMIZE} ${${icmaker_target}_F_CPU} ${${icmaker_target}_MACRO_DEFINITIONS} ${${icmaker_target}_AVR_COMMANDS} -o ${icmaker_target}.elf
                           DEPENDS ${${icmaker_target}_AVR_COMMANDS}
                           COMMENT "Linking AVR executable ${icmaker_target}.elf" VERBATIM)
        # hex target (flash)
        add_custom_command(OUTPUT ${icmaker_target}.hex
                          COMMAND ${AVRGCC_OBJCOPY}
                          ARGS -O ihex -R .eeprom
                               ${icmaker_target}.elf
                               ${ICMAKER_AVR_OUTPUT_DIR}/${icmaker_target}.hex
                          MAIN_DEPENDENCY ${icmaker_target}.elf
                          COMMENT "Generating AVR hex file ${icmaker_target}.hex" VERBATIM)
        # eep target (eeprom)
        if(${ICMAKER_AVR_BUILD_OPT_EEPROM})
            add_custom_command(OUTPUT ${icmaker_target}.eep
                              COMMAND ${AVRGCC_OBJCOPY}
                              ARGS -j .eeprom --set-section-flags=.eeprom=alloc,load --change-section-lma .eeprom=0 -O ihex
                                   ${icmaker_target}.elf
                                   ${ICMAKER_AVR_OUTPUT_DIR}/${icmaker_target}.eep
                              MAIN_DEPENDENCY ${icmaker_target}.elf
                              COMMENT "Generating AVR eep file ${icmaker_target}.eep" VERBATIM)
        endif()

        # program device target
        if(NOT ("${AVRGCC_AVRDUDE}" STREQUAL "AVRGCC_ACRDUDE-NOTFOUND"))
            set(_flash -U flash:w:${ICMAKER_AVR_OUTPUT_DIR}/${icmaker_target}.hex)
            if(${ICMAKER_AVR_BUILD_OPT_EEPROM})
                set(_eeprom -U eeprom:w:${ICMAKER_AVR_OUTPUT_DIR}/${icmaker_target}.eep)
            endif()
            set(_avrdude_params "")
            set(_avrdude_params ${_avrdude_params} "-p" ${ICMAKER_AVR_BUILD_OPT_MCU})
            set(_avrdude_params ${_avrdude_params} "-c" "${ICMAKER_AVR_AVRDUDE_PROGRAMMER}")
            if(ICMAKER_AVR_AVRDUDE_PORT)
                set(_avrdude_params ${_avrdude_params} "-P" "${ICMAKER_AVR_AVRDUDE_PORT}")
            endif()
            set(_avrdude_params ${_avrdude_params} ${ICMAKER_AVR_AVRDUDE_EXTRA_OPTIONS})
            add_custom_target(${icmaker_target}-program
                              COMMAND ${AVRGCC_AVRDUDE} ${_avrdude_params} ${_flash} ${_eeprom}
                              DEPENDS ${icmaker_target}.hex
                              COMMENT "Programming ${icmaker_target}"
                              VERBATIM)
            if(${icmaker_target}_AVR_FUSE)
                add_custom_target(${icmaker_target}-fuse
                                  COMMAND ${AVRGCC_AVRDUDE} ${_avrdude_params} ${${icmaker_target}_AVR_FUSE}
                                  DEPENDS ${icmaker_target}.hex
                                  COMMENT "Programming ${icmaker_target}"
                                  VERBATIM)
            endif()
        endif()

        # build it with all
        if(${ICMAKER_AVR_BUILD_OPT_EEPROM})
            add_custom_target(${icmaker_target} ALL DEPENDS ${icmaker_target}.hex ${icmaker_target}.eep)
        else()
            add_custom_target(${icmaker_target} ALL DEPENDS ${icmaker_target}.hex)
        endif()
    endmacro()

    macro(ICMAKER_AVR_FUSE)
        # parse options
        parse_arguments(_options "L;H" "" ${ARGN})
        if(_options_L)
            set(_fuse_l "-U lfuse:w:0x${_options_L}:m")
        else()
            message(STATUS "Warning: lfuse not set for target ${icmaker_target}")
        endif()
        if(_options_H)
            set(_fuse_h "-U hfuse:w:0x${_options_H}:m")
        else()
            message(STATUS "Warning: hfuse not set for target ${icmaker_target}")
        endif()
        set(${icmaker_target}_AVR_FUSE "${_fuse_l} ${_fuse_h}")
    endmacro()

else()
    message(STATUS "Warning: avrgcc not found, won't build avr targets.")
    macro(ICMAKER_BUILD_AVR_HEX)
        message(STATUS "Warning: ignoring build avr command for target ${icmaker_target}")
    endmacro()
    macro(ICMAKER_AVR_FUSE)
    endmacro()
endif()
