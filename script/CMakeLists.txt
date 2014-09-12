IF (EXISTS "${CMAKE_SOURCE_DIR}/script/ic" AND IS_DIRECTORY "${CMAKE_SOURCE_DIR}/script/ic")
  ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/script/ic)
ENDIF ()

IF(WIN32)
  CONFIGURE_FILE("setenv.bat" "${CMAKE_CURRENT_BINARY_DIR}/setenv.bat" COPYONLY)
  CONFIGURE_FILE("open_cmd_debug.bat.in" "${CMAKE_CURRENT_BINARY_DIR}/open_cmd_debug.bat" COPYONLY)
  CONFIGURE_FILE("open_cmd_release.bat.in" "${CMAKE_CURRENT_BINARY_DIR}/open_cmd_release.bat" COPYONLY)
  CONFIGURE_FILE("open_devenv_debug.bat.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/open_devenv_debug.bat"
                 NEWLINE_STYLE CRLF)
  CONFIGURE_FILE("open_devenv_release.bat.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/open_devenv_release.bat"
                 NEWLINE_STYLE CRLF)
ELSE()
  CONFIGURE_FILE("setenv" "${CMAKE_CURRENT_BINARY_DIR}/setenv" COPYONLY)
ENDIF()
