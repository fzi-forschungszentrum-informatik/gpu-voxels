#from https://raw.githubusercontent.com/highfidelity/qca/master/cmake/modules/FindQt5Transitional.cmake

find_package(Qt5Core QUIET)

# When there is a QT installation found (checked by QtCore package), then
# proceed in finding the different components of qt5.
if (Qt5Core_FOUND)
  # set the components that should be found by the transitional script.
  set(_components
      Core
      Gui
      DBus
      Designer
      # Declarative
      Script
      ScriptTools
      Network
      Test
      Xml
      Svg
      Sql
      Widgets
      PrintSupport
      Concurrent
      UiTools
      WebKit
      WebKitWidgets
      OpenGL
      Widgets
    )

  # call find_package for each of the components.
  foreach(_component ${_components})
    message(STATUS "Searching Qt component ${_component}...")

    find_package(Qt5${_component})

    if(Qt5${_component}_FOUND)
      message(STATUS "-->Qt5${_component} has been found!")
    endif()

    list(APPEND QT_LIBRARIES ${Qt5${_component}_LIBRARIES})
  endforeach()

  # Now for downwards compatibility, set all QT_{component}-Variables from
  # QT4, since icmaker depends on theses variables.
  foreach(_component ${_components})
    find_package(Qt5${_component} REQUIRED)
    string(TOUPPER ${_component} _componentUpper)

    set(QT_QT${_componentUpper}_FOUND TRUE)
    set(QT_QT${_componentUpper}_INCLUDE_DIRS ${Qt5${_component}_INCLUDE_DIRS})
    set(QT_QT${_componentUpper}_LIBRARIES Qt5::${_component})

    if ("${_component}" STREQUAL "WebKit")
      find_package(Qt5WebKitWidgets REQUIRED)
      list(APPEND QT_LIBRARIES ${Qt5WebKitWidgets_LIBRARIES} )
    endif()
    if ("${_component}" STREQUAL "Gui")
      find_package(Qt5Widgets REQUIRED)
      find_package(Qt5PrintSupport REQUIRED)
      find_package(Qt5Svg REQUIRED)
      list(APPEND QT_LIBRARIES ${Qt5Widgets_LIBRARIES}
                               ${Qt5PrintSupport_LIBRARIES}
                               ${Qt5Svg_LIBRARIES} )
    endif()
  endforeach()

  set(Qt5Transitional_FOUND TRUE)
  set(QT5_FOUND TRUE)  # deprecated
  set(Qt5_FOUND TRUE)  # better
  set(QT_FOUND TRUE)
  set(QT5_BUILD TRUE)

  # Temporary until upstream does this:
  foreach(_component ${_components})
    if (TARGET Qt5::${_component})
      set_property(TARGET Qt5::${_component}
        APPEND PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES ${Qt5${_component}_INCLUDE_DIRS})
      set_property(TARGET Qt5::${_component}
        APPEND PROPERTY
          INTERFACE_COMPILE_DEFINITIONS ${Qt5${_component}_COMPILE_DEFINITIONS})
    endif()
  endforeach()

  set_property(TARGET Qt5::Core
        PROPERTY
          INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  if (WIN32 AND NOT Qt5_NO_LINK_QTMAIN)
      set(_isExe $<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>)
      set(_isWin32 $<BOOL:$<TARGET_PROPERTY:WIN32_EXECUTABLE>>)
      set(_isNotExcluded $<NOT:$<BOOL:$<TARGET_PROPERTY:Qt5_NO_LINK_QTMAIN>>>)
      get_target_property(_configs Qt5::Core IMPORTED_CONFIGURATIONS)
      foreach(_config ${_configs})
          set_property(TARGET Qt5::Core APPEND PROPERTY
              IMPORTED_LINK_INTERFACE_LIBRARIES_${_config}
                  $<$<AND:${_isExe},${_isWin32},${_isNotExcluded}>:Qt5::WinMain>
          )
      endforeach()
      unset(_configs)
      unset(_isExe)
      unset(_isWin32)
      unset(_isNotExcluded)
  endif()
  # End upstreamed stuff.

  get_filename_component(_modules_dir "${CMAKE_CURRENT_LIST_DIR}/../modules" ABSOLUTE)
#  include("${_modules_dir}/ECMQt4To5Porting.cmake") # TODO: Port away from this.
  # include_directories(${QT_INCLUDES}) # TODO: Port away from this.

  if (Qt5_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  endif()
endif()
