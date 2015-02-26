# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("icl_core" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
    EnumHelper.cpp
    os_fs.cpp
    os_lxrt.cpp
    os_string.cpp
    Noncopyable.cpp
    SchemeParser.cpp
    SequenceNumber.cpp
    StringHelper.cpp
    TimeBase.cpp
    TimeSpan.cpp
    TimeStamp.cpp
    DataHeader.cpp
    )

ICMAKER_ADD_HEADERS(
    Array2D.h
    AtScopeEnd.h
    BaseTypes.h
    BitfieldHelper.h
    Convert.h
    DataHeader.h
    EnumHelper.h
    Explicit.h
    ExpectedType.h
    Finalizable.h
    icl_core.h
    ImportExport.h
    KeyValueDirectory.h
    KeyValueDirectory.hpp
    List.h
    Map.h
    msvc_inttypes.h
    msvc_stdint.h
    Multimap.h
    Noncopyable.h
    os_fs.h
    os_lxrt.h
    os_mem.h
    os_ns.h
    os_string.h
    os_thread.h
    os_time.h
    os.h
    Queue.h
    RemoveMemberPointer.h
    RingBuffer.h
    RingBuffer.hpp
    SchemeParser.h
    SearchableStack.h
    SequenceNumber.h
    Set.h
    SetHelper.h
    Singleton.h
    Singleton.hpp
    SingletonCreationPolicies.h
    SingletonLifetimePolicies.h
    SingletonThreadingModels.h
    StringHelper.h
    SystemString.h
    TemplateHelper.h
    TimeBase.h
    TimeSpan.h
    TimeStamp.h
    UnionFind.h
    Vector.h
    VectorWrapper.h
    ZBuffer.h
    )

IF(ICMAKER_DEPRECATED_STYLE)
ICMAKER_ADD_HEADERS(
    Deprecate.h
    tList.h
    tMap.h
    tNoncopyable.h
    tQueue.h
    tRingBuffer.h
    tSchemeParser.h
    tSearchableStack.h
    tSequenceNumber.h
    tString.h
    tTimeSpan.h
    tTimeStamp.h
    tVector.h
)
ENDIF()

IF(WIN32)
  ICMAKER_ADD_SOURCES(
      os_win32_fs.cpp
      os_win32_mem.cpp
      os_win32_string.cpp
      os_win32_thread.cpp
      os_win32_time.cpp
    )
    ICMAKER_ADD_HEADERS(
      os_win32_fs.h
      os_win32_mem.h
      os_win32_string.h
      os_win32_thread.h
      os_win32_time.h
    )    
ENDIF()

IF(UNIX OR APPLE)
  # TODO: make this a FindFoo.cmake
  ICMAKER_GLOBAL_CPPDEFINES(
    -D_IC_BUILDER_HAS_TIME_H_
    -D_IC_BUILDER_HAS_SYS_TIME_H_
  )
  
  ICMAKER_ADD_SOURCES(
    os_posix_fs.cpp
    os_posix_hacks.cpp
    os_posix_mem.cpp
    os_posix_string.cpp
    os_posix_thread.cpp
    os_posix_time.cpp
  )

  ICMAKER_ADD_HEADERS(
    os_posix_fs.h
    os_posix_mem.h
    os_posix_string.h
    os_posix_thread.h
    os_posix_time.h
  )
ENDIF()

ICMAKER_LOCAL_CPPDEFINES(-DICL_CORE_EXPORT_SYMBOLS)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_ICL_CORE_)
ICMAKER_INCLUDE_DIRECTORIES(${ICL_CORE_INCLUDE_DIRS})

ICMAKER_DEPENDENCIES(EXPORT
  Boost
  Boost_DATE_TIME
  Boost_REGEX

)

ICMAKER_DEPENDENCIES(OPTIONAL
  OpenSpliceDDS
  Zlib
)

ICMAKER_DEPENDENCIES(OPTIONAL EXPORT
  Eigen3
)

IF (OpenSpliceDDS_FOUND)
  ICMAKER_ADD_DDS_IDL(
    iTimeSpan.idl
    iTimeStamp.idl
    IMPORT_EXPORT_DEFINE ICL_CORE_IMPORT_EXPORT
    IMPORT_EXPORT_HEADER icl_core/ImportExport.h
    )
  ICMAKER_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
  ICMAKER_IDL_DIRECTORIES(${ICL_CORE_IDL_DIRS})
ENDIF ()

IF(ANDROID)
  ICMAKER_SYSTEM_DEPENDENCIES(-lgnustl_shared)
ENDIF()
ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(${icmaker_target})
