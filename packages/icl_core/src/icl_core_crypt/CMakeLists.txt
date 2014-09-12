# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("icl_core_crypt" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  Sha2.h
  Sha2.hpp
  Sha2Impl.h
  Sha2Impl.hpp
  )

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  )

ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_ICL_CORE_CRYPT_)

ICMAKER_BUILD_ANNOUNCEMENT()
ICMAKER_INSTALL_HEADERS(icl_core_crypt)
