# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

SET(TINYXML_IDE_FOLDER "tinyxml")

ICMAKER_REGISTER_PACKAGE(tinyxml)
ADD_SUBDIRECTORY(src)
ICMAKER_CONFIGURE_PACKAGE()
