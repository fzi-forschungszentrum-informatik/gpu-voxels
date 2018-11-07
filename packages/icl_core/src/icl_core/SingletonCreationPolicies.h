// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2009-06-16
 *
 * \brief   Contains creation policies for icl_core::Singleton.
 *
 * \b icl_core::SCPCreateUsingNew uses operator new and operator delete.
 * \b icl_core::SCPCreateStatic keeps the instance as a static function
 * variable.
 * \b icl_core::SCPCreateUsingMalloc uses malloc() and free().
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_SINGLETON_CREATION_POLICIES_H_INCLUDED
#define ICL_CORE_SINGLETON_CREATION_POLICIES_H_INCLUDED

#include <stdlib.h>

namespace icl_core {

//! Creates and destroys objects using operator new and operator delete.
template
<class T>
class SCPCreateUsingNew
{
public:
  static T *create()
  {
    return new T;
  }

  static void destroy(T *object)
  {
    delete object;
  }
};

//! Creates objects as static function variables.
template
<class T>
class SCPCreateStatic
{
public:
  static T *create()
  {
    static T instance;
    return &instance;
  }

  static void destroy(T *object)
  {
  }
};

//! Creates and destroys objects using malloc() and free().
template
<class T>
class SCPCreateUsingMalloc
{
public:
  static T *create()
  {
    return static_cast<T *>(malloc(sizeof(T)));
  }

  static void destroy(T *object)
  {
    free(object);
  }
};


}

#endif
