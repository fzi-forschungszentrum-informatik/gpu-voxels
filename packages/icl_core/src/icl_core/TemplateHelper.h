// this is a -*- C++ -*- file

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file    TemplateHelper.h
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2005-09-29
 *
 * \brief   Helper definitions for template programming.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_TEMPLATE_HELPER_H_INCLUDED
#define ICL_CORE_TEMPLATE_HELPER_H_INCLUDED

namespace icl_core {

/*! Converts the type \a T to either a reference or a const
 *  reference. If \a T already is a reference it is left untouched.
 *
 *  \note You have to add a typename declaration if you use this
 *  converter.
 */
template <typename T>
struct ConvertToRef
{
  typedef const T& ToConstRef;
  typedef T& ToRef;
};
template <typename T>
struct ConvertToRef<T&>
{
  typedef T& ToConstRef;
  typedef T& ToRef;
};


/*! Provides a wrapper for default-constructing an object. This can be
 *  used for default-constructed default arguments in functions. If
 *  you directly use the default constructor of a template type the
 *  compiler will always complain if \a T has no default constructor,
 *  even if the default arguments are never used. If you use this
 *  wrapper the default constructor is only needed if the default
 *  arguments are used somewhere in the code. Otherwise \a T need not
 *  have a default constructor.
 */
template <typename T>
struct DefaultConstruct
{
  static T C()
  {
    return T();
  }
};

}

#endif
