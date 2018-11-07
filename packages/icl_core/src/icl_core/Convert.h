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
 * \author  Jan Oberlaender <oberlaender@fzi.de>
 * \date    2015-01-30
 *
 * Generic concept for the conversion of objects between two objects
 * of semantically compatible datatypes.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONVERT_H_INCLUDED
#define ICL_CORE_CONVERT_H_INCLUDED

namespace icl_core {

/*! Convert an object \a from to an object \a to which has a
 *  semantically compatible datatype.  This concept is designed to
 *  help get rid of the multitude of custom, signature-incompatible,
 *  badly documented conversion routines, and provide the user of your
 *  datatypes with a single, convenient way of converting between
 *  different semantically compatible types.  The default
 *  implementation here simply uses static_cast.  If you want to
 *  convert between two datatypes, such as different types of 3d
 *  transforms, a custom conversion method may be necessary and can be
 *  implemented by template specialization of this function.  Say, for
 *  example, you want to provide a converter from \c Eigen::Affine3d
 *  to your own custom 3D transform datatype (we'll call it \c
 *  CustomTransform).  Then you implement the following
 *  specialization:
 *  \code
 *  template <>
 *  void convert<CustomTransform, Eigen::Affine3d>(const Eigen::Affine3d& from, CustomTransform& to)
 *  {
 *    // Your implementation goes here.
 *  }
 *  \endcode
 *  Anyone who needs a conversion between the two datatypes can then
 *  simply write
 *  \code
 *  Eigen::Affine3d some_tf = ...;
 *  CustomTransform my_tf = icl_core::convert<CustomTransform>(some_tf);
 *  // Or alternatively:
 *  CustomTransform my_tf; icl_core::convert<CustomTransform>(some_tf, my_tf);
 *  \endcode
 *  \note Some information loss may take place if \a TTo does not
 *  support all the features of \a TFrom.  This means that \code
 *  convert<From>(convert<To>(from)) == from \endcode does not
 *  necessarily hold!
 *  \param[in] from An object of type \a TFrom.
 *  \param[out] to An object of type \a TTO.
 */
template <typename TTo, typename TFrom>
void convert(const TFrom& from, TTo& to)
{
  to = static_cast<TTo>(from);
}

/*! Convenience wrapper for \see void convert<TTo, TFrom>(const
 *  TFrom&, TTo&).  You can provide a template specialization of this
 *  for your own datatypes if you have a more performant way to
 *  implement this method.
 *  \note Some information loss may take place if \a TTo does not
 *  support all the features of \a TFrom.  This means that \code
 *  convert<From>(convert<To>(from)) == from \endcode does not
 *  necessarily hold!
 *  \param from An object of type \a TFrom.
 *  \returns An object of type \a TTo which is semantically equivalent
 *           to the original \a from.  Some information loss may take
 *           place if \a TTo does not support all the features of \a
 *           TFrom.
 */
template <typename TTo, typename TFrom>
TTo convert(const TFrom& from)
{
  TTo to;
  convert(from, to);
  return to;
}

}

#endif
