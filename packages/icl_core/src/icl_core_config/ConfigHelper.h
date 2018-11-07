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
 * \date    2012-07-09
 *
 * Helper macros for the enhanced batch convenience macros for
 * icl_core_config.  The batch macros CONFIG_VALUE, CONFIG_ENUM,
 * MEMBER_VALUE etc. are only available on certain compilers that
 * support the decltype or __typeof__ macros.  To test whether these
 * macros are available, you can include this file check against the
 * ICL_CORE_CONFIG_HAS_ENHANCED_CONFIG_MACROS macro.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_HELPER_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_HELPER_H_INCLUDED

#if defined(_MSC_VER)
# if _MSC_VER >= 1600
#  define ICL_CORE_CONFIG_HAS_ENHANCED_CONFIG_MACROS
#  define ICL_CORE_CONFIG_TYPEOF(value) decltype(value)
# else
#  include <boost/typeof/typeof.hpp>
#  define ICL_CORE_CONFIG_HAS_ENHANCED_CONFIG_MACROS
#  define ICL_CORE_CONFIG_TYPEOF(value) BOOST_TYPEOF(value)
//#  pragma message("The CONFIG_VALUE convenience macros are only available in Visual Studio 2010 and newer.")
//#  define ICL_CORE_CONFIG_TYPEOF(value) THIS_FEATURE_IS_NOT_AVAILABLE_ON_YOUR_COMPILER
# endif
#else
# define ICL_CORE_CONFIG_HAS_ENHANCED_CONFIG_MACROS
# define ICL_CORE_CONFIG_TYPEOF(value) __typeof__(value)
#endif

#endif
