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
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-06-30
 *
 * \brief   Contains macros to deprecate classes, types, functions and variables.
 *
 * Deprecation warnings can be disabled by compiling with the
 * ICL_CORE_NO_DEPRECATION macro defined.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_DEPRECATE_H_INCLUDED
#define ICL_CORE_DEPRECATE_H_INCLUDED

// Define deprecation macros for Visual Studio.
#if defined(_MSC_VER) && !defined(ICL_CORE_NO_DEPRECATION)
# define ICL_CORE_VC_DEPRECATE __declspec(deprecated)
# define ICL_CORE_VC_DEPRECATE_COMMENT(arg) __declspec(deprecated(arg))
#else
# define ICL_CORE_VC_DEPRECATE
# define ICL_CORE_VC_DEPRECATE_COMMENT(arg)
#endif

// Define deprecation macros for GCC.
#if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1)) && !defined(ICL_CORE_NO_DEPRECATION)
# define ICL_CORE_GCC_DEPRECATE __attribute__((deprecated))
# if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5))
#  define ICL_CORE_GCC_DEPRECATE_COMMENT(arg) __attribute__((deprecated(arg)))
# else
#  define ICL_CORE_GCC_DEPRECATE_COMMENT(arg) __attribute__((deprecated))
# endif
#else
# define ICL_CORE_GCC_DEPRECATE
# define ICL_CORE_GCC_DEPRECATE_COMMENT(arg)
#endif

// Special comment for deprecation due to obsolete style.
#define ICL_CORE_VC_DEPRECATE_STYLE ICL_CORE_VC_DEPRECATE_COMMENT("Please follow the new Coding Style Guidelines.")
#define ICL_CORE_GCC_DEPRECATE_STYLE ICL_CORE_GCC_DEPRECATE_COMMENT("Please follow the new Coding Style Guidelines.")

// Special comment for changing to new source sink pattern.
#define ICL_CORE_VC_DEPRECATE_SOURCESINK ICL_CORE_VC_DEPRECATE_COMMENT("Please follow the new Source Sink Pattern.")
#define ICL_CORE_GCC_DEPRECATE_SOURCESINK ICL_CORE_GCC_DEPRECATE_COMMENT("Please follow the new Source Sink Pattern.")

// Special comment for moving to ROS workspace.
#define ICL_CORE_VC_DEPRECATE_MOVE_ROS ICL_CORE_VC_DEPRECATE_COMMENT("This was moved to a ROS package. Please use the implementation in ros_icl or ros_sourcesink instead.")
#define ICL_CORE_GCC_DEPRECATE_MOVE_ROS ICL_CORE_GCC_DEPRECATE_COMMENT("This was moved to a ROS package. Please use the implementation in ros_icl or ros_sourcesink instead.")

// Special comment for deprecation due to obsolete style which
// provides the name of the function that superseded the obsolete one.
#define ICL_CORE_VC_DEPRECATE_STYLE_USE(arg) ICL_CORE_VC_DEPRECATE_COMMENT("Please follow the new Coding Style Guidelines and use " #arg " instead.")
#define ICL_CORE_GCC_DEPRECATE_STYLE_USE(arg) ICL_CORE_GCC_DEPRECATE_COMMENT("Please follow the new Coding Style Guidelines and use " #arg " instead.")

#endif
