// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2014-06-09
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_COMPILE_ISSUES_H_INCLUDED
#define GPU_VOXELS_HELPERS_COMPILE_ISSUES_H_INCLUDED

// This file contains workarounds for compile issues.


/* ---------------------------------------------------------
 * Issue: https://svn.boost.org/trac/boost/ticket/9392
 *        NVCC does not support all qualifiers.
 *
 * Depending on invocation we have to check  
 * if it's the GCC, the NVCC using GCC to compile host code
 * or the NVCC compiling device code. In the latter 2 cases
 * we have to make sure NVCC understands boost noinline
 * commands.
 * --------------------------------------------------------- */

#if defined(__CUDACC__) && !defined(__GNUC__)
# define noinline false
#endif

#if defined(__CUDACC__) && defined(__GNUC__)
# undef __noinline__
# define __noinline__
#endif

#endif // GPU_VOXELS_HELPERS_COMPILE_ISSUES_H_INCLUDED
