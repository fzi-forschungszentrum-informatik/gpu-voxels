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
 * \date    2008-01-28
 *
 * \brief   Contains global filesystem related functions,
 *          encapsulated into the icl_core::os namespace
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_OS_FS_H_INCLUDED
#define ICL_CORE_OS_FS_H_INCLUDED

#include "icl_core/ImportExport.h"
#include "icl_core/os_ns.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

#if defined _SYSTEM_POSIX_
# include "icl_core/os_posix_fs.h"
#elif defined _SYSTEM_WIN32_
# include "icl_core/os_win32_fs.h"
#else
# error "No os_fs implementation defined for this platform."
#endif

namespace icl_core {
//! Namespace for operating system specific implementations.
namespace os {

inline int rename(const char *old_filename, const char *new_filename)
{
  return ICL_CORE_OS_IMPL_NS::rename(old_filename, new_filename);
}

inline int unlink(const char *filename)
{
  return ICL_CORE_OS_IMPL_NS::unlink(filename);
}

#ifdef _IC_BUILDER_ZLIB_
/*!
 * Zip the specified file using the gzip algorithm.
 * Append the \a additional_extension to the original filename.
 */
bool ICL_CORE_IMPORT_EXPORT zipFile(const char *filename, const char *additional_extension = "");

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

bool ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE
ZipFile(const char *filename, const char *additional_extension = "") ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

#endif

}
}

#endif
