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
 * \date    2014-07-10
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_FILE_HANDLING_H_INCLUDED
#define GPU_VOXELS_HELPERS_FILE_HANDLING_H_INCLUDED
#include <cstdlib>
#include <boost/filesystem/path.hpp>

namespace gpu_voxels {
namespace file_handling {


/*! Read environment variable GPU_VOXELS_MODEL_PATH into \a path
 *  \returns \c true, if variable could be read, \c false otherwise
 */
inline bool getGpuVoxelsPath(boost::filesystem::path& path)
{
  char const* tmp = std::getenv("GPU_VOXELS_MODEL_PATH");
  if (tmp == NULL)
  {
    return false;
  }

  path = boost::filesystem::path(tmp);
  return true;
}

}  // end of ns
}  // end of ns

#endif
