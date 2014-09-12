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
 * \author  Jörg Bauer
 * \date    2014-01-22
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_GLOBAL_LOCK_H_INCLUDED
#define GPU_VOXELS_GLOBAL_LOCK_H_INCLUDED

#include <boost/interprocess/interprocess_fwd.hpp>

#include <gpu_voxels/logging/logging_gpu_voxels.h>

namespace gpu_voxels {

class GlobalLock
{
public:

  GlobalLock();

  void lock();

  void unlock();

private:

  boost::interprocess::managed_shared_memory* m_shared_mem;
};

}  // end of namespace
#endif
