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
#include "GlobalLock.h"
#include <utility>
#include <iostream>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

namespace gpu_voxels {

GlobalLock::GlobalLock()
{
  m_shared_mem = new boost::interprocess::managed_shared_memory(boost::interprocess::open_only,
                                                                "MySharedMemory");
}

void GlobalLock::lock()
{
  std::pair<boost::interprocess::interprocess_mutex*, size_t> map_mutex = m_shared_mem->find<
      boost::interprocess::interprocess_mutex>("mutex");
  map_mutex.first->lock();
  LOGGING_DEBUG_C(Gpu_voxels, GlobalLock, "Locking" << endl);
}

void GlobalLock::unlock()
{
  std::pair<boost::interprocess::interprocess_mutex*, size_t> map_mutex = m_shared_mem->find<
      boost::interprocess::interprocess_mutex>("mutex");
  map_mutex.first->unlock();
  LOGGING_DEBUG_C(Gpu_voxels, GlobalLock, "Unlocking" << endl);
}

} // end of namespace
