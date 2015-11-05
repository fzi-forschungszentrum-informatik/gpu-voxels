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
 * \author  Matthias Wagner
 * \date    2014-07-09
 *
 * \brief This class is for the management of the interprocess communication with the provider.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGER_H_INCLUDED

#include <gpu_voxels/helpers/CompileIssues.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <gpu_visualization/logging/logging_visualization.h>

namespace gpu_voxels {
namespace visualization {

class SharedMemoryManager
{
public:
  /**
   * If open_only is false and the no segment with the given name exists, a new one will be created.
   *
   * @throws boost::interprocess::interprocess_exception if open_only is true and the segment does not exist.
   */
  SharedMemoryManager(std::string segment_name, bool open_only)
  {
    if (open_only)
    {
      m_segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only,
                                                             segment_name.c_str());
    }
    else
    {
      LOGGING_WARNING(Visualization, "SharedMemoryManager CREATES shared memory segment. This is meant for debugging puposes ONLY" << endl);
      boost::interprocess::permissions per;
      per.set_unrestricted();
      m_segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_or_create,
                                                             segment_name.c_str(), 65536, 0, per);
    }
    m_segment_name = segment_name;
  }
  ~SharedMemoryManager()
  {
  }

  const std::string& getSegmentName() const
  {
    return m_segment_name;
  }

  boost::interprocess::managed_shared_memory& getMemSegment()
  {
    return m_segment;
  }
private:
  std::string m_segment_name;
  boost::interprocess::managed_shared_memory m_segment;
}
;

} //end of namespace visualization
} //end of namespace gpu_voxels

#endif /* GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGER_H_INCLUDED */
