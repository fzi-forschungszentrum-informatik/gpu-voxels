
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
 * \author  Andreas Hermann
 * \date    2014-06-08
 *
 */
//----------------------------------------------------------------------

#include "GpuVoxelsMap.h"
#include <gpu_voxels/helpers/PointcloudFileHandler.h>

namespace gpu_voxels {

GpuVoxelsMap::GpuVoxelsMap()
{
}
GpuVoxelsMap::~GpuVoxelsMap()
{
}

void GpuVoxelsMap::lockSelf(const std::string& function_name) const
{
#ifdef DBG_LOCKING
  LOGGING_WARNING_C(Gpu_Voxels_Map, GpuVoxelsMap, "!!!!!!!!!!!!!!!!!!! " << function_name << ": Tries to lock 'self' !!!!!!!!!!!!!!!!!!!" << endl);
#endif
  bool locked_this = false;
  uint32_t counter = 0;

  while (!locked_this)
  {
    locked_this = m_mutex.try_lock_for(boost::chrono::milliseconds(5));
    if(!locked_this)
    {
      counter++;
      if(counter > 40)
      {
        LOGGING_WARNING_C(Gpu_Voxels_Map, GpuVoxelsMap, function_name << ": Could not lock self since 40 trials (5 ms each)!" << endl);
        counter = 0;
      }
      boost::this_thread::yield();
    }
  }
#ifdef DBG_LOCKING
  LOGGING_WARNING_C(Gpu_Voxels_Map, GpuVoxelsMap, "!!!!!!!!!!!!!!!!!!! " << function_name << ": Locked 'self' !!!!!!!!!!!!!!!!!!!" << endl);
#endif
}

void GpuVoxelsMap::unlockSelf(const std::string& function_name) const
{
#ifdef DBG_LOCKING
  LOGGING_WARNING_C(Gpu_Voxels_Map, GpuVoxelsMap, "!!!!!!!!!!!!!!!!!!! " << function_name << ": Unlocks 'self' !!!!!!!!!!!!!!!!!!!" << endl);
#endif
  m_mutex.unlock();
}

void GpuVoxelsMap::unlockBoth(const GpuVoxelsMap* map1, const GpuVoxelsMap* map2, const std::string& function_name) const
{
#ifdef DBG_LOCKING
  LOGGING_WARNING_C(Gpu_Voxels_Map, GpuVoxelsMap, "!!!!!!!!!!!!!!!!!!! " << function_name << ": Unlocks 'both' !!!!!!!!!!!!!!!!!!!" << endl);
#endif

  map1->m_mutex.unlock();
  map2->m_mutex.unlock();

}

void GpuVoxelsMap::lockBoth(const GpuVoxelsMap* map1, const GpuVoxelsMap* map2, const std::string& function_name) const
{
#ifdef DBG_LOCKING
  LOGGING_WARNING_C(Gpu_Voxels_Map, GpuVoxelsMap, "!!!!!!!!!!!!!!!!!!! " << function_name << ": Tries to lock 'both' !!!!!!!!!!!!!!!!!!!" << endl);
#endif
  bool locked_map1 = false;
  bool locked_map2 = false;
  uint32_t m1_counter = 0;
  uint32_t m2_counter = 0;

  while (!locked_map1 || !locked_map2)
  {
    // lock mutexes
    while (!locked_map1)
    {
      locked_map1 = map1->m_mutex.try_lock_for(boost::chrono::milliseconds(5));
      m1_counter++;
      if(!locked_map1)
      {
        if(m1_counter >= 40)
        {
          LOGGING_WARNING_C(Gpu_Voxels_Map, GpuVoxelsMap, function_name << ": Could not lock map1 map since 40 trials (5 ms each)!" << endl);
          m1_counter = 0;
        }
        boost::this_thread::yield();
      }
    }
    while (!locked_map2 && (m2_counter < 40))
    {
      locked_map2 = map2->m_mutex.try_lock_for(boost::chrono::milliseconds(5));
      if(!locked_map2) boost::this_thread::yield();
      m2_counter++;
    }
    if (!locked_map2)
    {
      LOGGING_WARNING_C(Gpu_Voxels_Map, GpuVoxelsMap, function_name << ": Could not lock map2 map since 40 trials (5 ms each)!" << endl);
      m2_counter = 0;
      map1->m_mutex.unlock();
      boost::this_thread::yield();
    }
  }
#ifdef DBG_LOCKING
  LOGGING_WARNING_C(Gpu_Voxels_Map, GpuVoxelsMap, "!!!!!!!!!!!!!!!!!!! " << function_name << ": Locked 'both' !!!!!!!!!!!!!!!!!!!" << endl);
#endif
}

bool GpuVoxelsMap::insertPointCloudFromFile(const std::string path, const bool use_model_path, const BitVoxelMeaning voxel_meaning,
                                            const bool shift_to_zero, const Vector3f &offset_XYZ, const float scaling)
{
  //load the points into the vector
  std::vector<Vector3f> points;

  if(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(path, use_model_path, points, shift_to_zero, offset_XYZ, scaling))
  {
    insertPointCloud(points, voxel_meaning);
    return true;
  }
  return false;
}

void GpuVoxelsMap::generateVisualizerData()
{
}

bool GpuVoxelsMap::rebuildIfNeeded()
{
  if(needsRebuild())
  {
    rebuild();
    return true;
  }
  else
    return false;
}

MapType GpuVoxelsMap::getMapType() const
{
  return m_map_type;
}

} // end of ns

