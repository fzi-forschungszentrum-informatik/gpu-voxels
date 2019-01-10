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
 * \author  Florian Drews
 * \date    2014-06-18
 *
 */
//----------------------------------------------------------------------/*
#include <gpu_voxels/vis_interface/VisProvider.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <map>


namespace gpu_voxels {

using namespace boost::interprocess;

static std::map<std::string, int> shm_states; // map from segment name to usage counter
static boost::mutex shm_states_mutex; // used to protect shm_states

managed_shared_memory increment_usage(std::string segment_name)
{
  boost::lock_guard<boost::mutex> lock(shm_states_mutex);
  if (shm_states.find(segment_name) == shm_states.end())
  {
    shm_states[segment_name] = 1;
  }
  else
  {
    shm_states[segment_name]++;
  }
  permissions per;
  per.set_unrestricted();
  return managed_shared_memory(open_or_create, segment_name.c_str(), 65536, 0, per);
}

void decrement_usage(std::string segment_name)
{
  boost::lock_guard<boost::mutex> lock(shm_states_mutex);
  int& usages = shm_states[segment_name];
  usages--;
  if (usages == 0)
  {
    shared_memory_object::remove(segment_name.c_str());

    // LOGGING_DEBUG_C(Gpu_voxels, VisProvider, "decrement_usage removed segment [" << segment_name << "] " << endl);
  }
}

VisProvider::VisProvider(std::string segment_name, std::string map_name)
  : m_segment(),
    m_visualizer_segment(),
    m_segment_name(segment_name),
    m_map_name(map_name),
    m_shm_mapName(),
    m_shm_draw_types(NULL)
{
  m_visualizer_segment = increment_usage(shm_segment_name_visualizer);
}

VisProvider::~VisProvider()
{
  if (m_segment.get_segment_manager() != NULL)
  {
    decrement_usage(m_segment_name);
  }
  decrement_usage(shm_segment_name_visualizer);
}

void VisProvider::openOrCreateSegment()
{
  // Only open/create if not already available
  if (m_segment.get_segment_manager() == NULL) // check whether it's already initialized
  {
    m_segment = increment_usage(m_segment_name);
  }
}

void VisProvider::setDrawTypes(DrawTypes set_draw_types)
{
  if (m_shm_draw_types == NULL)
    m_shm_draw_types = m_visualizer_segment.find_or_construct<DrawTypes>(shm_variable_name_set_draw_types.c_str())(
        DrawTypes());
  *m_shm_draw_types = set_draw_types;
}

} // end of ns

