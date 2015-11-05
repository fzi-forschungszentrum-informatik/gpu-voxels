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

namespace gpu_voxels {

using namespace boost::interprocess;

VisProvider::VisProvider(std::string segment_name, std::string map_name)
  : m_segment(),
    m_visualizer_segment(),
    m_segment_name(segment_name),
    m_map_name(map_name),
    m_shm_mapName(),
    m_shm_draw_types(NULL)
{
}

VisProvider::~VisProvider()
{
  // destroying only the named objects leads weird problems of lacking program execution
  shared_memory_object::remove(m_segment_name.c_str());
//  bool destruction_successful = shared_memory_object::remove(m_segment_name.c_str());
//  if(!destruction_successful)
//  {
//    LOGGING_ERROR_C(Gpu_voxels, VisProvider, "Destructor of VisProvider Shared Memory [" << m_segment_name <<
//                    "] failed! Please delete remaining shared mem files at /dev/shm manually before restarting the provider/visualizer!" << endl);
//  }

  // Now we try to delete the visualizer shared mem as well. This is likely to fail, as some other instance of this call
  // may have already deleted it. Therefore we don't care about the result.
  shared_memory_object::remove(shm_segment_name_visualizer.c_str());
}

void VisProvider::openOrCreateSegment()
{
  // Only open/create if not already available
  if (m_segment.get_segment_manager() == NULL) // check whether it's already initialized
  {
    permissions per;
    per.set_unrestricted();
    m_segment = managed_shared_memory(open_or_create, m_segment_name.c_str(), 65536, 0, per);
  }
}

void VisProvider::setDrawTypes(DrawTypes set_draw_types)
{
  if (m_visualizer_segment.get_segment_manager() == NULL) // check whether it's already initialized
  {
    permissions per;
    per.set_unrestricted();
    m_visualizer_segment = managed_shared_memory(open_or_create, shm_segment_name_visualizer.c_str(), 65536, 0, per);
  }
  if (m_shm_draw_types == NULL)
    m_shm_draw_types = m_visualizer_segment.find_or_construct<DrawTypes>(shm_variable_name_set_draw_types.c_str())(
        DrawTypes());
  *m_shm_draw_types = set_draw_types;
}

} // end of ns

