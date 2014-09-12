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
 * \date    2014-04-04
 *
 */
//----------------------------------------------------------------------

#include "Provider.h"

namespace gpu_voxels {
namespace NTree {
namespace Provider {

using namespace boost::interprocess;

Provider::Provider()
{
  m_sensor_orientation = gpu_voxels::Vector3f(0);
  m_sensor_position = gpu_voxels::Vector3f(0);
  m_collide_with = NULL;
  m_changed = false;
  m_parameter = NULL;
}

Provider::~Provider()
{
  // destroying only the named objects leads weird problems of lacking program execution
  shared_memory_object::remove(m_segment_name.c_str());
}

void Provider::init(Provider_Parameter& parameter)
{
  m_shared_mem_id = boost::lexical_cast<std::string>(parameter.shared_segment);
  permissions per;
  per.set_unrestricted();
  m_segment = managed_shared_memory(open_or_create, m_segment_name.c_str(), 65536, 0, per);
  m_parameter = &parameter;
}

void Provider::updateSensorPose(float yaw, float pitch, float roll)
{
  m_sensor_orientation.x = roll;
  m_sensor_orientation.y = pitch;
  m_sensor_orientation.z = yaw;
}

void Provider::updateSensorPose(float yaw, float pitch, float roll, gpu_voxels::Vector3f position)
{
  updateSensorPose(yaw, pitch, roll);
  m_sensor_position = position;
}

void Provider::setCollideWith(Provider* collide_with)
{
  m_collide_with = collide_with;
}

void Provider::setChanged(bool changed)
{
  m_changed = changed;
}

bool Provider::getChanged()
{
  return m_changed;
}

void Provider::lock()
{
  m_mutex.lock();
}

void Provider::unlock()
{
  m_mutex.unlock();
}

}
}
}
