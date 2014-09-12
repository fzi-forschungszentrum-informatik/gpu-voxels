// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
//----------------------------------------------------------------------
#include "KinematicLink.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace gpu_voxels {

KinematicLink::KinematicLink(JointType joint_type)
    : m_joint_type(joint_type)
{
}


KinematicLink::~KinematicLink()
{
}

void KinematicLink::setDHParam(float d, float theta, float a, float alpha, float joint_value)
{
  m_dh_parameters.d      = d;
  m_dh_parameters.theta  = theta;
  m_dh_parameters.a      = a;
  m_dh_parameters.alpha  = alpha;
  m_dh_parameters.value  = joint_value;
}

void KinematicLink::setDHParam(DHParameters dh_parameters)
{
  m_dh_parameters = dh_parameters;
}

} // end of namespace
