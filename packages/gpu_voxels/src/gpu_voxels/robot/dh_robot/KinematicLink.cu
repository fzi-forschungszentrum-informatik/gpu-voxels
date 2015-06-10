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
#include "gpu_voxels/logging/logging_robot.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace gpu_voxels {
namespace robot {


void DHParameters::convertDHtoM(Matrix4f& m) const
{
//  printf("theta, d, a, alpha : \t%f, %f, %f, %f\n", theta, d, a, alpha);
  float ca = 0;
  float sa = 0;
  float ct = 0;
  float st = 0;
  float d_current = d;
  float theta_current = theta;
  float b = 0; //currently only b = 0

  switch(joint_type)
  {
    case PRISMATIC:
    {
      // if prismatic joint, increment d with joint value
      d_current = d + value;
      break;
    }
    case REVOLUTE:
    {
      // if revolute joint, increment theta with joint value
      theta_current = theta + value;
      break;
    }
    default:
    {
      LOGGING_ERROR_C(RobotLog, DHParameters, "Illegal joint type" << endl);
    }
  }

  ca = (float) cos(alpha);
  sa = (float) sin(alpha);
  ct = (float) cos(theta_current);
  st = (float) sin(theta_current);

  m.a11 = ct;
  m.a12 = -st * ca;
  m.a13 = st * sa;
  m.a14 = a * ct - b * st;

  m.a21 = st;
  m.a22 = ct * ca;
  m.a23 = -ct * sa;
  m.a24 = a * st + b * ct;

  m.a31 = 0.0;
  m.a32 = sa;
  m.a33 = ca;
  m.a34 = d_current;

  m.a41 = 0.0;
  m.a42 = 0.0;
  m.a43 = 0.0;
  m.a44 = 1.0;
}

KinematicLink::KinematicLink(const DHParameters &dh_parameters)
{
  m_dh_parameters = dh_parameters;
}

KinematicLink::KinematicLink(float d, float theta, float a, float alpha, float joint_value, DHJointType joint_type)
{
  m_dh_parameters.d      = d;
  m_dh_parameters.theta  = theta;
  m_dh_parameters.a      = a;
  m_dh_parameters.alpha  = alpha;
  m_dh_parameters.value  = joint_value;
  m_dh_parameters.joint_type = joint_type;
}

KinematicLink::~KinematicLink()
{
}

void KinematicLink::setDHParam(float d, float theta, float a, float alpha, float joint_value, DHJointType joint_type)
{
  m_dh_parameters.d      = d;
  m_dh_parameters.theta  = theta;
  m_dh_parameters.a      = a;
  m_dh_parameters.alpha  = alpha;
  m_dh_parameters.value  = joint_value;
  m_dh_parameters.joint_type = joint_type;
}

void KinematicLink::setDHParam(DHParameters dh_parameters)
{
  m_dh_parameters = dh_parameters;
}

void KinematicLink::getMatrixRepresentation(Matrix4f& m)
{
  m_dh_parameters.convertDHtoM(m);
}

} // end of namespace
} // end of namespace
