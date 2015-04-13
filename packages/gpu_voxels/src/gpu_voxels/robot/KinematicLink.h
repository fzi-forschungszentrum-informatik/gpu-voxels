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
 * \date    2012-09-13
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_KINEMATIC_LINK_H_INCLUDED
#define GPU_VOXELS_KINEMATIC_LINK_H_INCLUDED

#include <cuda_runtime.h>
#include <vector>
#include <boost/shared_ptr.hpp>

#include <gpu_voxels/helpers/cuda_handling.hpp>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/CudaMath.h>



namespace gpu_voxels {


typedef enum
{
  REVOLUTE = 0,
  PRISMATIC
} DHJointType;

class DHParameters
{
public:
  float d;
  float theta;        // initial rotation
  float a;            // intial translation
  float alpha;
  float value;       /* joint value
                        (rotation for revolute joints,
                        translation for prismatic joints) */
  DHJointType joint_type;

  DHParameters()
      : d(0.0), theta(0.0), a(0.0), alpha(0.0), value(0.0), joint_type(REVOLUTE)
      {
      }

  DHParameters(float _d, float _theta, float _a, float _alpha, float _value, DHJointType _joint_type = REVOLUTE)
      : d(_d), theta(_theta), a(_a), alpha(_alpha), value(_value), joint_type(_joint_type)
      {
      }

  void convertDHtoM(Matrix4f& m) const;
};








class KinematicLink;
typedef boost::shared_ptr<KinematicLink>  KinematicLinkSharedPtr;

class KinematicLink
{
public:

  __host__
  KinematicLink(DHJointType joint_type);

  //! destructor.
  __host__
  ~KinematicLink();

  __host__
  void setJointType(DHJointType joint_type)
  {
    m_dh_parameters.joint_type = joint_type;
  }

  __host__
  void setJointValue(float value)
  {
    m_dh_parameters.value = value;
  }

  __host__
  DHJointType getJointType()
  {
    return m_dh_parameters.joint_type;
  }

  __host__
  float getJointValue()
  {
    if (m_dh_parameters.joint_type == PRISMATIC)
    {
      return m_dh_parameters.a + m_dh_parameters.value;
    }
    else
    {
      return m_dh_parameters.theta + m_dh_parameters.value;
    }
  }

  __host__
  DHParameters getDHParam()
  {
    return m_dh_parameters;
  }

  void setDHParam(float d, float theta, float a, float alpha, float joint_value);
  void setDHParam(DHParameters dh_parameters);

  void getMatrixRepresentation(Matrix4f& m);

private:
  float m_joint_value;
  DHParameters m_dh_parameters;
  Matrix4f m_dh_transformation;
};

} // end of namespace
#endif
