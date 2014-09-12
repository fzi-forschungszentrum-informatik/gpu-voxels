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


class KinematicLink;
typedef boost::shared_ptr<KinematicLink>  KinematicLinkSharedPtr;

class KinematicLink
{
public:

  typedef enum
  {
    REVOLUTE = 0,
    PRISMATIC
  } JointType;

  struct DHParameters
  {
    float d;
    float theta;        // initial rotation
    float a;            // intial translation
    float alpha;
    float value;       /* joint value
                          (rotation for revolute joints,
                          translation for prismatic joints) */
    DHParameters()
        : d(0.0), theta(0.0), a(0.0), alpha(0.0), value(0.0)
        {
        }

    DHParameters(float _d, float _theta, float _a, float _alpha, float _value)
        : d(_d), theta(_theta), a(_a), alpha(_alpha), value(_value)
        {
        }
  };

  __host__
  KinematicLink(JointType joint_type);

  //! destructor.
  __host__
  ~KinematicLink();

  __host__
  void setJointType(JointType joint_type)
  {
    m_joint_type = joint_type;
  }

  __host__
  void setJointValue(float value)
  {
    m_dh_parameters.value = value;
  }

  __host__
  JointType getJointType()
  {
    return m_joint_type;
  }

  __host__
  float getJointValue()
  {
    if (m_joint_type == PRISMATIC)
    {
      return m_dh_parameters.a;
    }
    else
    {
      return m_dh_parameters.theta;
    }
  }

  __host__
  DHParameters getDHParam()
  {
    return m_dh_parameters;
  }

  void setDHParam(float d, float theta, float a, float alpha, float joint_value);
  void setDHParam(DHParameters dh_parameters);

private:

//  template<class T>
//  void tokenize_to_vector(const std::string &s, std::vector<T> &out)
//  {
//    typedef boost::tokenizer<boost::escaped_list_separator<char> >  tokenizer;
//
//    tokenizer tok(s);
//    for (tokenizer::iterator it(tok.begin()); it!= tok.end(); it++)
//    {
//      std::string f(*it);
//      boost::trim(f);
//      out.push_back(boost::lexical_cast<T>(f));
//    }
//  }


  float m_joint_value;
  JointType m_joint_type;
  DHParameters m_dh_parameters;
  Matrix4f m_dh_transformation;
};

} // end of namespace
#endif
