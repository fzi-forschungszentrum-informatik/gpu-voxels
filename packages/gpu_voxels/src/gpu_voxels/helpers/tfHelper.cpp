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
 * \date    2016-08-07
 *
 */
//----------------------------------------------------------------------
#include "tfHelper.h"

namespace gpu_voxels
{

  tfHelper::tfHelper()
  {

  }

  tfHelper::~tfHelper()
  {

  }


  void tfHelper::publish(const Matrix4f &transform, const std::string &parent, const std::string &child)
  {
    tf::Transform tf_transform;
    tf_transform.setOrigin( tf::Vector3(transform.a14, transform.a24, transform.a34) );
    tf_transform.setBasis( tf::Matrix3x3(transform.a11, transform.a12, transform.a13,
                                         transform.a21, transform.a22, transform.a23,
                                         transform.a31, transform.a32, transform.a33) );

    m_tf_br.sendTransform(tf::StampedTransform(tf_transform, ros::Time::now(), parent, child));
  }



  bool tfHelper::lookup(const std::string &parent, const std::string &child, Matrix4f &transform, float waitTime)
  {
    tf::StampedTransform tf_transform;

    if(m_tf_li.waitForTransform(parent, child, ros::Time(0), ros::Duration(waitTime), ros::Duration(0.05)))
    {
      try
      {
        m_tf_li.lookupTransform(parent, child, ros::Time(0), tf_transform);
      }
      catch (tf::TransformException ex)
      {
        ROS_ERROR("tfHelper::lookup error: %s",ex.what());
        return false;
      }
    }else{
      ROS_ERROR_STREAM("tfHelper::lookup: Could not find a transform from '"<< parent <<"' to '"<< child <<"'' for "<< waitTime <<" seconds!");
      return false;
    }

    transform.setIdentity();
    transform.a11 = tf_transform.getBasis()[0].x();
    transform.a12 = tf_transform.getBasis()[0].y();
    transform.a13 = tf_transform.getBasis()[0].z();
    transform.a21 = tf_transform.getBasis()[1].x();
    transform.a22 = tf_transform.getBasis()[1].y();
    transform.a23 = tf_transform.getBasis()[1].z();
    transform.a31 = tf_transform.getBasis()[2].x();
    transform.a32 = tf_transform.getBasis()[2].y();
    transform.a33 = tf_transform.getBasis()[2].z();
    transform.a14 = tf_transform.getOrigin().x();
    transform.a24 = tf_transform.getOrigin().y();
    transform.a34 = tf_transform.getOrigin().z();

    return true;
  }

}//end namespace gpu_voxels
