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
#ifndef GPU_VOXELS_HELPER_TF_HELPER_H_INCLUDED
#define GPU_VOXELS_HELPER_TF_HELPER_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

namespace gpu_voxels
{

class tfHelper
{
public:

  tfHelper();

  ~tfHelper();

  /*!
   * \brief publish Publishes the given transform as a TF
   * \param transform The input matrix of the transform
   * \param parent The TF transformations parent name
   * \param child The TF transformations child name
   */
  void publish(const Matrix4f &transform, const std::string &parent, const std::string &child);

  /*!
   * \brief lookup Tries to look up a TF and returns it as matrix
   * \param parent The TF transformations parent name
   * \param child The TF transformations child name
   * \param transform The TFs transformation as matrix
   * \param waitTime How long to wait for the TF until failure.
   * \return True if successful. False is lookup failed.
   */
  bool lookup(const std::string &parent, const std::string &child, Matrix4f &transform, float waitTime = 5.0);


private:
  tf::TransformBroadcaster m_tf_br;
  tf::TransformListener m_tf_li;
};

}//end namespace gpu_voxels
#endif // GPU_VOXELS_HELPERS_TF_HELPER_H_INCLUDED
