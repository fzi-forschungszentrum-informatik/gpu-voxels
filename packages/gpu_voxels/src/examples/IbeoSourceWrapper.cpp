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
 * \date    2014-10-28
 *
 */
//----------------------------------------------------------------------
#include "IbeoSourceWrapper.h"
#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include <gpu_voxels/helpers/CudaMath.h>

#include <icl_sourcesink/SourceSinkManager.h>

/* Include sources that we want to be able to handle.
 * Replace these if you want to handle a different data type.
 */
#include <icl_hardware_ibeo_noapi/source/IbeoSources.h>

#include <icl_hardware_ncom/NcomFileSource.h>
#include <icl_gps_types/AbsolutePose.h>
#include <icl_math/Pose.h>

namespace nibeo = icl_hardware::ibeo;
namespace nncom = icl_hardware::ncom;

namespace gpu_voxels {

IbeoSourceWrapper::IbeoSourceWrapper(CallbackFunction callback, std::string ibeo_uri, std::string ncom_uri, Vector3f additional_translation) :
    m_callback(callback),
    m_ibeo_uri(ibeo_uri),
    m_ncom_uri(ncom_uri),
    m_abort(false),
    m_additional_translation(additional_translation)
{
}

void IbeoSourceWrapper::run()
{
    // Create ibeo source
    icl_sourcesink::SourceSinkManager manager;
    nibeo::IbeoSourceNoAPI::Ptr ibeo_source = manager.createSource<nibeo::IbeoMsg>(m_ibeo_uri, std::vector<std::string>(), true);
    nncom::NcomSource::Ptr ncom_source = manager.createSource<nncom::Ncom>(m_ncom_uri);

    // We create a data element ptr that we will use to work on.
    nibeo::IbeoMsgStamped::Ptr ibeo_element;
    nncom::NcomStamped::Ptr ncom_element;
    icl_gps::AbsolutePose* ref_pose = NULL;

    // for each time step
    for (; manager.good(); manager.advance())
    {
       ibeo_element = ibeo_source->get();
       ncom_element = ncom_source->get();

      // Skip messages other than IbeoScanMsg
      nibeo::IbeoScanMsg scan_msg;
      if(!scan_msg.fromIbeoMsg(*ibeo_element))
          continue;

      LOGGING_INFO(Gpu_voxels, "Number of points in ibeo msg: " << scan_msg.number_of_points << endl);

      // Copy points to std::vector
      std::vector<Vector3f> point_cloud(scan_msg.number_of_points);
      for(int i = 0; i < scan_msg.number_of_points; ++i)
      {
          point_cloud[i].x = (*scan_msg.scan_points)[i].x;
          point_cloud[i].y = (*scan_msg.scan_points)[i].y;
          point_cloud[i].z = (*scan_msg.scan_points)[i].z;
      }

      // Handle position
      nncom::Ncom ncom_msg = *ncom_element; // Unwrap message
      icl_gps::AbsolutePose abs_pose(ncom_msg->mLon, ncom_msg->mLat, ncom_msg->mAlt, ncom_msg->mTrack);
       LOGGING_INFO(Gpu_voxels, "Current pose. long: " << abs_pose.longitude() << " lat: " << abs_pose.latitude() << " alt: " << abs_pose.altitude() << " bear: " << abs_pose.bearing() << endl);

      if(!ref_pose) // Use first position as reference
      {
          ref_pose = new icl_gps::AbsolutePose(abs_pose.longitude(), abs_pose.latitude(), abs_pose.altitude(), abs_pose.bearing());
          LOGGING_INFO(Gpu_voxels, "Set reference pose. long: " << abs_pose.longitude() << " lat: " << abs_pose.latitude() << " alt: " << abs_pose.altitude() << " bear: " << abs_pose.bearing() << endl);
      }
      icl_gps::RelativePose rel_pose = abs_pose - *ref_pose;
      icl_math::Pose2d rel_pose2d = rel_pose.toPose2d();

      LOGGING_INFO(Gpu_voxels, "Relative pose. north: " << rel_pose.north() << " east: " << rel_pose.east() << " alt: " << rel_pose.altitude() << endl);

      Matrix4f matrix;

      // copy pose2d rotation matrix
      matrix.a11 = rel_pose2d(0,0);     matrix.a12 = rel_pose2d(0,1);
      matrix.a21 = rel_pose2d(1,0);     matrix.a22 = rel_pose2d(1,1);

      // copy pose2d translation vector
      matrix.a14 = rel_pose2d(0,2);
      matrix.a24 = rel_pose2d(1,2);

      // set generals structure of transformation matrix
      matrix.a33 = 1.0f;
      matrix.a44 = 1.0f;

      // set z-translation (altitude)
      matrix.a34 = static_cast<float>(rel_pose.altitude());

      // set additional translation
      matrix.a14 += m_additional_translation.x;
      matrix.a24 += m_additional_translation.y;
      matrix.a34 += m_additional_translation.z;

      LOGGING_INFO(Gpu_voxels, "Matrix4f: " << endl << matrix.a11 << " " << matrix.a12 << " " << matrix.a13 << " " << matrix.a14 << endl
                   << matrix.a21 << " " << matrix.a22 << " " << matrix.a23 << " " << matrix.a24 << endl
                   << matrix.a31 << " " << matrix.a32 << " " << matrix.a33 << " " << matrix.a34 << endl
                   << matrix.a41 << " " << matrix.a42 << " " << matrix.a43 << " " << matrix.a44 << endl);

      LOGGING_INFO(Gpu_voxels, "Transform point cloud..." << endl);

      CudaMath::transform(point_cloud, matrix); // todo: keep point cloud in gpu mem

      m_callback(point_cloud);
    }
}

void IbeoSourceWrapper::stop()
{
    m_abort = true;
}

}









