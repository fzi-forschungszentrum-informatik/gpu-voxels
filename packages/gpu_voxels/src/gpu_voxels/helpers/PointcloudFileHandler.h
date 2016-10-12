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
 * \date    2014-07-10
 *
 * This is a singleton implementation of a helper to load
 * various pointcloud filetypes.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_POINTCLOUD_FILE_HANDLER_H_INCLUDED
#define GPU_VOXELS_HELPERS_POINTCLOUD_FILE_HANDLER_H_INCLUDED
#include <cstdlib>
#include <boost/filesystem/path.hpp>
#include "gpu_voxels/helpers/cuda_datatypes.h"

/**
 * @namespace gpu_voxels::file_handling
 * Parser for different pointcloud files
 */
namespace gpu_voxels {
namespace file_handling {

// Forward declaration for the specific readers:
class PcdFileReader;
class XyzFileReader;
class BinvoxFileReader;

class PointcloudFileHandler
{
public:

  /*!
   * \brief Instance generator
   * \return Pointer to singleton instance of PointcloudFileHandler
   */
  static PointcloudFileHandler* Instance();

  /*!
   * \brief loadPointCloud loads a PCD file and returns the points in a vector.
   * \param path Filename
   * \param points points are written into this vector
   * \param shift_to_zero If true, the pointcloud is shifted, so its minimum coordinates lie at zero
   * \param offset_XYZ Additional transformation offset
   * \return true if succeeded, false otherwise
   */
  bool loadPointCloud(const std::string _path, const bool use_model_path, std::vector<Vector3f> &points, const bool shift_to_zero = false,
                      const Vector3f &offset_XYZ = Vector3f(), const float scaling = 1.0);

  ~PointcloudFileHandler();

private:
  /*!
   * \brief Private ctor, as this is a singleton
   */
  PointcloudFileHandler();

  /*!
   * \brief Private copy ctor, as this is a singleton
   */
  PointcloudFileHandler(PointcloudFileHandler const&){}

  /*!
   * \brief m_instance singleton instance
   */
  static PointcloudFileHandler* m_instance;

  /*!
   * \brief centerPointCloud Centers a pointcloud relative to its maximum coordinates
   * \param points Working cloud
   */
  void centerPointCloud(std::vector<Vector3f> &points);

  /*!
   * \brief shiftPointCloudToZero Moves a pointcloud, so that its minimum coordinates are shifted to zero.
   * \param points Working cloud
   */
  void shiftPointCloudToZero(std::vector<Vector3f> &points);

  XyzFileReader* xyz_reader;
  PcdFileReader* pcd_reader;
  BinvoxFileReader* binvox_reader;

};


}  // end of ns
}  // end of ns

#endif
