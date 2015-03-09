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
 * \date    2014-06-08
 *
 * This class holds a generic interface to all kinds of maps that
 * are offered by GPU Voxels.
 *
 * TODO: Move the code from Provider to here?
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_GPU_VOXELS_MAP_H_INCLUDED
#define GPU_VOXELS_GPU_VOXELS_MAP_H_INCLUDED

#include <cstdio>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <stdint.h> // for fixed size datatypes
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>

#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {

class GpuVoxelsMap;
typedef boost::shared_ptr<GpuVoxelsMap> GpuVoxelsMapSharedPtr;

class GpuVoxelsMap
{
public:
  //! Constructor
  GpuVoxelsMap();

  //! Destructor
  virtual ~GpuVoxelsMap();

  /*!
   * \brief getMapType returns the type of the map
   * \return the type of the map
   */
  MapType getMapType();

  /*!
   * \brief insertGlobalData Inserts a pointcloud with global coordinates
   * \param point_cloud The pointcloud to insert
   */
  virtual void insertGlobalData(const std::vector<Vector3f> &point_cloud, VoxelType voxel_type) = 0;

  virtual bool insertRobotConfiguration(const MetaPointCloud *robot_links, bool with_self_collision_test) = 0;

  /**
   * @brief insertMetaPointCloud Inserts a MetaPointCloud into the map.
   * @param meta_point_cloud The MetaPointCloud to insert
   * @param voxel_type Voxel type of all voxels
   */
  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, VoxelType voxel_type) = 0;

  /**
   * @brief insertMetaPointCloud Inserts a MetaPointCloud into the map. Each pointcloud
   * inside the MetaPointCloud will get it's own voxel type as given in the voxel_types
   * parameter. The number of pointclouds in the MetaPointCloud and the size of voxel_types
   * have to be identical.
   * @param meta_point_cloud The MetaPointCloud to insert
   * @param voxel_types Vector with voxel types
   */
  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, const std::vector<VoxelType>& voxel_types) = 0;

  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param other The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWith(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0, const Vector3ui &offset = Vector3ui()) = 0;

  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param other The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param resolution_level The resolution used for collision checking. resolution_level = 0 delivers the highest accuracy whereas each increase haves the resolution.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithResolution(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3ui &offset = Vector3ui()) = 0;

  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param other The map to do a collision check with.
   * \param types_in_collision The voxel types in collision.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithTypes(const GpuVoxelsMapSharedPtr other, voxelmap::BitVectorVoxel&  types_in_collision, float coll_threshold = 1.0, const Vector3ui &offset = Vector3ui()) = 0;

  /*!
   * \brief insertPointcloudFromFile inserts a pointcloud from a file into the map
   * The coordinates are interpreted as global coordinates
   * \param path filename (Must end in .xyz for PCD files or .binvox for Binvox files)
   * \param shift_to_zero if true, the map will be shifted, so that its minimum lies at zero.
   * \param offset_XYZ if given, the map will be transformed by this XYZ offset. If shifting is active, this happens after the shifting.
   * \return true if succeeded, false otherwise
   */
  bool insertPointcloudFromFile(const std::string path, VoxelType voxel_type, const bool shift_to_zero = false,
                 const Vector3f &offset_XYZ = Vector3f(), const float scaling = 1.0);

  // Maintanance functions
  /*!
   * \brief writeToDisk serializes the map and dumps it to a file
   * \param path filename
   */
  virtual void writeToDisk(const std::string path) = 0;

  /*!
   * \brief readFromDisk reads a serialized mapdump from disk
   * \param path filename
   * \return true if succeeded, false otherwise
   */
  virtual bool readFromDisk(const std::string path) = 0;

  /*!
   * \brief generateVisualizerData Generates data for the visualizer.
   */
  void generateVisualizerData();

  /*!
   * \brief clearMap This empties the map.
   */
  virtual void clearMap() = 0;

  /*!
   * \brief clearVoxelType Clears the map from a specific VoxelType
   * \param voxel_type The type to delete from the map
   */
  virtual void clearVoxelType(VoxelType voxel_type) = 0;

  /*!
   * \brief needsRebuild Checks, if map is fragmented and needs a rebuild.
   * Use this function in combination with 'rebuild()' to schedule map rebuilds on your own.
   * \return True, if rebuild is advised.
   */
  virtual bool needsRebuild() = 0;

  /*!
   * \brief rebuild Rebuilds the map to free memory.
   * Use this in combination with 'needsRebuild()' to suppress unneeded rebuilds.
   * Caution: This is time intensive!
   * \return True if successful, false otherwise.
   */
  virtual bool rebuild() = 0;

  /*!
   * \brief rebuildIfNeeded Rebuilds the map if required. Caution: A rebuild is time intensive!
   * \return True if successful, false otherwise.
   */
  virtual bool rebuildIfNeeded();

  /*!
   * \brief getMemoryUsage
   * \return Returns the size of the used memory in byte
   */
  virtual std::size_t getMemoryUsage() = 0;

  /*!
   * \brief getDimensions
   * \return Returns the dimensions of the map in voxels.
   */
  virtual Vector3ui getDimensions() = 0;

  /*!
   * \brief getMetricDimensions
   * \return Returns the dimensions of the map in meter.
   */
  virtual Vector3f getMetricDimensions() = 0;

protected:
  MapType m_map_type;
  file_handling::PointcloudFileHandler pointcloud_file_handler;
};

} // end of namespace
#endif
