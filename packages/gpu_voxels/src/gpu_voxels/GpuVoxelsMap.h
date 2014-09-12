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
#include <gpu_voxels/helpers/pcd_handling.h>

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

  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, VoxelType voxel_type) = 0;

  /*!
   * \brief translateContent This does a base transform about the given coordinates.
   * Offset is calculated and Voxel-Adresses are altered accordingly.
   * Important: For performance reasons do not use this function to model dynamic activities!
   *
   * \param x The offset x coordinate
   * \param y The offset y coordinate
   * \param z The offset z coordinate
   */
  void translateContent(int32_t x, int32_t y, int32_t z);

  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param other The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWith(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0) = 0;

  /*!
   * \brief collideWithRelativeTransform This does a base transform about the given coordinates
   * followed by a collision check with 'other'.
   * Important: This is only performant for VoxelLists!
   * \param other The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param x The offset x coordinate
   * \param y The offset y coordinate
   * \param z The offset z coordinate
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  size_t collideWithRelativeTransform(const GpuVoxelsMapSharedPtr other, float coll_threshold, int32_t x,
                                      int32_t y, int32_t z);

  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param other The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param resolution_level The resolution used for collision checking. resolution_level = 0 delivers the highest accuracy whereas each increase haves the resolution.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithResolution(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0, const uint32_t resolution_level = 0) = 0;

  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param other The map to do a collision check with.
   * \param types_in_collision The voxel types in collision.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithTypes(const GpuVoxelsMapSharedPtr other, voxelmap::BitVectorVoxel&  types_in_collision, float coll_threshold = 1.0) = 0;

  /*!
   * \brief insertPCD inserts a pointcloud into the map
   * The coordinates are interpreted as global coordinates
   * \param path filename
   * \param shift_to_zero if true, the map will be shifted, so that its minimum lies at zero.
   * \param offset_XYZ if given, the map will be transformed by this XYZ offset. If shifting is active, this happens after the shifting.
   * \return true if succeeded, false otherwise
   */
  bool insertPCD(const std::string path, VoxelType voxel_type, const bool shift_to_zero = false,
                 const Vector3f &offset_XYZ = Vector3f());

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

protected:
  MapType m_map_type;
};

} // end of namespace
#endif
