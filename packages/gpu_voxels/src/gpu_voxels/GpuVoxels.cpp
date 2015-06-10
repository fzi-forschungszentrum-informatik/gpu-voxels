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
 */
//----------------------------------------------------------------------
#include "GpuVoxels.h"
#include <gpu_voxels/logging/logging_gpu_voxels.h>

namespace gpu_voxels {

using namespace boost::interprocess;

GpuVoxels::GpuVoxels(const uint32_t dim_x, const uint32_t dim_y, const uint32_t dim_z,
                     const float voxel_side_length) :
    m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z), m_voxel_side_length(voxel_side_length)
{
}

GpuVoxels::~GpuVoxels()
{
  // as the map objects are shared pointers, they get deleted by this.
  m_managed_maps.clear();
}

bool GpuVoxels::addPrimitives(const primitive_array::PrimitiveType prim_type, const std::string &array_name)
{
  // check if array with same name already exists
  ManagedPrimitiveArraysIterator it = m_managed_primitive_arrays.find(array_name);
  if (it != m_managed_primitive_arrays.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' already exists." << endl);
    return false;
  }

  primitive_array::PrimitiveArraySharedPtr primitive_array_shared_ptr;
  VisProviderSharedPtr vis_primitives_shared_ptr;

  primitive_array::PrimitiveArray* orig_prim_array = new primitive_array::PrimitiveArray(prim_type);
  VisPrimitiveArray* vis_prim_array = new VisPrimitiveArray(orig_prim_array, array_name);
  primitive_array_shared_ptr = primitive_array::PrimitiveArraySharedPtr(orig_prim_array);
  vis_primitives_shared_ptr = VisProviderSharedPtr(vis_prim_array);

  std::pair<std::string, ManagedPrimitiveArray> named_primitives_array_pair(array_name,
                                                    ManagedPrimitiveArray(primitive_array_shared_ptr, vis_primitives_shared_ptr));
  m_managed_primitive_arrays.insert(named_primitives_array_pair);
  return true;
}

bool GpuVoxels::delPrimitives(const std::string &array_name)
{
  ManagedPrimitiveArraysIterator it = m_managed_primitive_arrays.find(array_name);
  if (it == m_managed_primitive_arrays.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' not found." << endl);
    return false;
  }
  m_managed_primitive_arrays.erase(it);
  return true;
}

bool GpuVoxels::modifyPrimitives(const std::string &array_name, std::vector<Vector4f>& prim_positions)
{
  ManagedPrimitiveArraysIterator it = m_managed_primitive_arrays.find(array_name);
  if (it == m_managed_primitive_arrays.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' not found." << endl);
    return false;
  }
  it->second.prim_array_shared_ptr->setPoints(prim_positions);
  return true;
}

bool GpuVoxels::addMap(const MapType map_type, const std::string &map_name)
{
  // check if map with same name already exists
  ManagedMapsIterator it = m_managed_maps.find(map_name);
  if (it != m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' already exists." << endl);
    return false;
  }

  GpuVoxelsMapSharedPtr map_shared_ptr;
  VisProviderSharedPtr vis_map_shared_ptr;

  switch (map_type)
  {
    case MT_PROBAB_VOXELMAP:
    {
      voxelmap::VoxelMap* orig_map = new voxelmap::VoxelMap(m_dim_x, m_dim_y, m_dim_z, m_voxel_side_length, MT_PROBAB_VOXELMAP);
      VisVoxelMap* vis_map = new VisVoxelMap(orig_map, map_name);
      map_shared_ptr = GpuVoxelsMapSharedPtr(orig_map);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_map);
      break;
    }

    case MT_BITVECTOR_VOXELLIST:
    {
      break;
    }
    case MT_BITVECTOR_OCTREE:
    {
      NTree::GvlNTreeDet* ntree = new NTree::GvlNTreeDet(m_voxel_side_length, MT_BITVECTOR_OCTREE);
      NTree::VisNTreeDet* vis_map = new NTree::VisNTreeDet(ntree, map_name);

      map_shared_ptr = GpuVoxelsMapSharedPtr(ntree);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_map);
      break;
    }
    case MT_BITVECTOR_MORTON_VOXELLIST:
      break;

    case MT_BITVECTOR_VOXELMAP:
    {
      // todo move code to correct type e.g. MT_BITVECTOR_VOXELMAP
      voxelmap::BitVectorVoxelMap* orig_map = new voxelmap::BitVectorVoxelMap(m_dim_x, m_dim_y, m_dim_z,
                                                                              m_voxel_side_length, MT_BITVECTOR_VOXELMAP);
      VisVoxelMap* vis_map = new VisVoxelMap(orig_map, map_name);

      map_shared_ptr = GpuVoxelsMapSharedPtr(orig_map);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_map);
      break;
    }
    case MT_PROBAB_VOXELLIST:
      break;
    case MT_PROBAB_OCTREE:
    {
      NTree::GvlNTreeProb* ntree = new NTree::GvlNTreeProb(m_voxel_side_length, MT_PROBAB_OCTREE);
      NTree::VisNTreeProb* vis_map = new NTree::VisNTreeProb(ntree, map_name);

      map_shared_ptr = GpuVoxelsMapSharedPtr(ntree);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_map);
      break;
    }
    case MT_PROBAB_MORTON_VOXELLIST:
      break;
    default:
      return false;
  }

  if (map_shared_ptr)
  {
    std::pair<std::string, ManagedMap> named_map_pair(map_name,
                                                      ManagedMap(map_shared_ptr, vis_map_shared_ptr));
    m_managed_maps.insert(named_map_pair);
  }

  return true;
}

bool GpuVoxels::delMap(const std::string &map_name)
{
  ManagedMapsIterator it = m_managed_maps.find(map_name);
  if (it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
    return false;
  }
  m_managed_maps.erase(it);
  return true;
}

GpuVoxelsMapSharedPtr GpuVoxels::getMap(const std::string &map_name)
{
  ManagedMapsIterator it = m_managed_maps.find(map_name);
  if (it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
    return GpuVoxelsMapSharedPtr();
  }
  return m_managed_maps.find(map_name)->second.map_shared_ptr;
}

// ---------- Robot Stuff ------------
bool GpuVoxels::addRobot(const std::string &robot_name,
                         const std::vector<std::string> &link_names,
                         const std::vector<robot::DHParameters> &dh_params,
                         const std::vector<std::string> &paths_to_pointclouds,
                         const bool use_model_path)
{
  // check if robot with same name already exists
  ManagedRobotsIterator it = m_managed_robots.find(robot_name);
  if (it != m_managed_robots.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Robot with name '" << robot_name << "' already exists." << endl);
    return false;
  }

  m_managed_robots.insert(
      std::pair<std::string, RobotInterfaceSharedPtr>(
          robot_name, RobotInterfaceSharedPtr(
            new robot::KinematicChain(link_names, dh_params, paths_to_pointclouds, use_model_path))));

  return true;
}

bool GpuVoxels::addRobot(const std::string &robot_name, const std::vector<std::string> &link_names,
              const std::vector<robot::DHParameters> &dh_params,
              const MetaPointCloud &pointclouds)
{
  // check if robot with same name already exists
  ManagedRobotsIterator it = m_managed_robots.find(robot_name);
  if (it != m_managed_robots.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Robot with name '" << robot_name << "' already exists." << endl);
    return false;
  }

  m_managed_robots.insert(
      std::pair<std::string, RobotInterfaceSharedPtr>(
          robot_name, RobotInterfaceSharedPtr(
            new robot::KinematicChain(link_names, dh_params, pointclouds))));

  return true;

}

bool GpuVoxels::addRobot(const std::string &robot_name, const std::string &path_to_urdf_file, const bool use_model_path)
{
  // check if robot with same name already exists
  ManagedRobotsIterator it = m_managed_robots.find(robot_name);
  if (it != m_managed_robots.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Robot with name '" << robot_name << "' already exists." << endl);
    return false;
  }

  m_managed_robots.insert(
      std::pair<std::string, RobotInterfaceSharedPtr>(
          robot_name, RobotInterfaceSharedPtr(new robot::UrdfRobot(path_to_urdf_file, use_model_path))));

  return true;
}

bool GpuVoxels::updateRobotPart(std::string robot_name, const std::string &link_name, const std::vector<Vector3f> pointcloud)
{
  ManagedRobotsIterator it = m_managed_robots.find(robot_name);
  if (it == m_managed_robots.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find robot '" << robot_name << "'" << endl);
    return false;
  }

  it->second->updatePointcloud(link_name, pointcloud);
  return true;
}

bool GpuVoxels::setRobotConfiguration(std::string robot_name,
                                const std::map<std::string, float> &jointmap)
{
  ManagedRobotsIterator it = m_managed_robots.find(robot_name);
  if (it == m_managed_robots.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find robot '" << robot_name << "'" << endl);
    return false;
  }
  it->second->setConfiguration(jointmap);
  return true;
}

bool GpuVoxels::getRobotConfiguration(const std::string& robot_name, std::map<std::string, float> &jointmap)
{
  ManagedRobotsIterator rob_it = m_managed_robots.find(robot_name);
  if (rob_it == m_managed_robots.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find robot '" << robot_name << "'" << endl);
    return false;
  }
  rob_it->second->getConfiguration(jointmap);
  return true;
}

bool GpuVoxels::insertRobotIntoMap(std::string robot_name, std::string map_name, const VoxelType voxel_type)
{
  ManagedRobotsIterator rob_it = m_managed_robots.find(robot_name);
  if (rob_it == m_managed_robots.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find robot '" << robot_name << "'" << endl);
    return false;
  }
  ManagedMapsIterator map_it = m_managed_maps.find(map_name);
  if (map_it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
    return false;
  }

  map_it->second.map_shared_ptr->insertMetaPointCloud(*rob_it->second->getTransformedClouds(), voxel_type);

  return true;
}

bool GpuVoxels::insertBoxIntoMap(const Vector3f &corner_min, const Vector3f &corner_max, std::string map_name, const VoxelType voxel_type, uint16_t points_per_voxel)
{
  ManagedMapsIterator map_it = m_managed_maps.find(map_name);
  if (map_it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
    return false;
  }

  float delta = m_voxel_side_length / points_per_voxel;
  std::vector<Vector3f> box_cloud;

  for(float x_dim = corner_min.x; x_dim <= corner_max.x; x_dim += delta)
  {
    for(float y_dim = corner_min.y; y_dim <= corner_max.y; y_dim += delta)
    {
      for(float z_dim = corner_min.z; z_dim <= corner_max.z; z_dim += delta)
      {
        Vector3f point(x_dim, y_dim, z_dim);
        box_cloud.push_back(point);
      }
    }
  }
  std::vector<std::vector<Vector3f> > box_clouds;
  box_clouds.push_back(box_cloud);
  MetaPointCloud boxes(box_clouds);
  boxes.syncToDevice();

  map_it->second.map_shared_ptr->insertMetaPointCloud(boxes, voxel_type);

  return true;
}

bool GpuVoxels::clearMap(const std::string &map_name)
{
  ManagedMapsIterator it = m_managed_maps.find(map_name);
  if (it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
    return false;
  }
  it->second.map_shared_ptr->clearMap();
  return true;
}

bool GpuVoxels::clearMap(const std::string &map_name, VoxelType voxel_type)
{
  ManagedMapsIterator it = m_managed_maps.find(map_name);
  if (it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
    return false;
  }
  it->second.map_shared_ptr->clearVoxelType(voxel_type);
  return true;
}

bool GpuVoxels::visualizeMap(const std::string &map_name, const bool force_repaint)
{
  ManagedMapsIterator it = m_managed_maps.find(map_name);
  if (it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
    return false;
  }
  return it->second.vis_provider_shared_ptr.get()->visualize(force_repaint);
}

bool GpuVoxels::visualizePrimitivesArray(const std::string &prim_array_name, const bool force_repaint)
{
  ManagedPrimitiveArraysIterator it = m_managed_primitive_arrays.find(prim_array_name);
  if (it == m_managed_primitive_arrays.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives Array with name '" << prim_array_name << "' not found." << endl);
    return false;
  }
  return it->second.vis_provider_shared_ptr.get()->visualize(force_repaint);
}

VisProvider* GpuVoxels::getVisualization(const std::string &map_name)
{
  ManagedMapsIterator it = m_managed_maps.find(map_name);
  if (it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
    return NULL;
  }
  return it->second.vis_provider_shared_ptr.get();
}

void GpuVoxels::getDimensions(uint32_t& dim_x, uint32_t& dim_y, uint32_t& dim_z, float& voxel_side_length)
{
  dim_x = m_dim_x;
  dim_y = m_dim_y;
  dim_z = m_dim_z;
  voxel_side_length = m_voxel_side_length;
}

}
