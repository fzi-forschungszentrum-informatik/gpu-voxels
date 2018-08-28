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
#include <gpu_voxels/helpers/GeometryGeneration.h>
#include <gpu_voxels/vis_interface/VisVoxelMap.h>
#include <gpu_voxels/vis_interface/VisTemplateVoxelList.h>
#include <gpu_voxels/vis_interface/VisPrimitiveArray.h>
#include <gpu_voxels/octree/VisNTree.h>

namespace gpu_voxels {

GpuVoxels::GpuVoxels()
  :m_dim(0)
  ,m_voxel_side_length(0)
{
  // Check for valid GPU:
  if(!cuTestAndInitDevice())
  {
    exit(123);
  }
}

GpuVoxels::~GpuVoxels()
{
  // as the objects are shared pointers, they get deleted by this.
  m_managed_maps.clear();
  m_managed_robots.clear();
  m_managed_primitive_arrays.clear();
}

void GpuVoxels::initialize(const uint32_t dim_x, const uint32_t dim_y, const uint32_t dim_z, const float voxel_side_length)
{
  if(m_dim.x == 0 || m_dim.y == 0|| m_dim.z == 0 || m_voxel_side_length == 0)
  {
    m_dim.x = dim_x;
    m_dim.y = dim_y;
    m_dim.z = dim_z;
    m_voxel_side_length = voxel_side_length;
  }
  else
  {
    LOGGING_WARNING(Gpu_voxels, "Do not try to initialize GpuVoxels multiple times. Parameters remain unchanged." << endl);
  }
}

boost::weak_ptr<GpuVoxels> GpuVoxels::masterPtr = boost::weak_ptr<GpuVoxels>();

GpuVoxelsSharedPtr GpuVoxels::getInstance()
{
  boost::shared_ptr<GpuVoxels> temp = gpu_voxels::GpuVoxels::masterPtr.lock();
  if(!temp)
  {
    temp.reset(new GpuVoxels());
    gpu_voxels::GpuVoxels::masterPtr = temp;
  }
  return temp;
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

  primitive_array::PrimitiveArray* orig_prim_array = new primitive_array::PrimitiveArray(m_dim, m_voxel_side_length, prim_type);
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

bool GpuVoxels::modifyPrimitives(const std::string &array_name, const std::vector<Vector4f>& prim_positions)
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

bool GpuVoxels::modifyPrimitives(const std::string &array_name, const std::vector<Vector4i>& prim_positions)
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

bool GpuVoxels::modifyPrimitives(const std::string &array_name, const std::vector<Vector3f>& prim_positions, const float& diameter)
{
  ManagedPrimitiveArraysIterator it = m_managed_primitive_arrays.find(array_name);
  if (it == m_managed_primitive_arrays.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' not found." << endl);
    return false;
  }
  it->second.prim_array_shared_ptr->setPoints(prim_positions, diameter);
  return true;
}

bool GpuVoxels::modifyPrimitives(const std::string &array_name, const std::vector<Vector3i>& prim_positions, const uint32_t &diameter)
{
  ManagedPrimitiveArraysIterator it = m_managed_primitive_arrays.find(array_name);
  if (it == m_managed_primitive_arrays.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' not found." << endl);
    return false;
  }
  it->second.prim_array_shared_ptr->setPoints(prim_positions, diameter);
  return true;
}

GpuVoxelsMapSharedPtr GpuVoxels::addMap(const MapType map_type, const std::string &map_name)
{
  GpuVoxelsMapSharedPtr map_shared_ptr;
  VisProviderSharedPtr vis_map_shared_ptr;

  // check if map with same name already exists
  ManagedMapsIterator it = m_managed_maps.find(map_name);
  if (it != m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' already exists." << endl);

    return map_shared_ptr;  // null-initialized shared_ptr!
  }

  switch (map_type)
  {
    case MT_PROBAB_VOXELMAP:
    {
      voxelmap::ProbVoxelMap* orig_map = new voxelmap::ProbVoxelMap(m_dim, m_voxel_side_length, MT_PROBAB_VOXELMAP);
      VisVoxelMap* vis_map = new VisVoxelMap(orig_map, map_name);
      map_shared_ptr = GpuVoxelsMapSharedPtr(orig_map);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_map);
      break;
    }

    case MT_BITVECTOR_VOXELLIST:
    {
      voxellist::BitVectorVoxelList* orig_list = new voxellist::BitVectorVoxelList(m_dim, m_voxel_side_length, MT_BITVECTOR_VOXELLIST);
      VisTemplateVoxelList<BitVectorVoxel, uint32_t>* vis_list = new VisTemplateVoxelList<BitVectorVoxel, uint32_t>(orig_list, map_name);
      map_shared_ptr = GpuVoxelsMapSharedPtr(orig_list);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_list);
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
    {
      LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED << endl);
      throw GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED;
    }

    case MT_BITVECTOR_VOXELMAP:
    {
      voxelmap::BitVectorVoxelMap* orig_map = new voxelmap::BitVectorVoxelMap(m_dim, m_voxel_side_length, MT_BITVECTOR_VOXELMAP);
      VisVoxelMap* vis_map = new VisVoxelMap(orig_map, map_name);

      map_shared_ptr = GpuVoxelsMapSharedPtr(orig_map);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_map);
      break;
    }

    case MT_COUNTING_VOXELLIST:
    {
      voxellist::CountingVoxelList *orig_list =
          new voxellist::CountingVoxelList(m_dim, m_voxel_side_length, MT_COUNTING_VOXELLIST);
      VisTemplateVoxelList<CountingVoxel, uint32_t> *vis_list =
          new VisTemplateVoxelList<CountingVoxel, uint32_t>(orig_list, map_name);
      map_shared_ptr = GpuVoxelsMapSharedPtr(orig_list);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_list);
      break;
    }

    case MT_PROBAB_VOXELLIST:
    {
      LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED << endl);
      throw GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED;
    }

    case MT_PROBAB_OCTREE:
    {
      NTree::GvlNTreeProb* ntree = new NTree::GvlNTreeProb(m_voxel_side_length, MT_PROBAB_OCTREE);
      NTree::VisNTreeProb* vis_map = new NTree::VisNTreeProb(ntree, map_name);

      map_shared_ptr = GpuVoxelsMapSharedPtr(ntree);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_map);
      break;
    }

    case MT_PROBAB_MORTON_VOXELLIST:
    {
      LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED << endl);
      throw GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED;
    }

    case MT_DISTANCE_VOXELMAP:
    {
      voxelmap::DistanceVoxelMap* orig_map = new voxelmap::DistanceVoxelMap(m_dim, m_voxel_side_length, MT_DISTANCE_VOXELMAP);
      VisVoxelMap* vis_map = new VisVoxelMap(orig_map, map_name);

      map_shared_ptr = GpuVoxelsMapSharedPtr(orig_map);
      vis_map_shared_ptr = VisProviderSharedPtr(vis_map);
      break;
    }

    default:
    {
      LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "THIS TYPE OF MAP IS UNKNOWN!" << endl);
      throw GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED;
    }
  }

  if (map_shared_ptr)
  {
    std::pair<std::string, ManagedMap> named_map_pair(map_name,
                                                      ManagedMap(map_shared_ptr, vis_map_shared_ptr));
    m_managed_maps.insert(named_map_pair);

    // sanity checking, that nothing went wrong:
    CHECK_CUDA_ERROR();
    return map_shared_ptr;
  }
  else
  {
    throw "Map was not set";
  }


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

RobotInterfaceSharedPtr GpuVoxels::getRobot(const std::string &rob_name)
{
  ManagedRobotsIterator it = m_managed_robots.find(rob_name);
  if (it == m_managed_robots.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Robot with name '" << rob_name << "' not found." << endl);
    return RobotInterfaceSharedPtr();
  }
  return m_managed_robots.find(rob_name)->second;
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

#ifdef _BUILD_GVL_WITH_URDF_SUPPORT_
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
          robot_name, RobotInterfaceSharedPtr(new robot::UrdfRobot(m_voxel_side_length, path_to_urdf_file, use_model_path))));

  return true;
}
#endif

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
                                const robot::JointValueMap &jointmap)
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

bool GpuVoxels::getRobotConfiguration(const std::string& robot_name, robot::JointValueMap &jointmap)
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

bool GpuVoxels::insertPointCloudFromFile(const std::string map_name, const std::string path,
                                         const bool use_model_path, const BitVoxelMeaning voxel_meaning,
                                         const bool shift_to_zero, const Vector3f &offset_XYZ, const float scaling)
{
  ManagedMapsIterator map_it = m_managed_maps.find(map_name);
  if (map_it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
    return false;
  }

  return map_it->second.map_shared_ptr->insertPointCloudFromFile(path, use_model_path, voxel_meaning,
                                                                 shift_to_zero, offset_XYZ, scaling);

}

bool GpuVoxels::insertPointCloudIntoMap(const PointCloud &cloud, std::string map_name, const BitVoxelMeaning voxel_meaning)
{
  ManagedMapsIterator map_it = m_managed_maps.find(map_name);
  if (map_it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
    return false;
  }

  map_it->second.map_shared_ptr->insertPointCloud(cloud, voxel_meaning);

  return true;
}

bool GpuVoxels::insertPointCloudIntoMap(const std::vector<Vector3f> &cloud, std::string map_name, const BitVoxelMeaning voxel_meaning)
{
  ManagedMapsIterator map_it = m_managed_maps.find(map_name);
  if (map_it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
    return false;
  }

  map_it->second.map_shared_ptr->insertPointCloud(cloud, voxel_meaning);

  return true;
}

bool GpuVoxels::insertMetaPointCloudIntoMap(const MetaPointCloud &cloud, std::string map_name, const std::vector<BitVoxelMeaning>& voxel_meanings)
{
  ManagedMapsIterator map_it = m_managed_maps.find(map_name);
  if (map_it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
    return false;
  }

  map_it->second.map_shared_ptr->insertMetaPointCloud(cloud, voxel_meanings);

  return true;
}

bool GpuVoxels::insertMetaPointCloudIntoMap(const MetaPointCloud &cloud, std::string map_name, const BitVoxelMeaning voxel_meaning)
{
  ManagedMapsIterator map_it = m_managed_maps.find(map_name);
  if (map_it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
    return false;
  }

  map_it->second.map_shared_ptr->insertMetaPointCloud(cloud, voxel_meaning);

  return true;
}

bool GpuVoxels::insertRobotIntoMap(std::string robot_name, std::string map_name, const BitVoxelMeaning voxel_meaning)
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

  map_it->second.map_shared_ptr->insertMetaPointCloud(*rob_it->second->getTransformedClouds(), voxel_meaning);

  return true;
}



bool GpuVoxels::insertRobotIntoMapSelfCollAware(std::string robot_name, std::string map_name,
                                                const std::vector<BitVoxelMeaning>& voxel_meanings,
                                                const std::vector<BitVector<BIT_VECTOR_LENGTH> >& collision_masks,
                                                BitVector<BIT_VECTOR_LENGTH>* colliding_meanings)
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

  return map_it->second.map_shared_ptr->insertMetaPointCloudWithSelfCollisionCheck(rob_it->second->getTransformedClouds(),
                                                                                   voxel_meanings, collision_masks, colliding_meanings);

}

bool GpuVoxels::insertBoxIntoMap(const Vector3f &corner_min, const Vector3f &corner_max, std::string map_name, const BitVoxelMeaning voxel_meaning, uint16_t points_per_voxel)
{
  ManagedMapsIterator map_it = m_managed_maps.find(map_name);
  if (map_it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
    return false;
  }

  float delta = m_voxel_side_length / points_per_voxel;

  std::vector<Vector3ui> coordinates = geometry_generation::createBoxOfPoints(corner_min, corner_max, delta, m_voxel_side_length);
  map_it->second.map_shared_ptr->insertCoordinateList(coordinates, voxel_meaning);

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

bool GpuVoxels::clearMap(const std::string &map_name, BitVoxelMeaning voxel_meaning)
{
  ManagedMapsIterator it = m_managed_maps.find(map_name);
  if (it == m_managed_maps.end())
  {
    LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
    return false;
  }
  it->second.map_shared_ptr->clearBitVoxelMeaning(voxel_meaning);
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

void GpuVoxels::getDimensions(uint32_t& dim_x, uint32_t& dim_y, uint32_t& dim_z)
{
  dim_x = m_dim.x;
  dim_y = m_dim.y;
  dim_z = m_dim.z;
}

void GpuVoxels::getDimensions(Vector3ui &dim)
{
  dim = m_dim;
}

void GpuVoxels::getVoxelSideLength(float& voxel_side_length)
{
  voxel_side_length = m_voxel_side_length;
}

}
