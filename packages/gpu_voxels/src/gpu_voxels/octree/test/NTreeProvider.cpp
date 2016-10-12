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
 * \date    2014-02-13
 *
 */
//----------------------------------------------------------------------
#include "NTreeProvider.h"
#include "VoxelMapProvider.h"
#include <gpu_voxels/vis_interface/VisualizerInterface.h>

//#include <gpu_voxels/octree/PointCloud.h>
//#include <gpu_voxels/octree/kernels/kernel_PointCloud.h>
#include <gpu_voxels/helpers/cuda_handling.h>
#include <gpu_voxels/octree/test/Helper.h>

#include <icl_core_performance_monitor/PerformanceMonitor.h>

#include <thrust/device_vector.h>

// CUDA
#include <cuda_runtime.h>

#include <sstream>
#include <istream>
#include <fstream>
#include <ostream>

using namespace std;

namespace gpu_voxels {
namespace NTree {
namespace Provider {

//const uint32_t BUFFER_SIZE = 10000000;

string ros_point_cloud_types[2] =
{ "front", "back" };

boost::mutex NTreeProvider::m_shared_mutex;

NTreeProvider::NTreeProvider() :
    Provider(), m_ntree(NULL), m_min_level(0), m_shm_superVoxelSize(NULL), m_shm_memHandle(NULL),
        m_shm_numCubes(NULL), m_shm_bufferSwapped(NULL), m_fps_rebuild(0), map_data_offset(0),
        m_spinner(NULL), m_node_handle(NULL), m_subscriber_front(NULL), m_subscriber_back(NULL),
        m_tf_listener(NULL), d_free_space_voxel(NULL), d_object_voxel(NULL),
        d_free_space_voxel2(NULL), d_object_voxel2(NULL), m_internal_buffer_1(false),
        m_d_cubes_1(NULL), m_d_cubes_2(NULL)
{
  m_segment_name = shm_segment_name_octrees;
}

NTreeProvider::~NTreeProvider()
{
  printf("NTreeProvider deconstructor called!\n");
  m_mutex.lock();

  if (m_spinner != NULL)
  {
    delete m_spinner;
    m_spinner = NULL;
  }
  if (m_node_handle != NULL)
  {
    delete m_node_handle;
    m_node_handle = NULL;
  }
  if (m_subscriber_front != NULL)
  {
    delete m_subscriber_front;
    m_subscriber_front = NULL;
  }
  if (m_subscriber_back != NULL)
  {
    delete m_subscriber_back;
    m_subscriber_back = NULL;
  }
  if (m_tf_listener != NULL)
  {
    delete m_tf_listener;
    m_tf_listener = NULL;
  }
  //  if (m_shm_superVoxelSize != NULL)
  //    m_segment.destroy_ptr(m_shm_superVoxelSize);
  //  printf("m_shm_memHandle\n");
  //  if (m_shm_memHandle != NULL)
  //    m_segment.destroy_ptr(m_shm_memHandle);
  //  printf("m_shm_numCubes\n");
  //  if (m_shm_numCubes != NULL)
  //    m_segment.destroy_ptr(m_shm_numCubes);
  //  printf("m_shm_bufferSwapped\n");
  //  if (m_shm_bufferSwapped != NULL)
  //    m_segment.destroy_ptr(m_shm_bufferSwapped);

  if (m_parameter->serialize)
  {
    string t = getTime_str();
    string filename = "./Octree_" + t + ".goc";
    printf("File '%s'\n", filename.c_str());
    ofstream out(filename.c_str());
    m_ntree->serialize(out);
    out.close();
  }

  printf("delete ntree\n");
  delete m_ntree;

  m_mutex.unlock();
  printf("NTreeProvider deconstructor finished!\n");
}

uint32_t NTreeProvider::generateCubes_wo_locking(Cube** ptr)
{
  uint8_t* selection_ptr = NULL;

#ifdef VISUALIZER_OBJECT_DATA_ONLY
  const uint32_t selection_size = 256;
  HANDLE_CUDA_ERROR(cudaMalloc((void** ) &selection_ptr, selection_size * sizeof(uint8_t)));
  uint8_t selection[selection_size];
  memset(&selection, 1, selection_size * sizeof(uint8_t));
// disable UNKNOWN and FREE
  for (uint32_t i = 0; i < selection_size; ++i)
  {
    if (((i & (ns_FREE | ns_UNKNOWN)) != 0) && ((i & ns_OCCUPIED) == 0))
    {
      selection[i] = 0;
    }
  }
  HANDLE_CUDA_ERROR(
      cudaMemcpy((void* ) selection_ptr, (void* ) &selection, selection_size * sizeof(uint8_t),
          cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#endif

  uint32_t cube_buffer_size;
  // m_internal_buffer tells, which buffer to use
  if(m_internal_buffer_1)
  {
    // extractCubes() allocates memory for the d_cubes_1, if the pointer is NULL
    cube_buffer_size = m_ntree->extractCubes(m_d_cubes_1, selection_ptr, m_min_level);
    *ptr = thrust::raw_pointer_cast(m_d_cubes_1->data());
    m_internal_buffer_1 = false;
  }else{
    // extractCubes() allocates memory for the d_cubes_2, if the pointer is NULL
    cube_buffer_size = m_ntree->extractCubes(m_d_cubes_2, selection_ptr, m_min_level);
    *ptr = thrust::raw_pointer_cast(m_d_cubes_2->data());
    m_internal_buffer_1 = true;
  }

  m_changed = false;

  return cube_buffer_size;
}

void NTreeProvider::visualize()
{
  m_shared_mutex.lock();
  m_mutex.lock();
  m_min_level = *m_shm_superVoxelSize - 1;

  Cube* ptr = NULL;
  uint32_t num_cubes = generateCubes_wo_locking(&ptr);

  HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, ptr));
  *m_shm_numCubes = num_cubes;
  *m_shm_bufferSwapped = true;
  m_mutex.unlock();
  m_shared_mutex.unlock();
}

void NTreeProvider::init(Provider_Parameter& parameter)
{
  m_mutex.lock();
  Provider::init(parameter);

  // setup sensor parameter
  m_sensor.object_data.m_initial_probability = INITIAL_OCCUPIED_PROBABILITY;
  m_sensor.object_data.m_update_probability = OCCUPIED_UPDATE_PROBABILITY;
  m_sensor.object_data.m_invalid_measure = 0;
  m_sensor.object_data.m_remove_max_range_data = true;
  m_sensor.object_data.m_sensor_range =
      parameter.sensor_max_range < 0 ? MAX_DETPTH_VALUE : parameter.sensor_max_range;
  m_sensor.object_data.m_use_invalid_measures = false;
  m_sensor.object_data.m_voxel_side_length = parameter.resolution_occupied;
  m_sensor.object_data.m_process_data = true;

  m_sensor.free_space_data = m_sensor.object_data; // copy data which doesn't matter

  // probabilities for free space aren't used for preprocessing of sensor data
  m_sensor.free_space_data.m_cut_x_boarder = KINECT_CUT_FREE_SPACE_X;
  m_sensor.free_space_data.m_cut_y_boarder = KINECT_CUT_FREE_SPACE_Y;
  m_sensor.free_space_data.m_invalid_measure = 0;
  m_sensor.free_space_data.m_remove_max_range_data = false;
  m_sensor.free_space_data.m_sensor_range =
      parameter.sensor_max_range < 0 ? MAX_DETPTH_VALUE : parameter.sensor_max_range;
  m_sensor.free_space_data.m_use_invalid_measures = true;
  m_sensor.free_space_data.m_voxel_side_length = parameter.resolution_free;
  m_sensor.free_space_data.m_process_data = true; // parameter.compute_free_space;

  // there should only be one segment of number_of_octrees
  std::pair<uint32_t*, std::size_t> r = m_segment.find<uint32_t>(shm_variable_name_number_of_octrees.c_str());
  if (r.second == 0)
  { // if it doesn't exist ..
    m_segment.construct<uint32_t>(shm_variable_name_number_of_octrees.c_str())(1);
  }
  else
  { // if it exit increase it by one
    (*r.first)++;
  }

  m_min_level = parameter.min_collision_level;

// get shared memory pointer
  m_shm_superVoxelSize = m_segment.find_or_construct<uint32_t>(shm_variable_name_super_voxel_size.c_str())(
      m_min_level + 1);
  m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(
      string(shm_variable_name_octree_handler_dev_pointer + m_shared_mem_id).c_str())(cudaIpcMemHandle_t());
  m_shm_numCubes = m_segment.find_or_construct<uint32_t>(
      string(shm_variable_name_number_cubes + m_shared_mem_id).c_str())(0);
  m_shm_bufferSwapped = m_segment.find_or_construct<bool>(
      string(shm_variable_name_octree_buffer_swapped + m_shared_mem_id).c_str())(false);
  *m_shm_bufferSwapped = false;

  const uint32_t blocks = parameter.num_blocks > 0 ? parameter.num_blocks : NUM_BLOCKS;
  const uint32_t threads = parameter.num_threads > 0 ? parameter.num_threads : NUM_THREADS_PER_BLOCK;

  m_ntree = new NTree<BRANCHING_FACTOR, LEVEL_COUNT, InnerNode, LeafNode>(blocks, threads,
                                                                          parameter.resolution_tree);

  if (parameter.mode == Provider_Parameter::MODE_DESERIALIZE)
  {
    printf("Deserialize '%s'\n", parameter.pc_file.c_str());

    ifstream in(parameter.pc_file.c_str());
    m_ntree->deserialize(in);
    in.close();
  }
  else
  {
    if (parameter.points.size() != 0)
    {
      Test::BuildResult<BRANCHING_FACTOR, LEVEL_COUNT, InnerNode, LeafNode> build_result;
      Test::buildOctree<BRANCHING_FACTOR, LEVEL_COUNT, InnerNode, LeafNode>(m_ntree, parameter.points,
                                                                            parameter.num_points,
                                                                            build_result, 1.0f,
                                                                            parameter.free_bounding_box,
                                                                            parameter.offset);
      map_data_offset = build_result.center;
    }
  }
  m_ntree->setMaxMemoryUsage(parameter.max_memory);

  m_sensor_position = gpu_voxels::Vector3f(m_ntree->m_center.x * m_ntree->m_resolution,
                                           m_ntree->m_center.y * m_ntree->m_resolution,
                                           m_ntree->m_center.z * m_ntree->m_resolution) * 0.001f; // position in meter
  m_sensor_orientation = gpu_voxels::Vector3f(0, 0, 0);

  printf("Octree mem usage: %f\n", double(m_ntree->getMemUsage()) * cBYTE2MBYTE);

  if (parameter.mode == Provider_Parameter::MODE_ROS)
  {
    // ROS
    int argc = 0;
    const char* argv = "";
    ros::init(argc, const_cast<char**>(&argv), "NTreeProvider");
    m_node_handle = new ros::NodeHandle();

    boost::function<void(const sensor_msgs::PointCloud2::ConstPtr& msg)> f_cb = boost::bind(
        &NTreeProvider::ros_point_cloud_front, this, _1);
    m_subscriber_front = new ros::Subscriber(m_node_handle->subscribe("/robot/point_cloud_front", 1, f_cb));
    ROS_INFO("Ready to receive /robot/point_cloud_front\n");

    boost::function<void(const sensor_msgs::PointCloud2::ConstPtr& msg)> f_cb2 = boost::bind(
        &NTreeProvider::ros_point_cloud_back, this, _1);
    m_subscriber_back = new ros::Subscriber(m_node_handle->subscribe("/robot/point_cloud_back", 1, f_cb2));
    ROS_INFO("Ready to receive /robot/point_cloud_back\n");

    m_spinner = new ros::AsyncSpinner(4);
    m_spinner->start();

    m_tf_listener = new tf::TransformListener();

    //ros::waitForShutdown();
    //ros::spin();
  }

  m_mutex.unlock();
}

void NTreeProvider::newSensorData(gpu_voxels::Vector3f* h_point_cloud, const uint32_t num_points,
                                  const uint32_t width, const uint32_t height)
{
  // not used
}

void NTreeProvider::newSensorData(const DepthData* h_depth_data, const uint32_t width, const uint32_t height)
{
// lock
  m_mutex.lock();

  const string prefix = __FUNCTION__;
  const string temp_timer = prefix + "_temp";
  PERF_MON_START(prefix);
  PERF_MON_START(temp_timer);

  gpu_voxels::Matrix3f orientation;

  gpu_voxels::Vector3f temp = m_sensor_orientation;
#ifdef MODE_KINECT
  orientation = gpu_voxels::Matrix3f::createFromYPR(KINECT_ORIENTATION.z, KINECT_ORIENTATION.y, KINECT_ORIENTATION.x);
  temp.z *= -1; // invert to fix the incorrect positioning for ptu-mode
#endif

  m_sensor.data_width = width;
  m_sensor.data_height = height;

  m_sensor.pose = Matrix4f::createFromRotationAndTranslation(
        Matrix3f::createFromYPR(temp.z, temp.y, temp.x) * orientation, m_sensor_position);

  const uint32_t ntree_resolution = m_ntree->m_resolution;

  // processSensorData() will allcate space for d_free_space_voxel and d_object_voxel if they are NULL
  m_sensor.processSensorData(h_depth_data, d_free_space_voxel, d_object_voxel);

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "processSensorData", prefix);

  // convert sensor origin in discrete coordinates of the NTree
  gpu_voxels::Vector3ui sensor_origin = gpu_voxels::Vector3ui(
      uint32_t(m_sensor_position.x * 1000.0f / ntree_resolution),
      uint32_t(m_sensor_position.y * 1000.0f / ntree_resolution),
      uint32_t(m_sensor_position.z * 1000.0f / ntree_resolution));

  m_ntree->insertVoxel(*d_free_space_voxel, *d_object_voxel, sensor_origin, m_parameter->resolution_free,
                       m_parameter->resolution_occupied);

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "insertVoxel", prefix);

//  if (m_ntree->checkTree())
//    exit(0);

#ifdef DO_REBUILDS
  if (m_ntree->needsRebuild()
      || (m_parameter->rebuild_frame_count != -1 && m_fps_rebuild == m_parameter->rebuild_frame_count))
  {
    printf("OLD allocInnerNodes %u, allocLeafNodes %u \n", m_ntree->allocInnerNodes, m_ntree->allocLeafNodes);
    m_ntree->rebuild();
    printf("NEW allocInnerNodes %u, allocLeafNodes %u \n", m_ntree->allocInnerNodes, m_ntree->allocLeafNodes);
    m_fps_rebuild = 0;
  }
#endif

// ################# COLLIDE ################
  if (m_collide_with != NULL)
  {
    m_collide_with->lock();

    PERF_MON_START(temp_timer);

    collide_wo_locking();

    PERF_MON_PRINT_INFO_P(temp_timer, "Collide", prefix);

    m_collide_with->unlock();
  }

  m_changed = true;
  ++m_fps_rebuild;

  PERF_MON_PRINT_INFO_P(prefix, "", prefix);

  m_mutex.unlock();

// exit(0);
}

void NTreeProvider::collide_wo_locking()
{
  const string prefix = __FUNCTION__;

  if (m_collide_with != NULL)
  {
    voxel_count num_collisions = 0;
    m_min_level = *m_shm_superVoxelSize - 1;

    if (NTreeProvider* _ntree = dynamic_cast<NTreeProvider*>(m_collide_with))
    {
      if (_ntree->m_ntree->m_resolution != m_ntree->m_resolution)
      {
        printf("ERROR: Both NTrees must have the same resolution (voxel side length)!\n");
        exit(0);
      }
      num_collisions = m_ntree->intersect_load_balance(_ntree->m_ntree, m_min_level, DefaultCollider(),
                                                       m_parameter->save_collisions);
      if (m_parameter->clear_collisions)
        m_ntree->clearCollisionFlags();
      m_changed = true;
      m_collide_with->setChanged(true);
    }
    else if (VoxelMapProvider* _provider = dynamic_cast<VoxelMapProvider*>(m_collide_with))
    {
      // compute offset since NTree centers its data in the middle of the tree, to handle the problem of only positive coordinates
      // compute offset so the NTree/Voxelmap intersection uses the same data as NTree/NTree intersection and therfore gets the same result
      Vector3i voxelmap_offset(0);
//      Vector3ui voxelmap_offset = m_ntree->m_center
//          - (_provider->getVoxelMap()->getDimensions() / gpu_voxels::Vector3ui(2));

      if (voxelmap::ProbVoxelMap* _voxelmap = dynamic_cast<voxelmap::ProbVoxelMap*>(_provider->getVoxelMap()))
      {
        printf("res octree %u res vmpa %f\n", m_ntree->m_resolution, _voxelmap->getVoxelSideLength());
        if (m_ntree->m_resolution != uint32_t(_voxelmap->getVoxelSideLength() * 1000.0f))
        {
          printf("ERROR: Both NTrees must have the same resolution (voxel side length)!\n");
          exit(0);
        }

        if (m_parameter->voxelmap_intersect_with_lb)
        {
          // todo Fix this and make a template for intersect_load_balance() and voxelmap
//          if (m_parameter->save_collisions)
//            num_collisions = m_ntree->intersect_load_balance<VOXELMAP_FLAG_SIZE, true, false,
//                gpu_voxels::Voxel, true>(*_voxelmap, voxelmap_offset, m_min_level, NULL);
//          else
//            num_collisions = m_ntree->intersect_load_balance<VOXELMAP_FLAG_SIZE, false, false,
//                gpu_voxels::Voxel, true>(*_voxelmap, voxelmap_offset, m_min_level, NULL);
        }
        else
        {
          if (m_parameter->save_collisions)
          {
            printf("Collide with collisions\n");
            num_collisions = m_ntree->intersect_sparse<true, false, false, ProbabilisticVoxel>(
                *_voxelmap, NULL, m_min_level, voxelmap_offset);
          }
          else
            num_collisions = m_ntree->intersect_sparse<false, false, false, ProbabilisticVoxel>(
                *_voxelmap, NULL, m_min_level, voxelmap_offset);
        }
      }
      if (m_parameter->clear_collisions)
        m_ntree->clearCollisionFlags();
      m_changed = true;
      m_collide_with->setChanged(true);
    }
    PERF_MON_ADD_DATA_NONTIME_P("NumCollisions", num_collisions, prefix);
  }
}

void NTreeProvider::collide()
{
  if (m_collide_with != NULL)
  {
    m_mutex.lock();
    m_collide_with->lock();

    collide_wo_locking();

    m_collide_with->unlock();
    m_mutex.unlock();
  }
}

bool NTreeProvider::waitForNewData(volatile bool* stop)
{
// wait till new data is required
  while (!*stop && !m_changed && m_min_level == (*m_shm_superVoxelSize - 1))
  {
    //boost::this_thread::yield();
    usleep(buffer_watch_delay);
  }
// wait till current buffer was read
  while (*m_shm_bufferSwapped && !*stop)
  {
    //boost::this_thread::yield();
    usleep(buffer_watch_delay);
  }
  return !*stop;
}

void NTreeProvider::ros_point_cloud(const sensor_msgs::PointCloud2::ConstPtr& msg, const uint32_t type)
{
  printf("ros callback\n");
  // lock
  m_mutex.lock();

  tf::StampedTransform transform;
  bool eval = false;
  if (m_node_handle->ok())
  {
    string name = ros_point_cloud_types[type];
    stringstream str_stream;
    str_stream << "/camera_depth_optical_frame_" << name;
    string from_frame = str_stream.str();
    string to_frame = "/odom";
    if (m_tf_listener->canTransform(to_frame, from_frame, ros::Time(0)))
    {
      try
      {
        m_tf_listener->lookupTransform(to_frame, from_frame, ros::Time(0), transform);
      } catch (tf::TransformException ex)
      {
        ROS_ERROR("%s", ex.what());
      }
      eval = true;
    }
  }

  if (eval)
  {
    const string prefix = "newSensorData";
    const string temp_timer = prefix + "_temp";
    PERF_MON_START(prefix);
    PERF_MON_START(temp_timer);

    std::size_t size = msg->height * msg->width;
    std::vector<Vector3f> points(size);
    for (std::size_t i = 0; i < size; ++i)
    {
      float* x = (float*) &msg->data[i * 16];
      float* y = (float*) &msg->data[i * 16 + 4];
      float* z = (float*) &msg->data[i * 16 + 8];

      Vector3f tmp = Vector3f(NAN, NAN, NAN);
      if (*z < 4.0)
      {
        if (points[i].z <= 2.5)
          tmp = Vector3f(*x, *y, *z);
      }
      points[i] = tmp;
    }

    m_sensor.data_width = msg->width;
    m_sensor.data_height = msg->height;

    // copy roation matrix
    m_sensor.pose.setIdentity();

    tf::Matrix3x3 m(transform.getRotation());
    m_sensor.pose.a11 = m[0].getX();
    m_sensor.pose.a12 = m[0].getY();
    m_sensor.pose.a13 = m[0].getZ();
    m_sensor.pose.a21 = m[1].getX();
    m_sensor.pose.a22 = m[1].getY();
    m_sensor.pose.a23 = m[1].getZ();
    m_sensor.pose.a31 = m[2].getX();
    m_sensor.pose.a32 = m[2].getY();
    m_sensor.pose.a33 = m[2].getZ();

    tf::Vector3 tmp = transform.getOrigin();
//    m_sensor.position = m_sensor_position
//        + gpu_voxels::Vector3f(tmp.getX() - 122, tmp.getY() + 82, tmp.getZ());
    Vector3f shift_origin = Vector3f(0, 0, 0);
    Vector3f sensor_position = m_sensor_position + gpu_voxels::Vector3f(tmp.getX() - shift_origin.x, tmp.getY() - shift_origin.y, tmp.getZ() - shift_origin.z);

    m_sensor.pose.a14 = sensor_position.x;
    m_sensor.pose.a24 = sensor_position.y;
    m_sensor.pose.a34 = sensor_position.z;

    // processSensorData() will allcate space for d_free_space_voxel and d_object_voxel if they are NULL
    m_sensor.processSensorData(&points.front(), d_free_space_voxel2, d_object_voxel2);
    printf("Num points %lu\n", d_free_space_voxel2->size());

    // convert sensor origin in discrete coordinates of the NTree
    const uint32_t ntree_resolution = m_ntree->m_resolution;
    gpu_voxels::Vector3ui sensor_origin = gpu_voxels::Vector3ui(
        uint32_t(sensor_position.x * 1000.0f / ntree_resolution),
        uint32_t(sensor_position.y * 1000.0f / ntree_resolution),
        uint32_t(sensor_position.z * 1000.0f / ntree_resolution));

    m_ntree->insertVoxel(*d_free_space_voxel2, *d_object_voxel2, sensor_origin, m_parameter->resolution_free,
                         m_parameter->resolution_occupied);

    PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "insertVoxel", prefix);

    //  if (m_ntree->checkTree())
    //    exit(0);

#ifdef DO_REBUILDS
    if (m_ntree->needsRebuild()
        || (m_parameter->rebuild_frame_count != -1 && m_fps_rebuild == m_parameter->rebuild_frame_count))
    {
      printf("BEFORE: Octree mem usage: %f\n", double(m_ntree->getMemUsage()) * cBYTE2MBYTE);
      printf("OLD allocInnerNodes %u, allocLeafNodes %u \n", m_ntree->allocInnerNodes,
             m_ntree->allocLeafNodes);
      m_ntree->rebuild();
      printf("NEW allocInnerNodes %u, allocLeafNodes %u \n", m_ntree->allocInnerNodes,
             m_ntree->allocLeafNodes);
      m_fps_rebuild = 0;
      printf("AFTER: Octree mem usage: %f\n", double(m_ntree->getMemUsage()) * cBYTE2MBYTE);
    }
#endif

    // ################# COLLIDE ################
    if (m_collide_with != NULL)
    {
      m_collide_with->lock();

      PERF_MON_START(temp_timer);

      collide_wo_locking();

      PERF_MON_PRINT_INFO_P(temp_timer, "Collide", prefix);

      m_collide_with->unlock();
    }

    m_changed = true;
    ++m_fps_rebuild;

    PERF_MON_PRINT_INFO_P(prefix, "", prefix);
  }
  m_mutex.unlock();
}

void NTreeProvider::ros_point_cloud_back(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  ros_point_cloud(msg, 1);
}

void NTreeProvider::ros_point_cloud_front(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  ros_point_cloud(msg, 0);
}

}
}
}
