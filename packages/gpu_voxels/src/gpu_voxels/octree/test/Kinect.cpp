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
#include "Kinect.h"

#include <stdio.h>
#include <iostream>
#include <math.h>
//using namespace std;

#include <gpu_voxels/octree/PointCloud.h>
#include <gpu_voxels/octree/Voxel.h>
#include <thrust/device_vector.h>

#include <pcl/io/oni_grabber.h>

namespace gpu_voxels {
namespace NTree {

Kinect::Kinect(Provider::Provider* provider, const Provider::Provider_Parameter* parameter) :
    SensorData(provider, parameter), m_parameter(parameter)
{
  m_running = false;
  m_working = false;
//  m_enable_raycasting = enable_raycasting;
//  m_cut_real_robot = cut_real_robot;
  m_counter = 0;
  m_frame = 0;
  m_frame_per_sec = parameter->kinect_fps;
  m_avg_callback = 0.0;
  m_data = new Vector3f[capture_size];
  //m_viewer =  pcl::visualization::CloudViewer("Sensor Data Viewer");
  //m_viewer.~CloudViewer();
}

//Kinect::Kinect(EnvironmentTree* ntree, Sensor* sensor, Vector3f voxel_dimension, bool enable_raycasting,
//               bool cut_real_robot) :
//    m_viewer("Sensor Data Viewer"), m_ntree(ntree), m_sensor(sensor), m_voxel_dimension(voxel_dimension), m_running(
//        false), m_enable_raycasting(enable_raycasting), m_cut_real_robot(cut_real_robot)
//{
//  m_data = new Vector3f[capture_size];
//}

Kinect::~Kinect()
{
  if (m_running)
  {
    stop();
  }
  delete[] m_data;
}

void Kinect::cloud_callback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{
  bool do_work = m_counter == 0;

  if (m_parameter->manual_kinect)
      m_counter++;
    else
    {
      if (m_frame_per_sec == 0)
        m_counter = 1;
      else
        m_counter = ((++m_counter) % uint32_t(30 / m_frame_per_sec));
    }

  if (do_work)
  {
#if defined(DEPTHCALLBACK_MESSAGES)  || defined(FEW_MESSAGES)
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "cloud_callback()" << endl);
#endif
//    std::string t = getTime_str();
//    const std::string filename = "./PCD_LOG/PCD_" + t + "_" + to_string((int) m_frame) + ".pcd";
//    pcl::io::savePCDFileASCII<pcl::PointXYZ>(filename, *cloud.get());

    timespec time_total = getCPUTime();
    for (uint32_t i = 0; i < cloud->points.size(); i++)
    {
      m_data[i].x = cloud->points[i].x;
      m_data[i].y = cloud->points[i].y;
      m_data[i].z = cloud->points[i].z;
      // printf("%f %f %f\n", cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
//
//    m_data[i].x = (float)1000.0*(cloud->points[i].x);
//    m_data[i].y = (float)1000.0*(cloud->points[i].y);
//    m_data[i].z = (float)1000.0*(cloud->points[i].z);

//    // cut kinect data to a specific range: (debugging)
//    const float max_range = 2500;
//    if (m_data[i].z > max_range)
//    {
//      m_data[i].x = NAN;
//      m_data[i].y = NAN;
//      m_data[i].z = NAN;
//    }

//printf("kinect point: %f, %f, %f\n", m_data[i].x,m_data[i].y, m_data[i].z);
    }

    m_provider->newSensorData(m_data, capture_size, width, height);

    double duration = timeDiff(time_total, getCPUTime());
    ++m_frame;
    m_avg_callback += duration;

#if defined(DEPTHCALLBACK_MESSAGES) || defined(FEW_MESSAGES)
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "cloud_callback: " << duration << " ms"<< endl);
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "AVG cloud_callback: " << m_avg_callback / m_frame << " ms" << endl);
#endif

    m_working = false;
  }
}

void Kinect::depth_callback(const boost::shared_ptr<openni_wrapper::DepthImage> &depth_image)
{
//  static const XnDepthPixel MAX_RANGE = 0xBB8;
//  static const XnDepthPixel MIN_RANGE = 0x01;
  //0x1FFF // ~8 m
  //0x1770 // 6 m
  //0xBB8 // 3 m

  bool do_work = m_counter == 0;
  if (m_parameter->manual_kinect)
      m_counter++;
    else
    {
      if (m_frame_per_sec == 0)
        m_counter = 1;
      else
        m_counter = ((++m_counter) % uint32_t(30 / m_frame_per_sec));
    }

  if (do_work)
  {
#if defined(DEPTHCALLBACK_MESSAGES)  || defined(FEW_MESSAGES)
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "depth_callback()" << endl);
#endif
    timespec time_total = getCPUTime();

#ifdef KINECT_PREPROCESSS_ON_GPU

    const XnDepthPixel* ptr = depth_image->getDepthMetaData().Data();

//    no sample value 0 no shadow value 0
//    width 640 height 480
//    constant_x 0.001753 constant_y 0.001753
//    centerX 319.500000 centerY 239.500000

    //   oni_wrapper->getConvertParameter(width, height, constant_x, constant_y, centerX, centerY);

//    float pitch = M_PI, yaw = 0, roll = M_PI / 2.0f;
//#ifdef PAN_TILT
//    double current_pan;
//    double current_tilt;
//    m_ptuController->Get_Position(current_pan, current_tilt);
//    usleep(50000);
//    if (m_ptuController->Get_Position(current_pan, current_tilt))
//    {
//      yaw = float(current_pan / 180 * M_PI) * (-1.0f);
//      //pitch = float(current_tilt / 180 * M_PI);
//      //printf("PTU: pan %f tilt %f\n", current_pan, current_tilt);
//      m_provider->updateSensorPose(yaw, pitch, roll);
//    }
//#else
//    m_provider->updateSensorPose(yaw, pitch, roll);
//#endif

//    ofstream log("Depthimage.txt");
//    for (uint32_t i = 0; i < width * height; ++i)
//    {
//      if((i % width) == 0)
//        log << std::endl;
//      log << ptr[i] << " ";
//    }
//    log.close();
//    exit(0);

    m_provider->newSensorData(ptr, width, height);

//    m_provider->newKinectData(ptr, width, height, constant_x, constant_y, centerX, centerY,
//                              depth_image->getNoSampleValue(), depth_image->getShadowValue());
#else
    uint32_t hist[MAX_RANGE] =
    { 0};

// ### sets unknown sensor values to max distance ###
    xn::DepthMetaData* my_meta_data = new xn::DepthMetaData();
    my_meta_data->CopyFrom(depth_image->getDepthMetaData());
    XnDepthPixel* ptr = my_meta_data->WritableData();
#ifdef DEPTHCALLBACK_MESSAGES
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "xres " << my_meta_data->XRes() << " yres "  << my_meta_data->YRes() << endl);
#endif
    for (uint32_t i = 0; i < my_meta_data->XRes() * my_meta_data->YRes(); ++i)
    {
//      if(ptr[i] == 0)
//        ptr[i] = 0x1FFF;
      if (ptr[i] == 0)
      ptr[i] = MAX_RANGE - 1;
      ptr[i] = std::min(std::max(ptr[i], MIN_RANGE), uint16_t(MAX_RANGE - 1));
      //++hist[0];
      // ++hist[ptr[i]];
    }

//    for (uint32_t i = 0; i < MAX_RANGE; ++i)
//    {
//      if (hist[i] != 0)
//        printf("%u -> # %u\n", i, hist[i]);
//    }

    boost::shared_ptr<xn::DepthMetaData> shared_ptr(my_meta_data);

    openni_wrapper::DepthImage* new_depth_image = new openni_wrapper::DepthImage(
        shared_ptr, depth_image->getBaseline(), depth_image->getFocalLength(), depth_image->getShadowValue(),
        depth_image->getNoSampleValue());
    const boost::shared_ptr<openni_wrapper::DepthImage> shared_depth_image(new_depth_image);
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > shared_point_cloud =
    oni_wrapper->convertToXYZPointCloud(shared_depth_image);

//    for (uint32_t x = 0; x < my_meta_data.XRes(); ++x)
//    {
//      for (uint32_t y = 0; y < my_meta_data.YRes(); ++y)
//      {
//        const XnDepthPixel* depth = &(my_meta_data)(x, y);
//        if (*depth == 0)
//          *depth = 0x1FFF;
////        printf("%X\n", *depth);
//      }
//    }

    float minZ = 99999, maxZ = 0;
    float minX = 99999, maxX = 0;
    float minY = 99999, maxY = 0;

    for (uint32_t i = 0; i < shared_point_cloud->points.size(); i++)
    {
      m_data[i].x = shared_point_cloud->points[i].x;
      m_data[i].y = shared_point_cloud->points[i].y;
      m_data[i].z = shared_point_cloud->points[i].z;

#ifdef DEPTHCALLBACK_MESSAGES
      if (isnanf(shared_point_cloud->points[i].z))
      {
        LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "NAN #" << i<< " "  << ptr[i] << endl);
      }
#endif

#ifdef DEPTHCALLBACK_MESSAGES
      minZ = std::min(shared_point_cloud->points[i].z, minZ);
      maxZ = std::max(shared_point_cloud->points[i].z, maxZ);

      minX = std::min(shared_point_cloud->points[i].x, minX);
      maxX = std::max(shared_point_cloud->points[i].x, maxX);

      minY = std::min(shared_point_cloud->points[i].y, minY);
      maxY = std::max(shared_point_cloud->points[i].y, maxY);
#endif

//      if (shared_point_cloud->points[i].z == 0.001000f)
//        printf("%f %f %f\n", shared_point_cloud->points[i].x, shared_point_cloud->points[i].y,
//               shared_point_cloud->points[i].z);
      //
      //    m_data[i].x = (float)1000.0*(cloud->points[i].x);
      //    m_data[i].y = (float)1000.0*(cloud->points[i].y);
      //    m_data[i].z = (float)1000.0*(cloud->points[i].z);

      //    // cut kinect data to a specific range: (debugging)
      //    const float max_range = 2500;
      //    if (m_data[i].z > max_range)
      //    {
      //      m_data[i].x = NAN;
      //      m_data[i].y = NAN;
      //      m_data[i].z = NAN;
      //    }

      //printf("kinect point: %f, %f, %f\n", m_data[i].x,m_data[i].y, m_data[i].z);
    }

#ifdef DEPTHCALLBACK_MESSAGES
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "minZ " << minZ << " maxZ "  << maxZ << " after convertToXYZPointCloud()" << endl);
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "minY " << minY << " maxY "  << maxY << " after convertToXYZPointCloud()" << endl);
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "minX " << minX << " maxX "  << maxX << " after convertToXYZPointCloud()" << endl);
#endif
#if defined(DEPTHCALLBACK_MESSAGES) || defined(FEW_MESSAGES)
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "cloud_callback preprocessing: "  << timeDiff(time_total, getCPUTime()) << " ms" << endl);
#endif

//    std::string file_name = "./KINECT_INPUT.pcd";
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_name, *cloud) == -1) //* load the file
//     {
//       PCL_ERROR("Couldn't read file test_pcd.pcd \n");
//     }
//     for (size_t i = 0; i < capture_size; ++i)
//       m_data[i] = Vector3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);

    // m_provider->newKinectData(m_data, capture_size);

#endif

    double duration = timeDiff(time_total, getCPUTime());

    ++m_frame;
    m_avg_callback += duration;

#if defined(DEPTHCALLBACK_MESSAGES) || defined(FEW_MESSAGES)
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "depth_callback: "  << duration << " ms" << endl);
    LOGGING_DEBUG_C(OctreeDepthCallbackLog, Kinect, "AVG depth_callback: "  << m_avg_callback / m_frame << " ms" << endl);
#endif

//    std::string t = getTime_str();
//    const std::string filename = "./PCD_LOG/PCD_" + t + "_" + to_string((int) m_frame) + ".pcd";
//    pcl::io::savePCDFileASCII<pcl::PointXYZ>(filename, *shared_point_cloud.get());

//    if (!m_viewer.wasStopped())
//    {
//      m_viewer.showCloud(shared_point_cloud);
//    }

    m_working = false;
  }
}

void Kinect::takeImage()
{
  m_working = true;
  m_counter = 0;

  while (m_working)
  {
    usleep(100000);
  }
}

void Kinect::run()
{
  LOGGING_INFO_C(OctreeDepthCallbackLog, Kinect, "Kinect: starting capture interface." << endl);
//  printf("  -  ray casting is %s\n", (m_enable_raycasting ? "on" : "off"));
//  printf("  -  cut robot   is %s\n", (m_cut_real_robot ? "on" : "off"));
  // oni_wrapper = new ONIWrapper();

  //m_interface = new pcl::OpenNIGrabber();

  if (m_parameter->mode == Provider::Provider_Parameter::MODE_KINECT_PLAYBACK)
    m_interface = new pcl::ONIGrabber(m_parameter->pc_file.c_str(), true, true);
  else
    m_interface = new pcl::OpenNIGrabber(m_parameter->kinect_id);

  //new pcl::ONIGrabber("./Recordings/Captured_0.oni", true, true);

  if (m_parameter->type == Provider::Provider_Parameter::TYPE_OCTOMAP
      || m_parameter->type == Provider::Provider_Parameter::TYPE_VOXEL_MAP)
  {
    boost::function<void(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> f_cb = boost::bind(
        &Kinect::cloud_callback, this, _1);
    m_interface->registerCallback(f_cb);
  }
  else
  {
    boost::function<void(const boost::shared_ptr<openni_wrapper::DepthImage>&)> f_cb = boost::bind(
        &Kinect::depth_callback, this, _1);
    m_interface->registerCallback(f_cb);
  }

//  boost::function<void(const openni_wrapper::DepthImage&)> f_cb = boost::bind(
//      &Kinect::depth_callback, this, _1);

  m_interface->start();
  m_running = true;
  LOGGING_INFO_C(OctreeDepthCallbackLog, Kinect, "Kinect: capture interface started." << endl);
}

void Kinect::stop()
{
  LOGGING_INFO_C(OctreeDepthCallbackLog, Kinect, "Kinect: stopping capture interface." << endl);
  while (m_working) // wait till work is done
    usleep(100000);

  m_interface->stop();
  delete m_interface;
  m_running = false;
  LOGGING_INFO_C(OctreeDepthCallbackLog, Kinect, "Kinect: capture interface stopped." << endl);
}

bool Kinect::isRunning()
{
  return m_running;
}

}
}
