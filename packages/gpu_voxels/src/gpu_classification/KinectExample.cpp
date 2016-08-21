#include <icl_core/os.h>
#include <icl_core_logging/Logging.h>
#include <icl_core_config/Config.h>
#include <gpu_classification/SegmentationHelper.h>
#include <gpu_classification/filter/IbeoNcomFilterGPU.h>
#include <gpu_voxels/GpuVoxels.h>
#include <gpu_classification/logging/logging_classification.h>


#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <tf/transform_listener.h>

#include <dynamic_reconfigure/server.h>
#include "classificator_reconfigure/ClassificatorConfig.h"

#include <cstdlib>
#include <signal.h>
#include <cstdio>


typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

void ctrlchandler(int)
{
  ros::shutdown();
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  ros::shutdown();
  exit(EXIT_SUCCESS);
}

std::vector<gpu_voxels::Vector3f> point_cloud;
bool changed = false;
gpu_voxels::MetaPointCloud mpc;
gpu_voxels::classification::SegmentationHelper segmentation;
gpu_voxels::GpuVoxelsSharedPtr gvl;
gpu_voxels::classification::IbeoNcomFilterGPU* gpu;
gpu_voxels::MetaPointCloud* transformMPC;

gpu_voxels::Vector3f addTrans;

tf::TransformListener* m_tf_listener;
PointCloud m_global_kinect_cloud;

bool displayReference = false;

void reconfigureCallback(classificator_reconfigure::ClassificatorConfig &config, uint32_t level)
{
  /*ROS_INFO("Reconfigure Request: %d %f %s %s %d",
            config.int_param, config.double_param,
            config.str_param.c_str(),
            config.bool_param?"True":"False",
            config.size);*/

  ROS_INFO("Reconfigure Callback: %s, %d", config.className.c_str(), config.meaning);

  //  stopPreview = config.stopPreview;

  segmentation.setGroundSegmentationParameters(config.segmentAngle, config.numberOfBins, config.rangeMin, config.rangeMax,
                                               config.thresholdLineGradient, config.thresholdHorizontalLineGradient,
                                               config.thresholdLowestGroundHeight, config.thresholdDistancePointToLine);
  segmentation.setObjectSegmentationParameters(config.relativeDistanceThreshold, config.absoluteDistanceThreshold,
                                               config.levelOfConcavity, config.levelOfNoise, config.zSimilarity,
                                               config.aabbScaling, config.normalMerge);

  addTrans = gpu_voxels::Vector3f(config.additionalTransX, config.additionalTransY, config.additionalTransZ);

  displayReference = config.displayReference;
}

void kinectCallback(const PointCloud::ConstPtr& msg)
{
  LOGGING_INFO(gpu_voxels::classification::Kinect, "Callback with " << msg->points.size() << " Points" << gpu_voxels::classification::endl);

  tf::StampedTransform stamped_transform;
  if(m_tf_listener->waitForTransform(msg->header.frame_id, "/world", ros::Time(0), ros::Duration(5.0), ros::Duration(0.05)))
  {
    try
    {
      m_tf_listener->lookupTransform(msg->header.frame_id, "/world",
                                     ros::Time(0), stamped_transform);
    }
    catch (tf::TransformException ex){
      ROS_ERROR("pointCloud CB error: %s",ex.what());
    }
  }
  else
  {
    ROS_ERROR("SharedWorkspaceDemo::pointCloudCB: Could not find a transform from 'cam' to 'world' for 5 seconds!");
    return;
  }

  pcl_ros::transformPointCloud(*msg, m_global_kinect_cloud, stamped_transform.inverse());


  point_cloud.clear();
  point_cloud.resize(m_global_kinect_cloud.points.size());
  for(uint i = 0; i < m_global_kinect_cloud.points.size(); i++)
  {
    point_cloud[i] = gpu_voxels::Vector3f(m_global_kinect_cloud.points[i].x, m_global_kinect_cloud.points[i].y, m_global_kinect_cloud.points[i].z);
  }

  std::vector<gpu_voxels::Vector3f> ground;
  std::vector<std::vector<gpu_voxels::Vector3f> > objects = std::vector<std::vector<gpu_voxels::Vector3f> >();
  if(!displayReference)
  {
    segmentation.segmentGround(point_cloud);

    ground = std::vector<gpu_voxels::Vector3f>(segmentation.getGroundSize());
    ground = segmentation.getGround();
    objects.push_back(segmentation.getNonGround());
    //objects = segmentation.getObjects();

    LOGGING_INFO(gpu_voxels::classification::Kinect, "Finished Segmentation" << gpu_voxels::classification::endl);

  }
  else
  {
    ground = point_cloud;
  }


  //insert in metapointcloud
  uint currentNumberOfPointclouds = mpc.getNumberOfPointclouds();
  if(currentNumberOfPointclouds == 0)
  {
    LOGGING_INFO(gpu_voxels::classification::Kinect, "Adding " << (int)ground.size() << " Points as ground" << gpu_voxels::classification::endl);
    mpc.addCloud(ground, false, "GroundCloud");

    LOGGING_INFO(gpu_voxels::classification::Kinect, "Adding " << ((int)point_cloud.size() - (int)ground.size()) << " of Points in " << (int)objects.size() << " Segments" << gpu_voxels::classification::endl);
    for(uint i = 0; i < objects.size(); i++)
    {
      std::stringstream ss;
      ss << "ObjectCloud" << i;
      std::string tempS = ss.str();
      mpc.addCloud(objects[i], false, tempS);
    }
  }
  else
  {
    LOGGING_INFO(gpu_voxels::classification::Kinect, "Updating " << (int)ground.size() << " Points as ground" << gpu_voxels::classification::endl);
    mpc.updatePointCloud("GroundCloud", ground, false);

    LOGGING_INFO(gpu_voxels::classification::Kinect, "Updating " << ((int)segmentation.getNumberOfSegmentedPoints()) << " of Points in " << (int)objects.size() << " Segments" << gpu_voxels::classification::endl);
    if(currentNumberOfPointclouds < objects.size() + 1)
    {
      for(uint i = currentNumberOfPointclouds; i < objects.size(); i++)
      {
        std::stringstream ss;
        ss << "ObjectCloud" << i;
        std::string tempS = ss.str();
        mpc.addCloud(objects[i], false, tempS);
      }
    }

    currentNumberOfPointclouds = mpc.getNumberOfPointclouds();
    LOGGING_INFO(gpu_voxels::classification::Kinect, "Number of Objects: " << (int)objects.size() << " Number of Clouds: " << currentNumberOfPointclouds << gpu_voxels::classification::endl);
    for(uint i = 0; i < currentNumberOfPointclouds - 1; i++)
    {
      std::stringstream ss;
      ss << "ObjectCloud" << i;
      std::string tempS = ss.str();
      if(i < objects.size())
      {
        mpc.updatePointCloud(tempS, objects[i], false);
      }
      else
      {
        std::vector<gpu_voxels::Vector3f> emptyCloud(1);
        emptyCloud.push_back(gpu_voxels::Vector3f(-1.0f, -1.0f, -1.0f));
        mpc.updatePointCloud(tempS, emptyCloud, false);
      }
    }

  }

  mpc.syncToDevice();


  //generate bitvoxel meanings
  std::vector<gpu_voxels::BitVoxelMeaning> objectMeanings = std::vector<gpu_voxels::BitVoxelMeaning>(mpc.getNumberOfPointclouds());

  objectMeanings[0] = gpu_voxels::eBVM_OCCUPIED;

  int tempMeaning = 11;
  for(uint i = 1; i < objectMeanings.size(); i++)
  {
    if(tempMeaning >= 20)
    {
      tempMeaning = 10;
    }
    objectMeanings[i] = static_cast<gpu_voxels::BitVoxelMeaning>(tempMeaning);
    tempMeaning++;
  }


  //visualize pointcloud
  gpu_voxels::Matrix4f translateMatrix;
  translateMatrix.a14 = addTrans.x;
  translateMatrix.a24 = addTrans.y;
  translateMatrix.a34 = addTrans.z;
  translateMatrix.a11 = 1.0f;
  translateMatrix.a22 = 1.0f;
  translateMatrix.a33 = 1.0f;
  translateMatrix.a44 = 1.0f;

  transformMPC = new gpu_voxels::MetaPointCloud(mpc);

  //MetaPointCloud temp(mpc);

  gpu->transformPointCloud(&translateMatrix, &mpc, transformMPC);

  gvl->clearMap("myKinectMap");

  gvl->getMap("myKinectMap")->insertMetaPointCloud(*transformMPC, objectMeanings);

  gvl->visualizeMap("myKinectMap");

  delete transformMPC;
}


int main(int argc, char *argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  ros::init(argc, argv, "kinectexample_node", ros::init_options::NoSigintHandler);

  icl_core::logging::initialize(argc, argv);

  //initialize

  gvl = gpu_voxels::GpuVoxels::getInstance();
  gvl->initialize(300, 300, 300, 0.01);
  gvl->addMap(gpu_voxels::MT_BITVECTOR_VOXELLIST, "myKinectMap");


  gpu = new gpu_voxels::classification::IbeoNcomFilterGPU("myKinectMap");


  segmentation.setGroundSegmentationParameters(icl_core::config::getDefault<double>("/groundsegmentation/segmentAngle", 10.0f),
                                               icl_core::config::getDefault<int>("/groundsegmentation/numberOfBins", 40),
                                               icl_core::config::getDefault<float>("/groundsegmentation/rangeMin", 0.0f),
                                               icl_core::config::getDefault<float>("/groundsegmentation/rangeMax", 1000.0f),
                                               icl_core::config::getDefault<float>("/groundsegmentation/thresholdLineGradient", 2.0f),
                                               icl_core::config::getDefault<float>("/groundsegmentation/thresholdHorizontalLineGradient", 0.1f),
                                               icl_core::config::getDefault<float>("/groundsegmentation/thresholdLowestGroundHeight", 2.0f),
                                               icl_core::config::getDefault<float>("/groundsegmentation/thresholdDistancePointToLine", 1.5f));
  segmentation.setObjectSegmentationParameters(icl_core::config::getDefault<float>("/objectsegmentation/relativeDistanceThreshold", 5.0f),
                                               icl_core::config::getDefault<float>("/objectsegmentation/absoluteDistanceThreshold", 100.0f),
                                               icl_core::config::getDefault<float>("/objectsegmentation/levelOfConcavity", 0.523f),
                                               icl_core::config::getDefault<float>("/objectsegmentation/levelOfNoise", 0.172f),
                                               icl_core::config::getDefault<float>("/objectsegmentation/zSimilarity", 1.0f),
                                               icl_core::config::getDefault<float>("/objectsegmentation/aabbScaling", 1.0f),
                                               icl_core::config::getDefault<float>("/objectsegmentation/normaleMerge", 0.8f));
  segmentation.setWithClassification(false);


  dynamic_reconfigure::Server<classificator_reconfigure::ClassificatorConfig> server;
  dynamic_reconfigure::Server<classificator_reconfigure::ClassificatorConfig>::CallbackType f;

  f = boost::bind(&reconfigureCallback, _1, _2);
  server.setCallback(f);

  ros::NodeHandle n;
  ros::Subscriber kinect_sub = n.subscribe("camera/depth/points", 1, kinectCallback);

  m_tf_listener = new tf::TransformListener();

  //    ros::AsyncSpinner spinner(1);
  //    spinner.start();

  LOGGING_INFO(gpu_voxels::classification::Kinect, "Finished Initialization" << gpu_voxels::classification::endl);

  ros::Rate r(30);
  while(ros::ok())
  {
    ros::spinOnce();
    r.sleep();
  }

}
