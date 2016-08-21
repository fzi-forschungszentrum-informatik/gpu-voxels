#include "IbeoNcomFilter.h"

#include <icl_sourcesink/SimpleURI.h>
#include <icl_velodyne/VelodynePCLPointCloud.h>
#include <gpu_classification/logging/logging_classification.h>

namespace gpu_voxels {
namespace classification {

using namespace gpu_voxels;
using namespace icl_sourcesink;
using namespace icl_hardware::ibeo;


IbeoNcomFilter::IbeoNcomFilter(const std::string& uri,
                               const std::string &name)
  : DataFilter<IbeoMsg, std::vector<Vector3f> >(uri, name),
    pointCloudName("myFilterCloud"),
    m_data(new icl_core::Stamped<std::vector<Vector3f> >),
    mpc(),
    m_additional_translation(20.0f, 20.0f, 0.0f),
    gpu("myFilterCloud"),
    sh()
{
  *m_data = std::vector<Vector3f>();
  ref_pose = NULL;

  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.1);

  //create new Map
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myFilterMap");

  cloud_number = 0;

  currentNumberOfPointclouds = 0;
  iterationCount = 0;

  displayGround = true;
  displayObjects = true;
  displayReference = false;

  velodyneRotationAngle = 3.38f;
}

IbeoNcomFilter::~IbeoNcomFilter()
{
}

void IbeoNcomFilter::set(const icl_core::Stamped<IbeoMsg>::Ptr &data)
{
  filter(*data, *m_data);
}

void IbeoNcomFilter::setNcomData(const icl_core::Stamped<icl_hardware::ncom::Ncom>::Ptr& data)
{
  ncom_element = data;
}

void IbeoNcomFilter::setVelodynePointCloudData(const icl_core::Stamped<VelodynePCLPointCloud::PointCloud>::Ptr& data)
{
  velodyne_element = data;
}

//void IbeoNcomFilter::setVelodynePointCloudDataStamped(const icl_core::Stamped<VelodynePCLPointCloud::PointCloud> data)
//{
//    velodyne_stamped = data;
//}

void IbeoNcomFilter::setVelodyneRotationAngle(float angle)
{
  velodyneRotationAngle = angle;
}

void IbeoNcomFilter::setFilterParameter(bool displayGround, bool displayObjects, bool displayReference, bool displayClasses, bool classificationToggle, Vector3f add_trans)
{
  LOGGING_INFO(Classification, "displayGround: " << displayGround << " displayObjects: " << displayObjects << " Trans: " << add_trans << endl);
  this->displayGround = displayGround;
  this->displayObjects = displayObjects;
  this->displayReference = displayReference;
  this->displayClasses = displayClasses;
  this->m_additional_translation = add_trans;
  sh.setWithClassification(classificationToggle);
}

void IbeoNcomFilter::setGroundSegmentationParameters(double segmentAngle, int numberOfBinsPerSegment, float rangeMin, float rangeMax, float thresholdLineGradient, float thresholdHorizontalLineGradient, float thresholdLowestGroundHeight, float thresholdDistancePointToLine)
{
  LOGGING_INFO(Classification, "segmentAngle: " << segmentAngle << " numberOfBinsPerSegment: " << numberOfBinsPerSegment << " rangeMin: " << rangeMin << " rangeMax: " << rangeMax << " thresholdLineGradient: " << thresholdLineGradient << " thresholdHorizontalLineGradient: " << thresholdHorizontalLineGradient << " thresholdLowestGroundHeight: " << thresholdLowestGroundHeight << " thresholdDistancePointToLine: " << thresholdDistancePointToLine << endl);
  sh.setGroundSegmentationParameters(segmentAngle, numberOfBinsPerSegment, rangeMin, rangeMax, thresholdLineGradient, thresholdHorizontalLineGradient, thresholdLowestGroundHeight, thresholdDistancePointToLine);
}

void IbeoNcomFilter::setObjectSegmentationParameters(float relativeDistanceThreshold, float absoluteDistanceThreshold, float levelOfConcavity, float levelOfNoise, float zSimilarity, float aabbScaling, float normalMerge)
{
  LOGGING_INFO(Classification, "relativeDistanceThreshold: " << relativeDistanceThreshold << " absoluteDistanceThreshold:" << absoluteDistanceThreshold << " levelOfConcavity: " << levelOfConcavity << " levelOfNoise: " << levelOfNoise << " zSimilarity: " << zSimilarity << " aabbScaling: " << aabbScaling << " normalMerge: " << normalMerge << endl);
  sh.setObjectSegmentationParameters(relativeDistanceThreshold, absoluteDistanceThreshold, levelOfConcavity, levelOfNoise, zSimilarity, aabbScaling, normalMerge);
}

void IbeoNcomFilter::setClassificationClasses(std::vector<ClassificationClass> classes)
{
  sh.setClassificationClasses(classes);
}

bool IbeoNcomFilter::filter(const icl_core::Stamped<IbeoMsg> &input_data, icl_core::Stamped<std::vector<Vector3f> > &output_data)
{
  uint displayedIteration = 1;

  IbeoScanMsg scan_msg;
  if(!scan_msg.fromIbeoMsg(input_data))
    return false;

  VelodynePCLPointCloud::PointCloud velodyne_pointcloud = *velodyne_element;
  //VelodynePCLPointCloud::PointCloud velodyne_p = velodyne_stamped.get();

  uint32_t velodynePointCloudSize = 0;

  if(velodyne_pointcloud.isOrganized())
  {
    velodynePointCloudSize = velodyne_pointcloud.width * velodyne_pointcloud.height;
  }
  else
  {
    velodynePointCloudSize = velodyne_pointcloud.width;
  }

  //Copy points to std::vector
  /*std::vector<Vector3f> ibeoCloud(scan_msg.number_of_points);
    std::vector<Vector3f> velodyneCloud(velodynePointCloudSize);
    std::vector<Vector3f> point_cloud;*/
  ibeoCloud = std::vector<Vector3f>(scan_msg.number_of_points * 1);
  velodyneCloud = std::vector<Vector3f>(velodynePointCloudSize * 1);

  for(int i = 0; i < scan_msg.number_of_points; i++)
  {
    ibeoCloud[i].x = (*scan_msg.scan_points)[i].x;
    ibeoCloud[i].y = (*scan_msg.scan_points)[i].y;
    ibeoCloud[i].z = (*scan_msg.scan_points)[i].z;
  }

  for(uint32_t i = 0; i < velodynePointCloudSize; i++)
  {
    VelodynePCLPointCloud::PointVelodyne point = velodyne_pointcloud.points.at(i);
    velodyneCloud[i].x = point.x;
    velodyneCloud[i].y = point.y;
    velodyneCloud[i].z = point.z;

  }

  Matrix4f velodyneTransformMatrix;
  velodyneTransformMatrix.a11 = cosf(velodyneRotationAngle);
  velodyneTransformMatrix.a12 = (-1) * sinf(velodyneRotationAngle);
  velodyneTransformMatrix.a21 = sinf(velodyneRotationAngle);
  velodyneTransformMatrix.a22 = cosf(velodyneRotationAngle);

  velodyneTransformMatrix.a14 = 0.95f;
  velodyneTransformMatrix.a24 = -0.22f;
  velodyneTransformMatrix.a34 = 1.64741f;

  velodyneTransformMatrix.a33 = 1.0f;
  velodyneTransformMatrix.a44 = 1.0f;



  //Vector3f* velodyneCloud_d = new Vector3f();
  //int* velodyneCloudSize = new int;

  if(displayGround || displayObjects)
  {
    //gpu.transformPointCloudTest(&velodyneTransformMatrix, &velodyneCloud, velodyneCloud_d, velodyneCloudSize);
    gpu.transformPointCloudSTD(&velodyneTransformMatrix, &velodyneCloud);

  }

  if(displayReference)
  {
    gpu.transformPointCloudSTD(&velodyneTransformMatrix, &velodyneCloud);
  }

  point_cloud = std::vector<Vector3f>();
  point_cloud.reserve(ibeoCloud.size() + velodyneCloud.size());
  point_cloud.insert(point_cloud.end(), ibeoCloud.begin(), ibeoCloud.end());
  point_cloud.insert(point_cloud.end(), velodyneCloud.begin(), velodyneCloud.end());

  LOGGING_INFO(Filter, "Loaded Pointcloud with " << point_cloud.size() << " Points from (Ibeo: " << scan_msg.number_of_points << " Velodyne: " << velodynePointCloudSize << ")" << endl);

  //Handle position
  icl_hardware::ncom::Ncom ncom_msg = *ncom_element;

  //icl_gps::AbsolutePose abs_pose(ncom_msg->mLon, ncom_msg->mLat, ncom_msg->mAlt, ncom_msg->mTrack);
  icl_gps::NEDPose ned_pose(ncom_msg->mLon, ncom_msg->mLat, ncom_msg->mAlt, ncom_msg->mTrack);

  //set reference pose if not already existing
  if(!ref_pose)
  {
    //ref_pose = new icl_gps::AbsolutePose(abs_pose.longitude(), abs_pose.latitude(), abs_pose.altitude(), abs_pose.bearing());
    ref_pose = new icl_gps::NEDPose(ned_pose.longitude(), ned_pose.latitude(), ned_pose.altitude(), ned_pose.bearing());
  }
  icl_gps::RelativePose rel_pose = ned_pose - *ref_pose;
  icl_math::Pose2d rel_pose2d = rel_pose.toPose2d();


  Matrix4f matrix;

  //rotation matrix around z-axis
  matrix.a11 = rel_pose2d(0,0);   matrix.a12 = rel_pose2d(0,1);
  matrix.a21 = rel_pose2d(1,0);   matrix.a22 = rel_pose2d(1,1);

  //translation vector
  matrix.a14 = rel_pose2d(0,2);
  matrix.a24 = rel_pose2d(1,2);

  //structure of transformation matrix
  matrix.a33 = 1.0f;
  matrix.a44 = 1.0f;

  //z-translation
  matrix.a34 = static_cast<float>(rel_pose.altitude());

  //additional translation
  matrix.a14 += m_additional_translation.x;
  matrix.a24 += m_additional_translation.y;
  matrix.a34 += m_additional_translation.z;// + 0.1f * iterationCount;

  transform_matrix = matrix;

  printf("\n(%f, %f, %f, %f)\n(%f, %f, %f, %f)\n(%f, %f, %f, %f)\n(%f, %f, %f, %f)\n",
         transform_matrix.a11, transform_matrix.a12, transform_matrix.a13, transform_matrix.a14,
         transform_matrix.a21, transform_matrix.a22, transform_matrix.a23, transform_matrix.a24,
         transform_matrix.a31, transform_matrix.a32, transform_matrix.a33, transform_matrix.a34,
         transform_matrix.a41, transform_matrix.a42, transform_matrix.a43, transform_matrix.a44);



  //printf("rein (%f, %f, %f)\n", point_cloud[200].x, point_cloud[200].y, point_cloud[200].z);

  processPointcloud();

  if(iterationCount > displayedIteration)
  {
    Matrix4f translateMatrix;
    translateMatrix.a14 = m_additional_translation.x;
    translateMatrix.a24 = m_additional_translation.y;
    translateMatrix.a34 = m_additional_translation.z;
    translateMatrix.a11 = 1.0f;
    translateMatrix.a22 = 1.0f;
    translateMatrix.a33 = 1.0f;
    translateMatrix.a44 = 1.0f;

    transformMPC = new MetaPointCloud(mpc);

    //MetaPointCloud temp(mpc);

    gpu.transformPointCloud(&translateMatrix, &mpc, transformMPC);

    visualizePointClouds(transformMPC, objectMeanings);

    //Output of the Filter

    //const Vector3f* pc = mpc.getPointCloud(cloud_number);
    std::vector<Vector3f> transformed_pointcloud(1);
    transformed_pointcloud[0] = Vector3f(1.0f, 2.0f, 3.0f);
    //printf("raus (%f, %f, %f)\n", transformed_pointcloud[200].x, transformed_pointcloud[200].y, transformed_pointcloud[200].z);
    *m_data = transformed_pointcloud;
    output_data = *m_data;

  }


  //visualizePointClouds(transformMPC, objectMeanings);


  //testTransformPointcloud();

  iterationCount++;
  return true;

}

void IbeoNcomFilter::processPointcloud()
{

  LOGGING_INFO(Filter, "Loaded Pointcloud with " << point_cloud.size() << endl);

  if(displayGround || displayObjects)
  {
    //sh.segmentPointCloud(ibeoCloud, velodyneCloud_d, *velodyneCloudSize);
    sh.segmentPointCloud(point_cloud);
  }

  std::vector<Vector3f> ground(sh.getGroundSize());
  ground = sh.getGround();
  std::vector<std::vector<Vector3f> > objects = sh.getObjects();


  currentNumberOfPointclouds = mpc.getNumberOfPointclouds();
  if(currentNumberOfPointclouds == 0)
  {
    if(displayGround)
    {
      LOGGING_INFO(Filter, "Adding " << (int)ground.size() << " Points as ground" << endl);
      mpc.addCloud(ground, false, "GroundCloud");
      //cloud_number = mpc.getCloudNumber("GroundCloud");
    }

    if(displayObjects)
    {
      LOGGING_INFO(Filter, "Adding " << sh.getNumberOfSegmentedPoints() << " of Points in " << (int)objects.size() << " Segments" << endl);
      for(uint i = 0; i < objects.size(); i++)
      {
        std::stringstream ss;
        ss << "ObjectCloud" << i;
        std::string tempS = ss.str();
        mpc.addCloud(objects[i], false, tempS);
      }
    }
    if(displayReference)
    {
      mpc.addCloud(ibeoCloud, false, "ReferenceIbeoCloud");
      mpc.addCloud(velodyneCloud, false, "ReferenceVelodyneCloud");
      //cloud_number = mpc.getCloudNumber("ReferenceCloud");
    }

  }
  else
  {
    if(displayGround)
    {
      LOGGING_INFO(Filter, "Updating " << (int)ground.size() << " Points as ground" << endl);
      mpc.updatePointCloud("GroundCloud", ground, false);
    }

    if(displayObjects)
    {

      //TODO: deleting of pointclouds if there are too many unused pointclouds
      // for example deleting of pointclouds, if segmentCount < currentNumberOfPointclouds / 2


      LOGGING_INFO(Filter, "Updating " << sh.getNumberOfSegmentedPoints() << " of Points in " << (int)objects.size() << " Segments " << currentNumberOfPointclouds << endl);
      if(currentNumberOfPointclouds < objects.size() + 1)
      {
        LOGGING_INFO(Filter, "Adding " << objects.size() - currentNumberOfPointclouds << " segments" << endl);
        for(uint i = currentNumberOfPointclouds; i < objects.size(); i++)
        {
          std::stringstream ss;
          ss << "ObjectCloud" << i;
          std::string tempS = ss.str();
          mpc.addCloud(objects[i], false, tempS);
        }
      }

      currentNumberOfPointclouds = mpc.getNumberOfPointclouds();
      LOGGING_INFO(Filter, "Number of Objects: " << (int)objects.size() << " Number of Clouds: " << currentNumberOfPointclouds << endl);
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
          std::vector<Vector3f> emptyCloud(1);
          emptyCloud.push_back(Vector3f(-1.0f, -1.0f, -1.0f));
          mpc.updatePointCloud(tempS, emptyCloud, false);
        }
      }
    }

    if(displayReference)
    {
      mpc.updatePointCloud("ReferenceIbeoCloud", ibeoCloud, false);
      mpc.updatePointCloud("ReferenceVelodyneCloud", velodyneCloud, false);
    }
  }

  mpc.syncToDevice();

  std::vector<Segment> segments = sh.getSegments();

  objectMeanings = std::vector<gpu_voxels::BitVoxelMeaning>(mpc.getNumberOfPointclouds());

  LOGGING_INFO(Filter, "Segments Size: " << segments.size() << " objectMeanings Size:" << objectMeanings.size() << endl);

  //objectMeanings[0] = static_cast<gpu_voxels::BitVoxelMeaning>(99);
  objectMeanings[0] = eBVM_OCCUPIED;
  int tempMeaning = 11;
  if(displayObjects)
  {
    //int tempMeaning = 10;
    for(uint i = 1; i < objectMeanings.size(); i++)
    {
      if(!displayClasses)
      {
        if(tempMeaning >= 99)
        {
          tempMeaning = 11;
        }
        objectMeanings[i] = static_cast<gpu_voxels::BitVoxelMeaning>(tempMeaning);
        tempMeaning++;
      }
      else
      {
        if(i < segments.size())
        {
          objectMeanings[i] = segments[i].meaning;
          LOGGING_TRACE(Filter, "Segment " << i << " segmented with meaning: " << segments[i].meaning << endl);
        }
        else
        {
          objectMeanings[i] = static_cast<gpu_voxels::BitVoxelMeaning>(19);
          LOGGING_TRACE(Filter, "Segment " << i << " segmented with no meaning" << endl);
        }
      }

    }
  }
  else if(displayReference)
  {
    objectMeanings[0] = static_cast<gpu_voxels::BitVoxelMeaning>(11);
  }
}

void IbeoNcomFilter::redrawPointcloud()
{
  Matrix4f translateMatrix;
  translateMatrix.a14 = m_additional_translation.x;
  translateMatrix.a24 = m_additional_translation.y;
  translateMatrix.a34 = m_additional_translation.z;
  translateMatrix.a11 = 1.0f;
  translateMatrix.a22 = 1.0f;
  translateMatrix.a33 = 1.0f;
  translateMatrix.a44 = 1.0f;

  transformMPC = new MetaPointCloud(mpc);

  //MetaPointCloud temp(mpc);

  gpu.transformPointCloud(&translateMatrix, &mpc, transformMPC);

  visualizePointClouds(transformMPC, objectMeanings);
}

void IbeoNcomFilter::visualizePointClouds(MetaPointCloud* mpc, std::vector<gpu_voxels::BitVoxelMeaning> bitMeanings)
{
  gvl->clearMap("myFilterMap");

  gvl->getMap("myFilterMap")->insertMetaPointCloud(*mpc, bitMeanings);

  gvl->visualizeMap("myFilterMap");
}

void IbeoNcomFilter::visualizePointCloud(MetaPointCloud* mpc)
{
  gvl->clearMap("myFilterMap");

  gvl->getMap("myFilterMap")->insertMetaPointCloud(*mpc, eBVM_OCCUPIED);

  gvl->visualizeMap("myFilterMap");
}

icl_core::Stamped<std::vector<Vector3f> >::Ptr IbeoNcomFilter::get() const
{
  return m_data;
}

}//end of namespace classification
}//end of namespace gpu_voxels
