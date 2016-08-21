#ifndef SEGMENTATIONHELPER_H
#define SEGMENTATIONHELPER_H

#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_classification/GroundPlaneSegmentationHelper.h>
#include <gpu_classification/ObjectSegmentationHelper.h>
#include <gpu_classification/DecisionTreeClassification.h>

#include <cuda_runtime.h>

namespace gpu_voxels {
namespace classification{



class SegmentationHelper
{
public:
  SegmentationHelper();

  /*
    * Takes the input pointcloud and runs the segmentation.
    * param pc: poinctloud to get segmented
    */
  void segmentPointCloud(std::vector<Vector3f> pc);
  void segmentPointCloud(std::vector<Vector3f> ibeo, Vector3f* velodyne, int size);
  void segmentPointCloud(Vector3f* pc, int size);

  /*
    * return: number of points in the ground pointcloud
    */
  int getGroundSize();

  /*
    * return: ground pointcloud
    */
  std::vector<Vector3f> getGround();

  /*
    * return: number of points in the nonground pointcloud
    */
  int getNonGroundSize();

  void setWithClassification(bool val);

  /*
    * return: nonground pointcloud
    */
  std::vector<Vector3f> getNonGround();

  /*
    * return Vector of the objects pointclouds
    */
  std::vector<std::vector<Vector3f> > getObjects();

  void setGroundSegmentationParameters(double segmentAngle, int numberOfBinsPerSegment, float rangeMin, float rangeMax, float thresholdLineGradient, float thresholdHorizontalLineGradient, float thresholdLowestGroundHeight, float thresholdDistancePointToLine);

  void setObjectSegmentationParameters(float relativeDistanceThreshold, float absoluteDistanceThreshold, float levelOfConcavity, float levelOfNoise, float zSimilarity, float aabbScaling, float normalMerge);

  void setClassificationClasses(std::vector<ClassificationClass> classes);

  std::vector<Segment> getSegments();

  int getSegmentsSize();

  int getNumberOfSegmentedPoints();

  void segmentGround(std::vector<Vector3f> pc);
  void segmentGround(std::vector<Vector3f> ibeo, Vector3f* velodyne, int size);
  void segmentGround(Vector3f* pc, int size);
  void segmentObjects(std::vector<Vector3f> nonground);
  void classifySegments(std::vector<Segment> segments);

private:

  /*
    * high level class of the ground segmentation algorithm
    */
  GroundPlaneSegmentationHelper gpsh;

  /*
    * high level class of the object segmentation algorithm
    */
  ObjectSegmentationHelper osh;

  /*
    * high level class of the decision tree classification algorithm
    */
  DecisionTreeClassification dtc;

  bool withClassification;

  // used for time measurement
  cudaEvent_t start;
  cudaEvent_t stop;

};



}//end namespace classification
}//end namespace gpu_voxels

#endif // SEGMENTATIONHELPER_H
