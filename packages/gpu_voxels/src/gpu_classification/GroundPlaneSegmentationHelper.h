#ifndef GROUNDPLANESEGMENTATIONHELPER_H
#define GROUNDPLANESEGMENTATIONHELPER_H

#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <thrust/host_vector.h>
#include <gpu_classification/logging/logging_classification.h>


namespace gpu_voxels{
namespace classification{

class GroundPlaneSegmentationHelper
{
public:

  //Constructor
  GroundPlaneSegmentationHelper(double segmentAngle, int bins, float rangeMin, float rangeMax, float thresholdLineGradient, float thresholdHorizontalLineGradient, float thresholdLowestGroundHeight, float thresholdDistancePointToLine);

  /*!
    * \param pc: pointcloud
    */
  void segmentGround(std::vector<Vector3f> pc);

  /*!
    * \param ibeo: vector of ibeo pointcloud
    * \param velodyne: pointer of velodyne pointcloud
    * \param size: size of the velodyne pointcloud
    */
  void segmentGround(std::vector<Vector3f> ibeo, Vector3f* velodyne, int size);

  /*!
    * \param pc: pointer to pointcloud
    * \param size: size of the pointcloud
    */
  void segmentGround(Vector3f* pc, int size);

  /*!
    * \return size of the vector of the points which belong to the ground
    */
  int getGroundSize();

  /*!
    * \return size of the vector of the points which don't belong to the ground
    */
  int getNonGroundSize();

  /*!
    * \return points which belong to the ground
    */
  std::vector<Vector3f> getGround();

  /*!
    * \return points which don't belong to the ground
    */
  std::vector<Vector3f> getNonGround();

  /*!
    * used to set all parameters after instantiation of this class
    */
  void setParameters(double segmentAngle, int numberOfBinsPerSegment, float rangeMin, float rangeMax, float thresholdLineGradient, float thresholdHorizontalLineGradient, float thresholdLowestGroundHeight, float thresholdDistancePointToLine);

private:

  // angle of one segment in degree
  double segmentAngle;

  // angle of one segment in radian
  double segmentRadian;

  //number of segments
  int segmentCount;

  //number of bins per segment
  int binPerSegment;

  //minimum range from the center for points to be processed
  float rangeMin;

  //maximum range from the center for points to be processed
  float rangeMax;

  //maximum gradient of groundline
  float thresholdLineGradient;

  //minimum gradient the groundline is allowed to have before it is treated as horizontal
  float thresholdHorizontalLineGradient;

  //
  float thresholdLowestGroundHeight;

  //maximum distance for each point to its groundline
  float thresholdDistancePointToLine;

  //range each of the bin covers
  thrust::host_vector<float2> binRanges_h;

  //initializes all devices_vectors with the new pointcloud. needs to be called before each run of the algorithm.
  void initialize(Vector3f* pc, int size);

  //calculates for each point in which segment it belongs.
  void calcSegments();

  //calculates for each point in which bin it belongs in its segment.
  void calcBins();

  //project each point to a plane which divides the segment of the point in half
  void convertTo2D();

  //extracts representative points for each bin
  void createPrototypePoints();

  //estimates a line through the prototypepoints
  void createLinePerSegment();

  //marks whether a point belongs to the ground
  void labelPointsAsGround();

  //calculates in which distance range each bin covers
  void fillBinRanges();

  //returns the index of the corresponding bin for the given distance
  int getIndexOfBin(double range);

  //calculates the distance from point to line
  float distancePointToSegmentLine(Vector3f p, float2 l);


  //bool isGround(int index);
};


}//end namespace classification
}//end namespace gpu_voxels

#endif // GROUNDPLANESEGMENTATIONHELPER_H
