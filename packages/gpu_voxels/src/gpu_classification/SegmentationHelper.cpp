#define IC_PERFORMANCE_MONITOR
#include "SegmentationHelper.h"
#include <icl_core_performance_monitor/PerformanceMonitor.h>


namespace gpu_voxels{
namespace classification {



SegmentationHelper::SegmentationHelper()
  :gpsh(10.0, 50, 0.1f, 200.0f, 1000.0f, 0.00001f, 1000.0f, 1.0f),
    osh(5.0f, 100.0f, 0.523f, 0.172f, 0.5f),
    dtc()
{

  PERF_MON_INITIALIZE(10, 1000);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
}


void SegmentationHelper::segmentPointCloud(std::vector<Vector3f> ibeo, Vector3f* velodyne, int size)
{
  segmentGround(ibeo, velodyne, size);

  segmentObjects(getNonGround());

  if(withClassification)
  {
    classifySegments(osh.getSegments());
  }
}

void SegmentationHelper::segmentPointCloud(Vector3f* pc, int size)
{
  segmentGround(pc, size);

  segmentObjects(getNonGround());

  if(withClassification)
  {
    classifySegments(osh.getSegments());
  }
}

void SegmentationHelper::segmentPointCloud(std::vector<Vector3f> pc)
{
  segmentGround(pc);

  segmentObjects(getNonGround());

  if(withClassification)
  {
    classifySegments(osh.getSegments());
  }
}

void SegmentationHelper::segmentGround(std::vector<Vector3f> pc)
{

  PERF_MON_START("groundSegmentation");
  PERF_MON_ENABLE("ground_prefix");
  PERF_MON_ADD_DATA_NONTIME_P("number of Points: ", pc.size(), "ground_prefix");
  gpsh.segmentGround(pc);
  PERF_MON_ADD_DATA_NONTIME_P("number of GroundPoints: ", getGroundSize(), "ground_prefix");
  PERF_MON_PRINT_INFO_P("groundSegmentation", "Groundsegmentation finished", "ground_prefix");
  //PERF_MON_SUMMARY_ALL_INFO;

}
void SegmentationHelper::segmentGround(std::vector<Vector3f> ibeo, Vector3f* velodyne, int size)
{
  HANDLE_CUDA_ERROR(cudaEventRecord(start, 0));
  gpsh.segmentGround(ibeo, velodyne, size);
  HANDLE_CUDA_ERROR(cudaEventRecord(stop, 0));

  HANDLE_CUDA_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  LOGGING_INFO(SegmentationHelp, "GroundPlaneSegmentation TIME: " << elapsedTime << "ms " << endl);
}
void SegmentationHelper::segmentGround(Vector3f* pc, int size)
{
  HANDLE_CUDA_ERROR(cudaEventRecord(start, 0));
  gpsh.segmentGround(pc, size);
  HANDLE_CUDA_ERROR(cudaEventRecord(stop, 0));

  HANDLE_CUDA_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  LOGGING_INFO(SegmentationHelp, "GroundPlaneSegmentation TIME: " << elapsedTime << "ms " << endl);
}

void SegmentationHelper::segmentObjects(std::vector<Vector3f> nonground)
{
  PERF_MON_START("objectSegmentation");
  PERF_MON_ENABLE("object_prefix");
  PERF_MON_ADD_DATA_NONTIME_P("number of Points: ", nonground.size(), "object_prefix");
  osh.segmentObjects(nonground);
  PERF_MON_ADD_DATA_NONTIME_P("number of Segments: ", getObjects().size(), "object_prefix");
  PERF_MON_PRINT_INFO_P("objectSegmentation", "Objectsegmentation finished", "object_prefix");
  //PERF_MON_SUMMARY_ALL_INFO;
}

void SegmentationHelper::classifySegments(std::vector<Segment> segments)
{
  HANDLE_CUDA_ERROR(cudaEventRecord(start, 0));
  dtc.classifySegments(segments);
  HANDLE_CUDA_ERROR(cudaEventRecord(stop, 0));

  HANDLE_CUDA_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  LOGGING_INFO(SegmentationHelp, "Classification TIME: " << elapsedTime << "ms " << endl);
}

int SegmentationHelper::getGroundSize()
{
  return gpsh.getGroundSize();
}

std::vector<Vector3f> SegmentationHelper::getGround()
{
  return gpsh.getGround();
}

int SegmentationHelper::getNonGroundSize()
{
  return gpsh.getNonGroundSize();
}

std::vector<Vector3f> SegmentationHelper::getNonGround()
{
  return gpsh.getNonGround();
}

std::vector<std::vector<Vector3f> > SegmentationHelper::getObjects()
{
  return osh.getObjects();
}

void SegmentationHelper::setWithClassification(bool val)
{
  withClassification = val;
}

void SegmentationHelper::setGroundSegmentationParameters(double segmentAngle, int numberOfBinsPerSegment, float rangeMin, float rangeMax, float thresholdLineGradient, float thresholdHorizontalLineGradient, float thresholdLowestGroundHeight, float thresholdDistancePointToLine)
{
  gpsh.setParameters(segmentAngle, numberOfBinsPerSegment, rangeMin, rangeMax, thresholdLineGradient, thresholdHorizontalLineGradient, thresholdLowestGroundHeight, thresholdDistancePointToLine);
}


void SegmentationHelper::setObjectSegmentationParameters(float relativeDistanceThreshold, float absoluteDistanceThreshold, float levelOfConcavity, float levelOfNoise, float zSimilarity, float aabbScaling, float normalMerge)
{
  osh.setParameters(relativeDistanceThreshold, absoluteDistanceThreshold, levelOfConcavity, levelOfNoise, zSimilarity, aabbScaling, normalMerge);
}

void SegmentationHelper::setClassificationClasses(std::vector<ClassificationClass> classes)
{
  dtc.setClasses(classes);
}

std::vector<Segment> SegmentationHelper::getSegments()
{
  return dtc.getClassifiedSegments();
}

int SegmentationHelper::getSegmentsSize()
{
  return dtc.getClassifiedSegmentsSize();
}

int SegmentationHelper::getNumberOfSegmentedPoints()
{
  return osh.getNumberOfSegmentedPoints();
}


}//end namespace classification
}//end namespace gpu_voxels
