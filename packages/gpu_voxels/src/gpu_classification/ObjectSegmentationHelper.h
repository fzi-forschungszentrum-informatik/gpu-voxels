#ifndef OBJECTSEGMENTATIONHELPER_H
#define OBJECTSEGMENTATIONHELPER_H

#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/GpuVoxels.h>
#include <gpu_classification/logging/logging_classification.h>
#include <thrust/host_vector.h>

namespace gpu_voxels{
namespace classification{

struct Segment
{
  //unique id for each segment
  int id;
  //
  int segmentIndex;
  //how many points are in this segment
  int pointCount;
  //estimation of a average surface normal
  Vector3f normal;
  //upper corner of the AABB
  Vector3f aabbMax;
  //lower corner of the AABB
  Vector3f aabbMin;
  //arithmetical mean of all positions
  Vector3f baryCenter;
  //center of the AABB
  Vector3f geoCenter;
  //footprint of segment on the xy-Plane
  float xyArea;
  //
  float volume;
  //ratios between two edges Vector3f(x/z, y/z, x/y)
  Vector3f edgeRatios;
  //points per volume
  float density;

  //used for the visualizer
  gpu_voxels::BitVoxelMeaning meaning;

  __host__ __device__
  Segment()
  {
    id = 0;
    segmentIndex = 0;
    pointCount = 0;
    normal = Vector3f(0.0f, 0.0f, 0.0f);
    aabbMax = Vector3f(0.0f, 0.0f, 0.0f);
    aabbMin = Vector3f(0.0f, 0.0f, 0.0f);
    baryCenter = Vector3f(0.0f, 0.0f, 0.0f);
    geoCenter = Vector3f(0.0f, 0.0f, 0.0f);
    xyArea = 0.0f;
    volume = 0.0f;
    edgeRatios = Vector3f(0.0f, 0.0f, 0.0f);
    density = 0.0f;
    meaning = static_cast<gpu_voxels::BitVoxelMeaning>(11);
  }

  __host__ __device__
  Segment(int id, int segInd, int pC, Vector3f n, Vector3f boxMax, Vector3f boxMin, Vector3f bC, Vector3f gC, float area, float v, Vector3f eR, float d)
  {
    this->id = id;
    segmentIndex = segInd;
    pointCount = pC;
    normal = n;
    aabbMax = boxMax;
    aabbMin = boxMin;
    baryCenter = bC;
    geoCenter = gC;
    xyArea = area;
    volume = v;
    edgeRatios = eR;
    density = d;
    meaning = static_cast<gpu_voxels::BitVoxelMeaning>(11);
  }

  __host__ __device__
  void setMeaning(int meaning)
  {
    this->meaning = static_cast<gpu_voxels::BitVoxelMeaning>(meaning);
  }

  __host__ __device__
  void setMeaning(gpu_voxels::BitVoxelMeaning meaning)
  {
    this->meaning = meaning;
  }
};


class ObjectSegmentationHelper
{
public:
  //Constructor
  ObjectSegmentationHelper(float dT, float adT, float loC, float loN, float zS);

  /*!
     * \brief segmentObjects
     * \param pc: pointcloud to divide into segments
     */
  void segmentObjects(std::vector<Vector3f> pc);

  /*!
     * \brief getObjects
     * \return pointcloud of the segments. indices are related to the return of getSegments
     */
  std::vector<std::vector<Vector3f> > getObjects();

  /*!
     * \brief getSegments
     * \return just the segmentinformation, not containing the actual pointclouds
     */
  std::vector<Segment> getSegments();

  /*!
     * \brief setParameters
     * \param relativeDistanceThreshold
     * \param absoluteDistanceThreshold
     * \param levelOfConcavity
     * \param levelOfNoise
     * \param zSimilarity
     * \param aabbScaling
     * \param normalMerge
     */
  void setParameters(float relativeDistanceThreshold, float absoluteDistanceThreshold, float levelOfConcavity, float levelOfNoise, float zSimilarity, float aabbScaling, float normalMerge);

  /*!
     * \brief getNumberOfSegmentedPoints
     * \return returns number of points in all segments
     */
  int getNumberOfSegmentedPoints();

  //    struct Neighbours;
  //    struct Distances;
  //    struct Segment;

private:

  //! defines the relative distancethreshold one point can have relative to its neighbours.
  float relativeDistanceThreshold;

  //! defines the absolute distancethreshold. If the distance between two points is greater than this threshold, they musn't be neighbours.
  float absoluteDistanceThreshold;

  //! defines the level of concavity. This is the biggest angle between two neighbouring surface normals to be counted as concave, in radians.
  float levelOfConcavity;

  //! defines the level of noise. This needed to compare if two surface normals are equaly enough to count them as concave.
  float levelOfNoise;

  //! defines how similar two z-Coordinates have to be to be treated as equal.
  float zSimilarity;

  float aabbScaling;
  float normalMerge;

  //! Vector of object pointclouds to save the output.
  std::vector<std::vector<Vector3f> > outputObjects;

  /*!
     * \param pc: input pointcloud to get segmented.
     */
  void initialize(std::vector<Vector3f> pc);

  /*!
     * builds the 4-NN-Graph with brute force distance matrix. (Not recommended because of huge memory usage)
     */
  void buildNeighbourhoodGraphDistanceMatrix();

  /*!
     * builds the 4-NN-Graph with a minimum distance search for each point. (kernel launch takes too much time)
     */
  void buildNeighbourhoodGraphMinimumSearch();

  /*!
     * build the 4-NN-Graph with a morton sorting predicate.
     */
  void buildNeighbourhoodGraphMortonPredicate();

  void buildNeighbourhoodGraphMortonPredicateTest();

  void sortMortonPath(int* rawPointer, float* x, float* y, float* z, int size);
  void invertMortonPath(int* rawPointerPath, int* rawPointerScatterMap, int* rawPointerOrder, int size);

  /*!
     * postprocessing of the 4-NN-Graph. delete all edges in the graph, that are longer than "absoluteDistanceThreshold".
     */
  void deleteLongEdges();

  /*!
     * calculate for each point its local surface normal.
     */
  void calcLocalSurfaceNormal();

  /*!
     * not implemented yet.
     */
  void movingAverageSurfaceNormals();

  /*!
     * assign a segment to each point.
     */
  void segmentPointCloud();

  void createSegments(int* segmentIndicesRawPointer, int size);
};

}//end of namespace classification
}//end of namespace gpu_voxels

#endif // OBJECTSEGMENTATIONHELPER_H
