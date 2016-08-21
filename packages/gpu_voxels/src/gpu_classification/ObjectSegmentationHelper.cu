#include "ObjectSegmentationHelper.h"

#include <float.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <cuda_runtime.h>

#define NUM_NEIGHBOURS 20

namespace gpu_voxels{
namespace classification{

/*!
 * used to store possibly n Neighbours. n should be the same as in Distances.
 * n should be less than 20
 */
struct Neighbours
{
  int indices[NUM_NEIGHBOURS];
  int size;

  __host__ __device__
  Neighbours()
  {
    size = NUM_NEIGHBOURS;
  }
};

/*!
 * used to store possibly n Distances. n should be the same as in Neighbours.
 * n should be less than 20
 */
struct Distances
{

  float distance[NUM_NEIGHBOURS];
  int size;

  __host__ __device__
  Distances()
  {
    size = NUM_NEIGHBOURS;
  }
};

//stores the pointinformation.
thrust::device_vector<Vector3f> pointcloudOH_d;

//stores for each point its four neighbour indices.
//thrust::device_vector<thrust::tuple<int, int, int, int> > neighbours_d;
thrust::device_vector<Neighbours> neighbours_dn;

//stores for each point the distance to the neighbour indexed at the same position as in the "neighbours_d" tuple.
//thrust::device_vector<thrust::tuple<float, float, float, float> > distances_d;
thrust::device_vector<Distances> distances_dn;

//stores the local surface normal of each point.
thrust::device_vector<Vector3f> localSurfaceNormals_d;

//stores the segmentinformation for each segment
thrust::device_vector<Segment> segments_d;

/*!
 * \param dT: distance threshold relative to neighbouring points
 * \param adT: absolute distance threshold no edges should exceed
 * \param loC: level of concavity
 * \param loN: level of noise
 * \param zS: how similar the z-coordinates are allowed to be
 */
ObjectSegmentationHelper::ObjectSegmentationHelper(float dT, float adT, float loC, float loN, float zS)
{
  relativeDistanceThreshold = dT;
  absoluteDistanceThreshold = adT;
  levelOfConcavity = loC;
  levelOfNoise = loN;
  zSimilarity = zS;
}

/*!
 * Is used in "calcLocalSurfaceNormal()" to calculate the localSurfaceNormal for each point.
 */
struct SurfaceNormalFunction
{
  Vector3f* pointCloud;
  Vector3f cameraPoint;

  /*!
     * param pc: pointer to the pointcloudOH_d array.
     * param cP: the point where the camera sits. is used to check whether the normals point to the camera or away from it.
     */
  SurfaceNormalFunction(Vector3f* pc, Vector3f cP)
  {
    pointCloud = pc;
    cameraPoint = cP;
  }

  /*!
     * \param m: tuple<index of the point which the normal is calculated for, struct of neighbours of this point>.
     * \return: local surface normal.
     */
  __host__ __device__
  Vector3f operator() (thrust::tuple<int, Neighbours > m)
  {
    //store all the information of m.
    int thisID = m.get<0>();
    Vector3f thisPoint = pointCloud[thisID];
    Neighbours n = m.get<1>();

    //store all the neighbours of the current point and check whether it exists.
    // TODO: Warning if not enough neighbours exist
    Vector3f neighbours[NUM_NEIGHBOURS];

    for(int i = 0; i < n.size; i++)
    {
      neighbours[i] = n.indices[i] != -1 ? pointCloud[n.indices[i]] : thisPoint;
    }

    Vector3f displacementVectors[NUM_NEIGHBOURS]; //displacement Vectors

    //calculate displacementVectors to the neighbours
    for(int i = 0; i < n.size; i++)
    {
      if( i == n.size - 1)
      {
        displacementVectors[i] = neighbours[0] - neighbours[i];
      }
      else
      {
        displacementVectors[i] = neighbours[i + 1] -  neighbours[i];
      }

    }

    Vector3f crossVectors[NUM_NEIGHBOURS]; //cross Vectors

    //calculate normal vectors
    for(int i = 0; i < n.size; i++)
    {
      if( i == n.size - 1)
      {
        crossVectors[i] = normalize(crossProduct(displacementVectors[0], displacementVectors[i]));
        //if vector doesn't face the camera, invert it
        crossVectors[i] = dotProduct(invertVector(thisPoint - cameraPoint), crossVectors[i]) < 0 ? crossVectors[i] : invertVector(crossVectors[i]);
      }
      else
      {
        crossVectors[i] = normalize(crossProduct(displacementVectors[i+1], displacementVectors[i]));
        //if vector doesn't face the camera, invert it
        crossVectors[i] = dotProduct(invertVector(thisPoint - cameraPoint), crossVectors[i]) < 0 ? crossVectors[i] : invertVector(crossVectors[i]);
      }

    }


    Vector3f result;

    //arithmetical mean of all the normals

    for(int i = 0; i < n.size; i++)
    {
      result.x += crossVectors[i].x;
      result.y += crossVectors[i].y;
      result.z += crossVectors[i].z;
    }
    result.x = result.x / n.size;
    result.y = result.y / n.size;
    result.z = result.z / n.size;

    result = normalize(result);
    result = dotProduct(invertVector(thisPoint - cameraPoint), result) < 0 ? result : invertVector(result);
    return normalize(result);
  }

private:

  /*!
     * calculates the cross product of two vectors a and b
     * \param a: first Vector
     * \param b: second Vector
     * \return: cross product of both the vectors
     */
  __host__ __device__
  Vector3f crossProduct(Vector3f a, Vector3f b)
  {
    return Vector3f(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  }

  /*!
     * normalizes a vector
     * \param a: vector to normalize
     * \return: normalized vector
     */
  __host__ __device__
  Vector3f normalize(Vector3f a)
  {
    float tempAbs = abs(a);
    if(tempAbs != 0)
    {
      return Vector3f(a.x/tempAbs, a.y/tempAbs, a.z/tempAbs);
    }
    else
    {
      return a;
    }
  }

  /*!
     * calculates the length of a vector
     * \param a: vector of which the length has to be calculated
     * \return: length of vector a
     */
  __host__ __device__
  float abs(Vector3f a)
  {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  }

  /*!
     * calculates the dot product of two vectors a and b
     * \param a: first vector
     * \param b: second vector
     * \return: cross product
     */
  __host__ __device__
  float dotProduct(Vector3f a, Vector3f b)
  {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  /*!
     * inverts a vector
     * \param a: vector to invert
     * \return: inverted vector
     */
  __host__ __device__
  Vector3f invertVector(Vector3f a)
  {
    return Vector3f((-1) * a.x, (-1) * a.y, (-1) * a.z);
  }
};

/*!
 * is used in "deleteLongEdges()" to delete the edges in the 4-NN-Graph that are longer than the "absoluteDistanceThreshold"
 */
struct DeleteLongEdgesFunction
{
  Vector3f* pointCloud;
  float distanceThreshold;
  float absoluteDistanceThreshold;

  /*!
     * \param pc: pointer to the pointcloudOH_d array.
     * \param dT: relative distancethreshold to the current points neighbours.
     * \param adT: absolute distance threshold.
     */
  DeleteLongEdgesFunction(Vector3f* pc, float dT, float adT)
  {
    pointCloud = pc;
    distanceThreshold = dT;
    absoluteDistanceThreshold = adT;
  }


  /*!
     * deletes all edges that are longer than the absolute distancethreshold, or that are too long compared to the distances of the other neighbours.
     * deleted neighbours get the index -1. deleted distances get the length 0.0f.
     * \param a: tuple<struct with distances , struct with neighbour indices>
     * \return: adjusted tuple<struct with distances , struct with neighbour indices>
     */
  __host__ __device__
  thrust::tuple<Distances, Neighbours > operator() (thrust::tuple<Distances, Neighbours > a)
  {
    //calculate the average distance to its neighbours
    float avgDist = 0.0f;

    Neighbours inputNeighbours = a.get<1>();
    Distances inputDistances = a.get<0>();
    for(int i = 0; i < inputNeighbours.size; i++)
    {
      avgDist += inputDistances.distance[i];
    }
    avgDist = avgDist / inputNeighbours.size;

    Distances tempDistances;
    Neighbours tempNeighbours;

    //check for absolute threshold and relative threshold
    for(int i = 0; i < inputNeighbours.size; i++)
    {
      tempDistances.distance[i] = (inputDistances.distance[i] - avgDist < distanceThreshold) && (inputDistances.distance[i] < absoluteDistanceThreshold) ? inputDistances.distance[i] : 0.0f;
      tempNeighbours.indices[i] = (inputDistances.distance[i] - avgDist < distanceThreshold) && (inputDistances.distance[i] < absoluteDistanceThreshold) ? inputNeighbours.indices[i] : -1;
    }

    return thrust::make_tuple(tempDistances, tempNeighbours);
  }
};


/*!
 * used to order the points of the pointcloud in morton order. after odering, the points form a morton path.
 */
struct MortonOrderingPredicate
{
  Vector3f* pointcloud;

  /*!
     * \param pc: pointer to pointcloud array
     */
  MortonOrderingPredicate(Vector3f* pc)
  {
    pointcloud = pc;
  }

  /*!
     * \param a: first index of a point
     * \param b: second index of a point
     * \return: whether point a is nearer at the start of the morton path.
     */
  __device__
  bool operator() (int a, int b)
  {
    float3 aVec = make_float3(pointcloud[a].x, pointcloud[a].y, pointcloud[a].z);
    float3 bVec = make_float3(pointcloud[b].x, pointcloud[b].y, pointcloud[b].z);

    int highestMSB = 0;
    bool result = false;
    //int currentMSB = XORMSB(aVec.z, bVec.z);
    //check different most significant bit for all three dimensions.
    int currentMSB = getMSDB(__float_as_int(aVec.x),__float_as_int(bVec.x));
    if(highestMSB < currentMSB)
    {
      highestMSB = currentMSB;
      result = aVec.x < bVec.x;
    }

    //currentMSB = XORMSB(aVec.y, bVec.y);
    currentMSB = getMSDB(__float_as_int(aVec.y),__float_as_int(bVec.y));
    if(highestMSB < currentMSB)
    {
      highestMSB = currentMSB;
      result = aVec.y < bVec.y;
    }

    //currentMSB = XORMSB(aVec.x, bVec.x);
    currentMSB = getMSDB(__float_as_int(aVec.z),__float_as_int(bVec.z));
    if(highestMSB < currentMSB)
    {
      highestMSB = currentMSB;
      result = aVec.z < bVec.z;
    }

    return result;
  }

private:

  /*!
     * \param a: first integer number
     * \param b: second integer number
     * \return: index of the first differing bit of integer numbers a and b.
     */
  __device__
  int getMSDB(int a, int b)
  {
    uint val = a ^ b;
    int count = 0;

    if(val == 0)
      return 0;

    while(val > 1)
    {
      val >>= 1;
      count++;
    }
    return count;
  }
};

struct FillInversMortonIndexScatterMap
{
  int* mortonPath;

  FillInversMortonIndexScatterMap(int* mP)
  {
    mortonPath = mP;
  }

  __host__ __device__
  int operator() (int a)
  {
    return mortonPath[a];
  }

};

struct KNNGraphConstruction
{
  int* mortonOrder;
  int* mortonPath;
  Neighbours* neighbours;
  Distances* distances;
  Vector3f* pointcloud;
  int pcSize;

  Neighbours resultNeighbours;
  Distances resultDistances;
  float maxDistance;

  KNNGraphConstruction(int* mO, int* mP, Neighbours* n, Distances* d, Vector3f* pc, int pcS)
  {
    mortonOrder = mO;
    mortonPath = mP;
    neighbours = n;
    distances = d;
    pointcloud = pc;
    pcSize = pcS;

    for(int i = 0; i < resultNeighbours.size; i++)
    {
      resultNeighbours.indices[i] = -1;
      resultDistances.distance[i] = FLT_MAX;
    }

    maxDistance = 0.0f;
  }

  __host__ __device__
  void operator() (int a)
  {
    for(int i = mortonOrder[a] - NUM_NEIGHBOURS; i < mortonOrder[a] + NUM_NEIGHBOURS; i++)
    {
      if(i < pcSize && i > 0)
      {
        if(mortonPath[i] != a)
        {
          insertPointInNeighbours(mortonPath[i], a);
        }
      }

    }

    thrust::tuple<int, int> bounds = findBounds(a);
    int u;
    int l;

    if(mortonOrder[bounds.get<1>()] < getPathPositionWithIndexAndOffset(a, NUM_NEIGHBOURS))
    {
      u = mortonOrder[a];
    }
    else
    {
      int I = 1;
      while(mortonOrder[bounds.get<1>()] < getPathPositionWithIndexAndOffset(a, powf(2, I)))
      {
        I++;
      }

      u = fminf(mortonOrder[a] + powf(2, I), pcSize - 1);
    }


    if(mortonOrder[bounds.get<0>()] > getPathPositionWithIndexAndOffset(a, NUM_NEIGHBOURS * (-1)))
    {
      l = mortonOrder[a];
    }
    else
    {
      int I = 1;
      while(mortonOrder[bounds.get<0>()] > getPathPositionWithIndexAndOffset(a, powf(2, I) * (-1)))
      {
        I++;
      }
      l = fmaxf(mortonOrder[a] - powf(2, I), 1);
    }

    if(l != u)
    {
      search(a, l, u, bounds);
    }

    neighbours[a] = resultNeighbours;
    distances[a] = resultDistances;
  }


private:

  __host__ __device__
  int getPathPositionWithIndexAndOffset(int a, int offset)
  {
    int tempOffset = fminf(fmaxf(offset, 0), pcSize - mortonOrder[a] - 1);
    return mortonOrder[mortonPath[mortonOrder[a] + tempOffset]];
}

__host__ __device__
thrust::tuple<int, int> findBounds(int a)
{
  int upperBoundIndex = a;
  int lowerBoundIndex = a;
  for(int i = 0; i < resultNeighbours.size; i++)
  {
    if(resultNeighbours.indices[i] != -1)
    {
      if(mortonOrder[a] < mortonOrder[resultNeighbours.indices[i]] && mortonOrder[upperBoundIndex] < mortonOrder[resultNeighbours.indices[i]])
      {
        upperBoundIndex = resultNeighbours.indices[i];
      }
      if(mortonOrder[a] > mortonOrder[resultNeighbours.indices[i]] && mortonOrder[lowerBoundIndex] > mortonOrder[resultNeighbours.indices[i]])
      {
        lowerBoundIndex = resultNeighbours.indices[i];
      }

      if(resultDistances.distance[i] > maxDistance)
      {
        maxDistance = resultDistances.distance[i];
      }
    }
  }
  return thrust::make_tuple(lowerBoundIndex, upperBoundIndex);
}

__host__ __device__
void search(int pointIndex, int l, int h, thrust::tuple<int, int> bounds)
{
  if((h-l) < 20)
  {
    for(int i = l; i < h; i++)
    {
      if(i < pcSize && i > 0)
      {
        insertPointInNeighbours(mortonPath[i], pointIndex);
      }
    }
    return;
  }

  int m = (h + l) / 2;

  insertPointInNeighbours(mortonPath[m], pointIndex);

  int tempH = h;
  tempH = tempH >= pcSize ? tempH - 1 : tempH;

  if((distancePointBox(pointcloud[pointIndex], pointcloud[mortonPath[l]], pointcloud[mortonPath[tempH]]) >= maxDistance))
  {
    return;
  }

  if(mortonOrder[pointIndex] < m)
  {
    search(pointIndex, l, m - 1, bounds);

    if(m < mortonOrder[bounds.get<1>()])
    {
      search(pointIndex, m + 1, h, bounds);
    }
  }
  else
  {
    search(pointIndex, m + 1, h, bounds);

    if(mortonOrder[bounds.get<0>()] < m)
    {
      search(pointIndex, l, m - 1, bounds);
    }
  }
}

__host__ __device__
float distancePointBox(Vector3f point, Vector3f firstCorner, Vector3f secondCorner)
{
  Vector3f min = Vector3f(fminf(firstCorner.x, secondCorner.x), fminf(firstCorner.y, secondCorner.y), fminf(firstCorner.z, secondCorner.z));
  Vector3f max = Vector3f(fmaxf(firstCorner.x, secondCorner.x), fmaxf(firstCorner.y, secondCorner.y), fmaxf(firstCorner.z, secondCorner.z));

  float dx = fmaxf(min.x - point.x, fmaxf(0, point.x - max.x));
  float dy = fmaxf(min.y - point.y, fmaxf(0, point.y - max.y));
  float dz = fmaxf(min.z - point.z, fmaxf(0, point.z - max.z));

  return sqrtf(dx * dx + dy * dy + dz * dz);
}

__host__ __device__
float distanceOfPoints(Vector3f a, Vector3f b)
{
  return (float)sqrt(powf((a.x - b.x), 2) + powf((a.y - b.y), 2) + powf((a.z - b.z), 2));
}

__host__ __device__
void insertPointInNeighbours(int a, int thisP)
{
  Vector3f aVec = pointcloud[a];
  Vector3f thisVec = pointcloud[thisP];
  Neighbours tempNeighbours;
  Distances tempDistances;
  bool changed = false;


  //check against all the other existing nearest neighbours
  for(int i = 0; i < tempNeighbours.size; i++)
  {
    float tempDist = resultDistances.distance[i];
    //if distance is small enough and it is not the current point itself
    if(distanceOfPoints(aVec, thisVec) < tempDist && distanceOfPoints(aVec, thisVec) > 0.000001f)
    {
      //copie first part of existing nearest neighbours
      for(int j = 0; j < i; j++)
      {
        tempNeighbours.indices[j] = resultNeighbours.indices[j];
        tempDistances.distance[j] = resultDistances.distance[j];
      }
      //insert new neighbour

      tempNeighbours.indices[i] = a;
      tempDistances.distance[i] = distanceOfPoints(aVec, thisVec);
      //copy rest of neighbours
      for(int j = i + 1; j < resultNeighbours.size; j++)
      {
        if(j != 0)
        {
          tempNeighbours.indices[j] = resultNeighbours.indices[j-1];
          tempDistances.distance[j] = resultDistances.distance[j-1];
        }
      }
      changed = true;
    }

    if(changed)
    {
      for(int i = 0; i < resultNeighbours.size; i++)
      {
        resultNeighbours.indices[i] = tempNeighbours.indices[i];
        resultDistances.distance[i] = tempDistances.distance[i];
      }
      return;
    }
    changed = false;
  }
}

};


/*!
 * used in "segmentPointCloud()" to merge the individual points to their segments
 */
struct SegmentFunction
{
  Vector3f* pointCloud;
  int* segmentIndices;
  int pcSize;
  Vector3f* localSurfaceNormals;
  float levelOfConcavity;
  float levelOfNoise;
  float zSimilarity;
  bool* hasChanged;

  /*!
    * \param pc: pointer to the pointcloud array
    * \param sI: segment indices of the points
    * \param pcs: pointcloudsize
    * \param lsn: pointer to the array which carries the local surface normals
    * \param loC: level of Concavity
    * \param loN: level of Noise
    * \param sZ: similarity of z-Coordinates
    * \param hC: hasChanged flag. Is needed, because this algorithm runs until no segment indices change any more.
    */
  SegmentFunction(Vector3f* pc, int* sI, int pcs, Vector3f* lsn, float loC, float loN, float sZ, bool* hC)
  {
    pointCloud = pc;
    segmentIndices = sI;
    pcSize = pcs;
    localSurfaceNormals = lsn;
    levelOfConcavity = loC;
    levelOfNoise = loN;
    zSimilarity = sZ;
    hasChanged = hC;
  }

  /*!
     * \param a: tuple<index of the current point, segment index of the current point, struct of neighbour indicess>
     */
  __host__ __device__
  void operator() (thrust::tuple<int, int, Neighbours > a)
  {
    int thisId = a.get<0>();
    //int thisSegment = a.get<1>();
    Vector3f thisPoint = pointCloud[thisId];
    Vector3f thisNormal = localSurfaceNormals[thisId];

    Neighbours neighbours = a.get<2>();

    Vector3f comparePoint;
    Vector3f comparePointNormal;

    //compare with neighbours, if the segments of the current point and the neighbour point should be merged
    for(int i = 0; i < neighbours.size; i++)
    {
      if((neighbours.indices[i]) != -1)
      {
        comparePoint = pointCloud[neighbours.indices[i]];
        comparePointNormal = localSurfaceNormals[neighbours.indices[i]];

        if(isPointsLocalConvex(thisPoint, comparePoint, thisNormal, comparePointNormal) || ((fabsf(thisNormal.z) < zSimilarity) && (fabsf(comparePointNormal.z) < zSimilarity)))
        {
          if(segmentIndices[thisId] > segmentIndices[neighbours.indices[i]])
          {
            segmentIndices[thisId] = segmentIndices[neighbours.indices[i]];
            *hasChanged = true;
          }
        }
      }
    }
  }

private:

  /*!
     * decides whether two points are convex compared to each other
     * \param a: first point
     * \param b: second point
     * \param nA: local surface normal of point a
     * \param nB: local surface normal of point b
     * \return: if the two points are local convex
     */
  __host__ __device__
  bool isPointsLocalConvex(Vector3f a, Vector3f b, Vector3f nA, Vector3f nB)
  {
    Vector3f dab = b - a;
    Vector3f dba = a - b;
    Vector3f normalA(nA.x, nA.y, nA.z);
    Vector3f normalB(nB.x, nB.y, nB.z);

    bool firstConcavArgument = dotProduct(normalA, dab) <= abs(dab) * cos(3.141592653589793f / 2.0f - levelOfConcavity);
    bool secondConcavArgument = dotProduct(normalB, dba) <= abs(dba) * cos(3.141592653589793f / 2.0f - levelOfConcavity);
    bool noiseArgument = dotProduct(normalA, normalB) >= 1 - abs(dab) * cos(3.141592653589793f / 2.0f - levelOfNoise);

    //printf("first %d[dot:%f, abs:%f, cos:%f], second %d, noise %d\n", firstConcavArgument, dotProduct(normalA, dab), abs(dab), cos(3.141592653589793f - levelOfConcavity), secondConcavArgument, noiseArgument);

    return (firstConcavArgument && secondConcavArgument) || noiseArgument;
  }

  /*!
     * calculates the dot product of two vectors a and b
     * \param a: first vector
     * \param b: second vector
     * \return: cross product
     */
  __host__ __device__
  float dotProduct(Vector3f a, Vector3f b)
  {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  /*!
     * calculates the length of a vector
     * \param a: vector of which the length has to be calculated
     * \return: length of vector a
     */
  __host__ __device__
  float abs(Vector3f a)
  {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  }
};

/*!
 * initializes the device_vectors. needs to be called before each run of the algorithm
 */
void ObjectSegmentationHelper::initialize(std::vector<Vector3f> pc)
{
  pointcloudOH_d = thrust::device_vector<Vector3f>(pc);
  //neighbours_d = thrust::device_vector<thrust::tuple<int, int, int, int> >(pointcloudOH_d.size());
  neighbours_dn = thrust::device_vector<Neighbours>(pointcloudOH_d.size());
  localSurfaceNormals_d = thrust::device_vector<Vector3f>(pointcloudOH_d.size());
  //distances_d = thrust::device_vector<thrust::tuple<float, float, float, float> >(pointcloudOH_d.size());
  distances_dn = thrust::device_vector<Distances>(pointcloudOH_d.size());
}

/*!
 * builds the k-NN-Graph with a morton predicate.
 */
void ObjectSegmentationHelper::buildNeighbourhoodGraphMortonPredicate()
{
  thrust::device_vector<int> mortonPath(pointcloudOH_d.size());
  thrust::sequence(mortonPath.begin(),
                   mortonPath.end());


  MortonOrderingPredicate mOP(thrust::raw_pointer_cast(&pointcloudOH_d[0]));

  //order the points on a morton path
  thrust::sort(mortonPath.begin(),
               mortonPath.end(),
               mOP);

  thrust::device_vector<int> mortonOrderScatterMap(pointcloudOH_d.size());
  thrust::device_vector<int> mortonOrder(pointcloudOH_d.size());


  FillInversMortonIndexScatterMap fimism(thrust::raw_pointer_cast(&mortonPath[0]));
  thrust::transform(mortonPath.begin(),
                    mortonPath.end(),
                    mortonOrderScatterMap.begin(),
                    fimism);
  thrust::scatter(mortonPath.begin(),
                  mortonPath.end(),
                  mortonOrderScatterMap.begin(),
                  mortonOrder.begin());

  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end = begin + mortonPath.size();

  KNNGraphConstruction knngc(thrust::raw_pointer_cast(&mortonOrder[0]),
      thrust::raw_pointer_cast(&mortonPath[0]),
      thrust::raw_pointer_cast(&neighbours_dn[0]),
      thrust::raw_pointer_cast(&distances_dn[0]),
      thrust::raw_pointer_cast(&pointcloudOH_d[0]),
      (int)pointcloudOH_d.size());

  thrust::for_each(begin,
                   end,
                   knngc);
}

/*!
 * deletes all long edges in the 4-NN-Graph
 */
void ObjectSegmentationHelper::deleteLongEdges()
{
  DeleteLongEdgesFunction dle(thrust::raw_pointer_cast(&pointcloudOH_d[0]), relativeDistanceThreshold, absoluteDistanceThreshold);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(distances_dn.begin(), neighbours_dn.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(distances_dn.end(), neighbours_dn.end())),
                    thrust::make_zip_iterator(thrust::make_tuple(distances_dn.begin(), neighbours_dn.begin())),
                    dle);
}

/*!
 * calculates the local surface normals for each point
 */
void ObjectSegmentationHelper::calcLocalSurfaceNormal()
{
  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end = begin + (pointcloudOH_d.size());

  SurfaceNormalFunction surfaceNormal(thrust::raw_pointer_cast(&pointcloudOH_d[0]), Vector3f(0.0f, 0.0f, -0.01f));

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(begin ,neighbours_dn.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(end ,neighbours_dn.end())),
                    localSurfaceNormals_d.begin(),
                    surfaceNormal);
}

/*!
 *
 */
void ObjectSegmentationHelper::movingAverageSurfaceNormals()
{

}

/*!
 * initializes every point to another segment. After that it merges this segments together.
 */
void ObjectSegmentationHelper::segmentPointCloud()
{
  LOGGING_DEBUG(ObjectSegmentation, "Started" << endl);
  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end = begin + (pointcloudOH_d.size());
  thrust::device_vector<int> segmentIndexList(pointcloudOH_d.size());


  //initialize segments with one point per segment
  thrust::sequence(segmentIndexList.begin(), segmentIndexList.end());


  //segment pointcloud
  thrust::device_vector<bool> hasChangedFlag(1, true);

  SegmentFunction segFunc(thrust::raw_pointer_cast(&pointcloudOH_d[0]),
      thrust::raw_pointer_cast(&segmentIndexList[0]),
      pointcloudOH_d.size(),
      thrust::raw_pointer_cast(&localSurfaceNormals_d[0]),
      levelOfConcavity,
      levelOfNoise,
      zSimilarity,
      thrust::raw_pointer_cast(&hasChangedFlag[0]));

  int count = 0;
  //run the merging till nothing changes
  while(hasChangedFlag[0])
  {
    hasChangedFlag[0] = false;
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(begin, segmentIndexList.begin(), neighbours_dn.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(end, segmentIndexList.end(), neighbours_dn.end())),
                     segFunc);
    count++;
  }

  LOGGING_DEBUG(ObjectSegmentation, "Finished Merging" << endl);

  createSegments(thrust::raw_pointer_cast(&segmentIndexList[0]), segmentIndexList.size());
  LOGGING_DEBUG(ObjectSegmentation, "Finished" << endl);
}

struct AvgVector
{
  __host__ __device__
  Vector3f operator() (thrust::tuple<int, Vector3f> a)
  {
    return normalize(Vector3f(a.get<1>().x / a.get<0>(), a.get<1>().y / a.get<0>(), a.get<1>().z / a.get<0>()));
  }

private:
  __host__ __device__
  Vector3f normalize(Vector3f a)
  {
    float tempAbs = abs(a);
    if(tempAbs != 0)
    {
      return Vector3f(a.x/tempAbs, a.y/tempAbs, a.z/tempAbs);
    }
    else
    {
      return a;
    }
  }

  __host__ __device__
  float abs(Vector3f a)
  {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  }
};

struct AABBMinMax
{
  __host__ __device__
  thrust::tuple<Vector3f, Vector3f> operator() (thrust::tuple<Vector3f, Vector3f> a, thrust::tuple<Vector3f, Vector3f> b)
  {
    Vector3f max = Vector3f(fmaxf(((Vector3f)a.get<0>()).x, ((Vector3f)b.get<0>()).x),
                            fmaxf(((Vector3f)a.get<0>()).y, ((Vector3f)b.get<0>()).y),
                            fmaxf(((Vector3f)a.get<0>()).z, ((Vector3f)b.get<0>()).z));

    Vector3f min = Vector3f(fminf(((Vector3f)a.get<1>()).x, ((Vector3f)b.get<1>()).x),
                            fminf(((Vector3f)a.get<1>()).y, ((Vector3f)b.get<1>()).y),
                            fminf(((Vector3f)a.get<1>()).z, ((Vector3f)b.get<1>()).z));

    return thrust::make_tuple(max, min);
  }
};

struct FillSegmentTransform
{
  __host__ __device__
  thrust::tuple<Vector3f, float, float, Vector3f, float> operator() (thrust::tuple<int, Vector3f, Vector3f> a)
  {
    int pointCount = a.get<0>();
    Vector3f aabbMin = a.get<1>();
    Vector3f aabbMax = a.get<2>();

    Vector3f geoCenter((aabbMin.x + aabbMax.x) / 2, (aabbMin.y + aabbMax.y) / 2, (aabbMin.z + aabbMax.z) / 2);
    Vector3f edges((aabbMax.x - aabbMin.x), (aabbMax.y - aabbMin.y), (aabbMax.z - aabbMin.z));
    float area = fabsf(edges.x * edges.y);
    float volume = fabsf(area * edges.z);
    Vector3f edgeRatios(edges.x / edges.y, edges.z / edges.x, edges.z / edges.y);
    float density = pointCount / volume;

    return thrust::make_tuple(geoCenter, area, volume, edgeRatios, density);
  }
};

struct FillSegment
{
  __host__ __device__
  Segment operator() (thrust::tuple<int , int, thrust::tuple<int, Vector3f, Vector3f, Vector3f, Vector3f, Vector3f, float, float, Vector3f, float> > a)
  {
    return Segment(a.get<0>(), a.get<1>(), a.get<2>().get<0>(), a.get<2>().get<1>(), a.get<2>().get<2>(), a.get<2>().get<3>(), a.get<2>().get<4>(), a.get<2>().get<5>(), a.get<2>().get<6>(), a.get<2>().get<7>(), a.get<2>().get<8>(), a.get<2>().get<9>());
  }
};

struct SegmentMergePredicate
{
  __host__ __device__
  bool operator() (thrust::tuple<int, Vector3f, Vector3f, Vector3f> a, thrust::tuple<int, Vector3f, Vector3f, Vector3f> b)
  {
    if(a.get<0>() < b.get<0>())
    {
      //printf("SegmentMergePredicate passed Index: %d|%d\n", a.get<0>(), b.get<0>());

      //bounding box collision check
      if(a.get<1>().x < b.get<2>().x) return false;
      if(a.get<1>().y < b.get<2>().y) return false;
      if(a.get<1>().z < b.get<2>().z) return false;

      if(a.get<2>().x > b.get<1>().x) return false;
      if(a.get<2>().y > b.get<1>().y) return false;
      if(a.get<2>().z > b.get<1>().z) return false;

      printf("SegmentMergePredicate passed AABB: %d|%d\n", a.get<0>(), b.get<0>());

      //check if Normals are similar
      float scalarProduct = a.get<3>().x * b.get<3>().x + a.get<3>().y * b.get<3>().y + a.get<3>().z * b.get<3>().z;
      if(scalarProduct > 0.8)
      {
        printf("SegmentMergePredicate TRUE: %d|%d\n", a.get<0>(), b.get<0>());
        return true;
      }
      return false;
    }
    else
    {
      return false;
    }
  }
};

struct SegmentMergeFunction
{
  int* mappings;

  SegmentMergeFunction(int* m)
  {
    mappings = m;
  }

  __host__ __device__
  thrust::tuple<int, int, int, Vector3f, Vector3f, Vector3f> operator() (thrust::tuple<int, int, int, Vector3f, Vector3f, Vector3f> a, thrust::tuple<int, int, int, Vector3f, Vector3f, Vector3f> b)
  {
    int outMapping;
    int outIndex;
    if(a.get<1>() < b.get<1>())
    {
      outMapping = a.get<0>();
      outIndex = a.get<1>();
      mappings[b.get<1>()] = a.get<1>();
    }
    else
    {
      outMapping = b.get<0>();
      outIndex = b.get<1>();
      mappings[a.get<1>()] = b.get<1>();
    }
    int outPointCount = a.get<2>() + b.get<2>();
    Vector3f outMax(fmaxf(a.get<3>().x, b.get<3>().x),
                    fmaxf(a.get<3>().y, b.get<3>().y),
                    fmaxf(a.get<3>().z, b.get<3>().z));
    Vector3f outMin(fminf(a.get<4>().x, b.get<4>().x),
                    fminf(a.get<4>().y, b.get<4>().y),
                    fminf(a.get<4>().z, b.get<4>().z));
    Vector3f outNormal((a.get<5>().x + b.get<5>().x) / 2,
                       (a.get<5>().y + b.get<5>().y) / 2,
                       (a.get<5>().z + b.get<5>().z) / 2);
    return thrust::make_tuple(outMapping, outIndex, outPointCount, outMax, outMin, outNormal);
  }
};

struct ReadjustSegmentIndices
{
  int* mappings;

  ReadjustSegmentIndices(int* m)
  {
    mappings = m;
  }

  __host__ __device__
  int operator() (int a)
  {
    if(a != -1 && mappings[a] != -1)
    {
      return mappings[a];
    }
    else
    {
      return a;
    }

  }
};

struct SegmentMerge
{
  float aabbScaling;
  float normalMerge;
  int segmentCount;
  int* segmentIndices;
  int* segmentMappings;
  int* pointCounts;
  Vector3f* aabbMax;
  Vector3f* aabbMin;
  Vector3f* normals;

  SegmentMerge(float aabbS, float nM, int sC, int* sI, int* sM, int* pC, Vector3f* max, Vector3f* min, Vector3f* n)
  {
    aabbScaling = aabbS;
    normalMerge = nM;
    segmentCount = sC;
    segmentIndices = sI;
    segmentMappings = sM;
    pointCounts = pC;
    aabbMax = max;
    aabbMin = min;
    normals = n;
  }

  __host__ __device__
  int2 operator() (int a)
  {
    int fstSeg = a % segmentCount;
    int sndSeg = a / segmentCount;


    //printf("%d < %d = %d", segmentIndices[fstSeg], segmentIndices[sndSeg], segmentIndices[fstSeg] < segmentIndices[sndSeg]);
    if(segmentIndices[fstSeg] < segmentIndices[sndSeg])
    {
      Vector3f fstMax = aabbMax[fstSeg];
      Vector3f fstMin = aabbMin[fstSeg];

      Vector3f diffFst = fstMax - fstMin;
      fstMax = fstMin + Vector3f(aabbScaling * diffFst.x, aabbScaling * diffFst.y, aabbScaling * diffFst.z);
      fstMin = fstMin + Vector3f(-1 * aabbScaling * diffFst.x, -1 * aabbScaling * diffFst.y, -1 * aabbScaling * diffFst.z);


      Vector3f sndMax = aabbMax[sndSeg];
      Vector3f sndMin = aabbMin[sndSeg];

      Vector3f diffSnd = sndMax - sndMin;
      sndMax = sndMin + Vector3f(aabbScaling * diffSnd.x, aabbScaling * diffSnd.y, aabbScaling * diffSnd.z);
      sndMin = sndMin + Vector3f(-1 * aabbScaling * diffSnd.x, -1 * aabbScaling * diffSnd.y, -1 * aabbScaling * diffSnd.z);

      //printf("1Max(%f|%f|%f) ... 1Min(%f|%f|%f) ... 2Max(%f|%f|%f) ... 2Min(%f|%f|%f)\n", fstMax.x, fstMax.y, fstMax.z , fstMin.x, fstMin.y, fstMin.z, sndMax.x, sndMax.y, sndMax.z, sndMin.x, sndMin.y, sndMin.z);


      if(fstMax.x < sndMin.x) return make_int2(-1, -1);
      if(fstMax.y < sndMin.y) return make_int2(-1, -1);
      if(fstMax.z < sndMin.z) return make_int2(-1, -1);

      if(fstMin.x > sndMax.x) return make_int2(-1, -1);
      if(fstMin.y > sndMax.y) return make_int2(-1, -1);
      if(fstMin.z > sndMax.z) return make_int2(-1, -1);

      float scalarProduct = normals[fstSeg].x * normals[sndSeg].x + normals[fstSeg].y * normals[sndSeg].y + normals[fstSeg].z * normals[sndSeg].z;
      //printf("S: %f N1(%f|%f|%f)... N2(%f|%f|%f)\n", scalarProduct, normals[fstSeg].x, normals[fstSeg].y, normals[fstSeg].z , normals[sndSeg].x, normals[sndSeg].y, normals[sndSeg].z);
      if(fabsf(scalarProduct) > normalMerge)
      {
        //printf("S: %f [%d|%d] [%d|%d]\n", scalarProduct, fstSeg, sndSeg, segmentIndices[fstSeg], segmentIndices[sndSeg]);
        //segmentMappings[segmentIndices[sndSeg]] = segmentIndices[fstSeg];
        pointCounts[fstSeg] = pointCounts[fstSeg] + pointCounts[sndSeg];
        pointCounts[sndSeg] = 0;
        aabbMax[fstSeg] = Vector3f(fmaxf(aabbMax[fstSeg].x, aabbMax[sndSeg].x), fmaxf(aabbMax[fstSeg].y, aabbMax[sndSeg].y), fmaxf(aabbMax[fstSeg].z, aabbMax[sndSeg].z));
        aabbMin[fstSeg] = Vector3f(fminf(aabbMin[fstSeg].x, aabbMin[sndSeg].x), fminf(aabbMin[fstSeg].y, aabbMin[sndSeg].y), fminf(aabbMin[fstSeg].z, aabbMin[sndSeg].z));

        return make_int2(segmentIndices[sndSeg], segmentIndices[fstSeg]);
      }
      return make_int2(-1, -1);
    }
    else
    {
      return make_int2(-1, -1);
    }
  }

private:

};

struct Int2LessThan
{

  __host__ __device__
  bool operator() (int2 a, int2 b)
  {
    return a.x < b.x;
  }
};

struct TargetMapPredicate
{
  __host__ __device__
  bool operator() (int2 a, int2 b)
  {
    return a.x == b.x;
  }
};

struct TargetMapFunction
{
  __host__ __device__
  int2 operator() (int2 a, int2 b)
  {
    return make_int2(a.x, fminf(a.y, b.y));
  }
};

struct TargetRemapping
{
  int* mapping;

  TargetRemapping(int* m)
  {
    mapping = m;
  }

  __host__ __device__
  void operator() (int2 a)
  {
    if(a.x != -1 && a.y != -1)
    {
      mapping[a.x] = a.y;
    }
  }

};

void ObjectSegmentationHelper::createSegments(int* segmentIndicesRawPointer, int size)
{
  thrust::device_ptr<int> segmentIndices = thrust::device_pointer_cast(segmentIndicesRawPointer);

  thrust::device_vector<int> pointIndices(pointcloudOH_d.size());
  thrust::sequence(pointIndices.begin(), pointIndices.end());
  LOGGING_INFO(ObjectSegmentation, "Initialized Sorting: segmentIndicesSize: " << size << " pointIndicesSize: " << pointIndices.size() << endl);
  //sort the points for segments
  thrust::sort_by_key(segmentIndices,
                      segmentIndices + size,
                      pointIndices.begin(),
                      thrust::less<int>());
  LOGGING_TRACE(ObjectSegmentation, "Sorted points for segments" << endl);

  //count number of segments
  thrust::device_vector<int>::iterator new_end;
  thrust::device_vector<int> uniqueSegments(pointcloudOH_d.size());
  new_end = thrust::unique_copy(segmentIndices,
                                segmentIndices + size,
                                uniqueSegments.begin());
  uniqueSegments.resize(thrust::distance(uniqueSegments.begin(), new_end));
  int segmentCount = thrust::distance(uniqueSegments.begin(), new_end);

  LOGGING_INFO(ObjectSegmentation, "Counted number of segments: " << segmentCount << endl);
  //    printf("Segmentcount: %d\n", segmentCount);

  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end = begin + segmentCount;

  //all properties for the segments
  thrust::device_vector<int> keys_out(segmentCount);
  thrust::device_vector<Vector3f> points(pointcloudOH_d.size());
  thrust::gather(pointIndices.begin(),
                 pointIndices.end(),
                 pointcloudOH_d.begin(),
                 points.begin());

  thrust::device_vector<int> segmentPointCount(segmentCount);
  thrust::device_vector<Vector3f> segmentNormals(segmentCount);
  thrust::device_vector<Vector3f> segmentAABBMax(segmentCount);
  thrust::device_vector<Vector3f> segmentAABBMin(segmentCount);


  thrust::equal_to<int> segmentEqual;

  //count points per segment
  thrust::device_vector<int> ones(pointcloudOH_d.size());
  thrust::plus<int> plusInt;
  thrust::fill(ones.begin(),
               ones.end(),
               1);
  thrust::reduce_by_key(segmentIndices,
                        segmentIndices + size,
                        ones.begin(),
                        keys_out.begin(),
                        segmentPointCount.begin(),
                        segmentEqual,
                        plusInt);

  LOGGING_TRACE(ObjectSegmentation, "Counted points per segment " << endl);

  //calculate Normals for each segment
  thrust::plus<Vector3f> plusVector;
  thrust::reduce_by_key(segmentIndices,
                        segmentIndices + size,
                        localSurfaceNormals_d.begin(),
                        keys_out.begin(),
                        segmentNormals.begin(),
                        segmentEqual,
                        plusVector);
  AvgVector avgV;
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.begin(), segmentNormals.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.end(), segmentNormals.end())),
                    segmentNormals.begin(),
                    avgV);
  LOGGING_TRACE(ObjectSegmentation, "Calculated normals per segment " << endl);

  //calculate AABB for each segment

  AABBMinMax aabbMinMax;
  thrust::reduce_by_key(segmentIndices,
                        segmentIndices + size,
                        thrust::make_zip_iterator(thrust::make_tuple(points.begin(), points.begin())),
                        keys_out.begin(),
                        thrust::make_zip_iterator(thrust::make_tuple(segmentAABBMax.begin(), segmentAABBMin.begin())),
                        segmentEqual,
                        aabbMinMax);

  LOGGING_TRACE(ObjectSegmentation, "Calculated AABB per segment " << endl);


  //++++++++++++++++++++ start segmentMatrix merging ++++++++++++++++++++++++++++++++++++++++
  LOGGING_TRACE(ObjectSegmentation, "start segmentMAtrix" << endl);
  thrust::device_vector<int> segmentMatrix(segmentCount * segmentCount);
  thrust::sequence(segmentMatrix.begin(), segmentMatrix.end());

  thrust::device_vector<int2> matrixMapping(segmentCount * segmentCount, make_int2(-1, -1));

  thrust::device_vector<int> segmentMapping(size, -1);

  LOGGING_DEBUG(ObjectSegmentation, "matrix Size : " << segmentMatrix.size() << " segCount: " << segmentCount << endl);


  SegmentMerge sm(aabbScaling,
                  normalMerge,
                  segmentCount,
                  thrust::raw_pointer_cast(&uniqueSegments[0]),
      thrust::raw_pointer_cast(&segmentMapping[0]),
      thrust::raw_pointer_cast(&segmentPointCount[0]),
      thrust::raw_pointer_cast(&segmentAABBMax[0]),
      thrust::raw_pointer_cast(&segmentAABBMin[0]),
      thrust::raw_pointer_cast(&segmentNormals[0]));
  thrust::transform(segmentMatrix.begin(),
                    segmentMatrix.end(),
                    matrixMapping.begin(),
                    sm);
  Int2LessThan ilt;
  thrust::sort(matrixMapping.begin(),
               matrixMapping.end(),
               ilt);

  TargetMapPredicate tmp;
  TargetMapFunction tmf;

  thrust::device_vector<int2> matrixMapping_out(segmentCount);
  thrust::device_vector<int2> targetMap(segmentCount);
  thrust::reduce_by_key(matrixMapping.begin(),
                        matrixMapping.end(),
                        matrixMapping.begin(),
                        matrixMapping_out.begin(),
                        targetMap.begin(),
                        tmp,
                        tmf);

  TargetRemapping tr(thrust::raw_pointer_cast(&segmentMapping[0]));
  thrust::for_each(targetMap.begin(),
                   targetMap.end(),
                   tr);

  LOGGING_TRACE(ObjectSegmentation, "end segmentMatrix" << endl);

  ReadjustSegmentIndices rsi(thrust::raw_pointer_cast(&segmentMapping[0]));
  thrust::transform(segmentIndices,
                    segmentIndices + size,
                    segmentIndices,
                    rsi);
  LOGGING_TRACE(ObjectSegmentation, "end readjust segmentindices" << endl);

  thrust::sort_by_key(segmentIndices,
                      segmentIndices + size,
                      pointIndices.begin(),
                      thrust::less<int>());

  thrust::gather(pointIndices.begin(),
                 pointIndices.end(),
                 pointcloudOH_d.begin(),
                 points.begin());

  //unique segments
  new_end = thrust::unique_copy(segmentIndices,
                                segmentIndices + size,
                                uniqueSegments.begin());
  uniqueSegments.resize(thrust::distance(uniqueSegments.begin(), new_end));
  segmentCount = thrust::distance(uniqueSegments.begin(), new_end);

  LOGGING_TRACE(ObjectSegmentation, "end unique " << segmentCount  << endl);


  //resize
  segmentPointCount.resize(segmentCount);
  segmentNormals.resize(segmentCount);
  segmentAABBMax.resize(segmentCount);
  segmentAABBMin.resize(segmentCount);

  //pointcount
  thrust::reduce_by_key(segmentIndices,
                        segmentIndices + size,
                        ones.begin(),
                        keys_out.begin(),
                        segmentPointCount.begin(),
                        segmentEqual,
                        plusInt);
  LOGGING_TRACE(ObjectSegmentation, "end pointcount" << endl);

  //normals
  thrust::reduce_by_key(segmentIndices,
                        segmentIndices + size,
                        localSurfaceNormals_d.begin(),
                        keys_out.begin(),
                        segmentNormals.begin(),
                        segmentEqual,
                        plusVector);
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.begin(), segmentNormals.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.end(), segmentNormals.end())),
                    segmentNormals.begin(),
                    avgV);
  LOGGING_TRACE(ObjectSegmentation, "end normals" << endl);

  //aabb
  thrust::reduce_by_key(segmentIndices,
                        segmentIndices + size,
                        thrust::make_zip_iterator(thrust::make_tuple(points.begin(), points.begin())),
                        keys_out.begin(),
                        thrust::make_zip_iterator(thrust::make_tuple(segmentAABBMax.begin(), segmentAABBMin.begin())),
                        segmentEqual,
                        aabbMinMax);
  LOGGING_TRACE(ObjectSegmentation, "end aabb" << endl);


  //++++++++++++++ end segmentMatrix merging ++++++++++++++++++++++++++++++
  //calculate barycentric center for each segment

  thrust::device_vector<Vector3f> segmentBaryCenter(segmentCount);
  thrust::reduce_by_key(segmentIndices,
                        segmentIndices + size,
                        points.begin(),
                        keys_out.begin(),
                        segmentBaryCenter.begin(),
                        segmentEqual,
                        plusVector);
  LOGGING_TRACE(ObjectSegmentation, "added up for barycenter| PointCount: " << segmentPointCount.size() << " BaryCenterSize " << segmentBaryCenter.size() << endl);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.begin(), segmentBaryCenter.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.end(), segmentBaryCenter.end())),
                    segmentBaryCenter.begin(),
                    avgV);

  LOGGING_TRACE(ObjectSegmentation, "Calculated Barycenter per segment " << endl);

  thrust::device_vector<Vector3f> segmentGeoCenter(segmentCount);
  thrust::device_vector<float> segmentXYArea(segmentCount);
  thrust::device_vector<float> segmentVolume(segmentCount);
  thrust::device_vector<Vector3f> segmentEdgeRatios(segmentCount);
  thrust::device_vector<float> segmentDensity(segmentCount);

  //calculate geometric center, xy-projected Area, Volume, Edgeratio, Density for each segment
  FillSegmentTransform fst;
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.begin(), segmentAABBMin.begin(), segmentAABBMax.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.end(), segmentAABBMin.end(), segmentAABBMax.end())),
                    thrust::make_zip_iterator(thrust::make_tuple(segmentGeoCenter.begin(), segmentXYArea.begin(), segmentVolume.begin(), segmentEdgeRatios.begin(), segmentDensity.begin())),
                    fst);

  LOGGING_TRACE(ObjectSegmentation, "Calculated all other infos per segment " << endl);

  //fill segments with properties
  segments_d = thrust::device_vector<Segment>(segmentCount);

  FillSegment fs;
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(begin, uniqueSegments.begin(), thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.begin(), segmentNormals.begin(), segmentAABBMax.begin(), segmentAABBMin.begin(), segmentBaryCenter.begin(), segmentGeoCenter.begin(), segmentXYArea.begin(), segmentVolume.begin(), segmentEdgeRatios.begin(), segmentDensity.begin())))),
                    thrust::make_zip_iterator(thrust::make_tuple(end, uniqueSegments.end(), thrust::make_zip_iterator(thrust::make_tuple(segmentPointCount.end(), segmentNormals.end(), segmentAABBMax.end(), segmentAABBMin.end(), segmentBaryCenter.end(), segmentGeoCenter.end(), segmentXYArea.end(), segmentVolume.end(), segmentEdgeRatios.end(), segmentDensity.end())))),
                    segments_d.begin(),
                    fs);

  LOGGING_TRACE(ObjectSegmentation, "filled segments" << endl);


  outputObjects = std::vector<std::vector<Vector3f> >(segmentCount);
  thrust::host_vector<int> segmentIndexList_h(segmentIndices, segmentIndices + size);
  thrust::host_vector<int> pointIndices_h(pointIndices);
  thrust::host_vector<Vector3f> pointCloud_h(points);

  if(uniqueSegments.size() > 0)
  {
    int oldSegmentIndex = uniqueSegments[0];
    uint currentSegment = 0;
    for(uint i = 0; i < pointCloud_h.size(); i++)
    {
      //int pointIndex = pointIndices_h[i];
      //Vector3f point = pointCloud_h[pointIndex];
      Vector3f point = pointCloud_h[i];

      if(segmentIndexList_h[i] == oldSegmentIndex)
      {
        if(currentSegment < outputObjects.size())
        {
          outputObjects.at(currentSegment).push_back(point);
        }
      }
      else
      {
        currentSegment++;
        oldSegmentIndex = segmentIndexList_h[i];
        if(currentSegment < outputObjects.size())
        {
          outputObjects.at(currentSegment).push_back(point);
        }
      }
    }
  }

}

/*!
 * run the hole algorithm
 */
void ObjectSegmentationHelper::segmentObjects(std::vector<Vector3f> pc)
{
  initialize(pc);
  LOGGING_TRACE(ObjectSegmentation, "Initialized with " << (int)pc.size() << " Points" << endl);

  buildNeighbourhoodGraphMortonPredicate();
  LOGGING_TRACE(ObjectSegmentation, "filled Distances and Neighbours using Morton Predicate" << endl);

  deleteLongEdges();
  LOGGING_TRACE(ObjectSegmentation, "deleted not necessary Edges" << endl);

  calcLocalSurfaceNormal();
  LOGGING_TRACE(ObjectSegmentation, "calculated local surface normal" << endl);

  //    movingAverageSurfaceNormals();

  segmentPointCloud();
  LOGGING_TRACE(ObjectSegmentation, "segmented all Objects" << endl);

}

std::vector<Segment> ObjectSegmentationHelper::getSegments()
{
  thrust::host_vector<Segment> segments_h = segments_d;
  std::vector<Segment> temp;

  for(uint i = 0; i < segments_h.size(); i++)
  {
    temp.push_back(segments_h[i]);
  }
  return temp;
}

std::vector<std::vector<Vector3f> > ObjectSegmentationHelper::getObjects()
{
  return outputObjects;
}

int ObjectSegmentationHelper::getNumberOfSegmentedPoints()
{
  return pointcloudOH_d.size();
}

void ObjectSegmentationHelper::setParameters(float relativeDistanceThreshold, float absoluteDistanceThreshold, float levelOfConcavity, float levelOfNoise, float zSimilarity, float aabbScaling, float normalMerge)
{
  this->relativeDistanceThreshold = relativeDistanceThreshold;
  this->absoluteDistanceThreshold = absoluteDistanceThreshold;
  this->levelOfConcavity = levelOfConcavity;
  this->levelOfNoise = levelOfNoise;
  this->zSimilarity = zSimilarity;
  this->aabbScaling = aabbScaling;
  this->normalMerge = normalMerge;
}

}//end of namespace classification
}//end of namespace gpu_voxels
