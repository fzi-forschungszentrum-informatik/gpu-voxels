#include "GroundPlaneSegmentationHelper.h"
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <stdio.h>


namespace gpu_voxels{
namespace classification{


//!stores the point information
thrust::device_vector<Vector3f>  pointcloud_d;

//!stores for each point its segmentnumber
thrust::device_vector<int> segmentIndices_d;

//!stores index of bin in the segment each point belongs to
thrust::device_vector<int> binIndices_d;

//!range each of the bin covers
thrust::device_vector<float2> binRanges_d;

//!stores whether a point is part of the ground or not
thrust::device_vector<bool> isGround_d;

//!stores the points projected on the line through the center of the specific segment
thrust::device_vector<float2> pointCloud2D_d;

//!stores the prototypepoints for each bin
thrust::device_vector<float2> prototypePoints_d;

//!stores the segment each prototypepoint belongs to
thrust::device_vector<int> prototypeSegments_d;

//!stores the bin each prototypepoint belongs to
thrust::device_vector<int> prototypeBins_d;

//!stores the line which represents the ground in each segment
thrust::device_vector<float2> lines_d;

//!stores all the points which are marked as ground
thrust::host_vector<Vector3f> groundPoints;

//!stores all the points which aren't marked as ground
thrust::host_vector<Vector3f> nonGroundPoints;

/*!
 * \param segmentAngle: defines the angle each segment covers
 * \param bins: defines how many bins are in each segment
 * \param rangeMin: defines the minimum distance from the center for the ground
 * \param rangeMax: defines the maximum distance from the center for the ground
 */
GroundPlaneSegmentationHelper::GroundPlaneSegmentationHelper(double segmentAngle, int bins, float rangeMin, float rangeMax, float thresholdLineGradient, float thresholdHorizontalLineGradient, float thresholdLowestGroundHeight, float thresholdDistancePointToLine)
{

  this->segmentAngle = segmentAngle;
  segmentRadian = segmentAngle * 3.141592653589793f / 180.0f;
  segmentCount = 360.0 / segmentAngle;
  binPerSegment = bins;
  this->rangeMin = rangeMin;
  this->rangeMax = rangeMax;
  this->thresholdLineGradient = thresholdLineGradient;
  this->thresholdHorizontalLineGradient = thresholdHorizontalLineGradient;
  this->thresholdLowestGroundHeight = thresholdLowestGroundHeight;
  this->thresholdDistancePointToLine = thresholdDistancePointToLine;

  binRanges_h = thrust::host_vector<float2>(bins);
  //calculate all distances to the bins
  fillBinRanges();
  binRanges_d = binRanges_h;

  prototypePoints_d = thrust::device_vector<float2>(segmentCount * binPerSegment);
  prototypeSegments_d = thrust::device_vector<int>(segmentCount * binPerSegment);
  prototypeBins_d = thrust::device_vector<int>(segmentCount * binPerSegment);
  lines_d = thrust::device_vector<float2>(segmentCount);

}

void GroundPlaneSegmentationHelper::setParameters(double segmentAngle, int numberOfBinsPerSegment, float rangeMin, float rangeMax, float thresholdLineGradient, float thresholdHorizontalLineGradient, float thresholdLowestGroundHeight, float thresholdDistancePointToLine)
{
  this->segmentAngle = segmentAngle;
  this->binPerSegment = numberOfBinsPerSegment;
  segmentRadian = segmentAngle * 3.141592653589793f / 180.0f;
  segmentCount = 360.0 / segmentAngle;
  this->rangeMin = rangeMin;
  this->rangeMax = rangeMax;
  this->thresholdLineGradient = thresholdLineGradient;
  this->thresholdHorizontalLineGradient = thresholdHorizontalLineGradient;
  this->thresholdLowestGroundHeight = thresholdLowestGroundHeight;
  this->thresholdDistancePointToLine = thresholdDistancePointToLine;

  binRanges_h = thrust::host_vector<float2>(binPerSegment);
  //calculate all distances to the bins
  fillBinRanges();
  binRanges_d = binRanges_h;

  prototypePoints_d = thrust::device_vector<float2>(segmentCount * binPerSegment);
  prototypeSegments_d = thrust::device_vector<int>(segmentCount * binPerSegment);
  prototypeBins_d = thrust::device_vector<int>(segmentCount * binPerSegment);
  lines_d = thrust::device_vector<float2>(segmentCount);
}

/*! Predicate to determine whether a Vector is in the origin of the Coordinatesystem
 *
 */
struct VectorIsNull
{
  /*!
     * \param a: Vector to check
     * \return: true if Vector lies in the origin
     */
  __host__ __device__
  bool operator() (Vector3f a)
  {
    if(a.x == 0 && a.y == 0 && a.z == 0)
    {
      return true;
    }
    return false;
  }
};


/*!
* Predicate to determine whether a point is outside the minimal or maximal range
*/
struct VectorOutOfRange
{
  float rangeMin;
  float rangeMax;

  VectorOutOfRange(float rMin, float rMax)
  {
    rangeMin = rMin;
    rangeMax = rMax;
  }

  __host__ __device__
  bool operator() (Vector3f a)
  {
    double dist = sqrt(a.x * a.x + a.y * a.y);
    if(dist < rangeMin || dist > rangeMax)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
};

/*!
* Predicate to filter all points that have one coordinate filled as NaN
*/
struct FilterNaN
{
  __host__ __device__
  bool operator() (Vector3f a)
  {
    if(isnan(a.x) || isnan(a.y) || isnan(a.z))
    {
      return true;
    }
    else
    {
      return false;
    }
  }
};

/*!
 * initializes all devices_vectors with the new pointcloud. needs to be called before each run of the algorithm.
 */
void GroundPlaneSegmentationHelper::initialize(Vector3f* pc, int size)
{
  thrust::device_ptr<Vector3f> temp_pc_device_pointer = thrust::device_pointer_cast(pc);
  pointcloud_d = thrust::device_vector<Vector3f>(temp_pc_device_pointer, temp_pc_device_pointer + size);

  //delete all Empty Vectors
  thrust::device_vector<Vector3f>::iterator new_end;

  LOGGING_DEBUG(GroundSegmentation, "PointcloudSize before filtering: " << pointcloud_d.size() << endl);

  FilterNaN fNaN;
  new_end = thrust::remove_if(pointcloud_d.begin(),
                              pointcloud_d.end(),
                              fNaN);
  pointcloud_d.resize(thrust::distance(pointcloud_d.begin(), new_end));

  LOGGING_DEBUG(GroundSegmentation, "PointcloudSize after filtering NaN: " << pointcloud_d.size() << endl);

  //delete Points in origin
  VectorIsNull isEmpty;
  new_end = thrust::remove_if(pointcloud_d.begin(),
                              pointcloud_d.end(),
                              isEmpty);
  pointcloud_d.resize(thrust::distance(pointcloud_d.begin(), new_end));

  LOGGING_DEBUG(GroundSegmentation, "PointcloudSize after filtering Empty: " << pointcloud_d.size() << endl);

  //delete all points out of the range [rangeMin, rangeMax]
  VectorOutOfRange outOfRange(rangeMin, rangeMax);
  new_end = thrust::remove_if(pointcloud_d.begin(),
                              pointcloud_d.end(),
                              outOfRange);

  pointcloud_d.resize(thrust::distance(pointcloud_d.begin(), new_end));

  LOGGING_DEBUG(GroundSegmentation, "PointcloudSize after filtering out of range: " << pointcloud_d.size() << endl);

  segmentIndices_d = thrust::device_vector<int>(pointcloud_d.size());
  binIndices_d = thrust::device_vector<int>(pointcloud_d.size());

  isGround_d = thrust::device_vector<bool>(pointcloud_d.size());

  pointCloud2D_d = thrust::device_vector<float2>(pointcloud_d.size());
}

/*!
 * used in "calcSegments()" to assign each point a segment
 */
struct segmentCalculation
{
  double segmentRadian;
  double segmentCount;

  /*!
     * \param sr: the angle of each segment in radian
     * \param sc: the number of total segments
     */
  segmentCalculation(double sr, double sc)
  {
    segmentRadian = sr;
    segmentCount = sc;
  }

  /*!
     * \param a: point to calculate the segment for
     * \return: segment point belongs to
     */
  __host__ __device__
  int operator() (Vector3f a)
  {
    double temp = atan2(a.y, a.x) / segmentRadian;
    int t = ((int)round(temp + segmentCount/2));
    if( t == segmentCount)
    {
      return 0;
    }
    else
    {
      return t;
    }
  }
};

/*!
 * used in "calcBins()" to calculate the bin of each point in its segment
 */
struct binCalculation
{
  float2* binRanges;
  uint binRanges_size;

  /*!
     * \param br: pointer to the ranges each bin should cover
     * \param size: number of bins
     */
  binCalculation(float2 *br, uint size)
  {
    binRanges = br;
    binRanges_size = size;
  }

  /*!
     * \param a: point to calculate the bin for
     * \return: bin point belongs to
     */
  __host__ __device__
  int operator() (Vector3f a)
  {
    double dist = sqrt(a.x * a.x + a.y * a.y);
    return getIndexOfBin(dist);
  }

private:
  /*!
     * \param range: distance from the center
     * \return: the index of the corresponding bin for the given distance
     */
  __host__ __device__
  int getIndexOfBin(double range)
  {
    for(uint i = 0; i < binRanges_size; i++)
    {
      float2 temp = binRanges[i];
      if(range > temp.x && range < temp.y)
      {
        return i;
      }
    }
    return -1;
  }
};

/*!
 * used in "convertTo2D()" to project each point to a plane which divides the segment of the point in half
 */
struct convertPointTo2D
{
  /*!
     * \param a: 3D-Point to be projected
     * \return: converted 2D-Point
     */
  __host__ __device__
  float2 operator() (Vector3f a)
  {
    return make_float2((float)sqrt(a.x * a.x + a.y * a.y), a.z);
  }
};

/*!
 * used in "createPrototypePoints()" to decide whether two prototypepoints are equal
 */
struct prototypePointsPredicate
{
  __host__ __device__
  bool operator() (thrust::tuple<int, int> a, thrust::tuple<int, int> b)
  {
    return (a.get<0>() == b.get<0>()) && (a.get<1>() == b.get<1>());
  }
};


/*!
 * used in "createPrototypePoints()" to sort the prototypepoints with ascending z-coordinate
 */
struct prototypePointsZMinCompare
{
  __host__ __device__
  float2 operator() (float2 a, float2 b)
  {
    if(a.y < b.y){
      return a;
    }
    else {
      return b;
    }
  }
};

/*!
 * used in "createLinePerSegment()" to rearange to prototypepoints in a one dimensional array
 */
struct createPrototypeScatterMap
{
  int binCount;

  /*!
     * \param bc: bins per segment
     */
  createPrototypeScatterMap(int bc)
  {
    binCount = bc;
  }

  /*!
     * \param a: segment and bin information
     * \return: place in the sorted vector to go
     */
  __host__ __device__
  int operator() (thrust::tuple<int, int> a)
  {
    return a.get<0>() * binCount + a.get<1>();
  }
};

/*!
 * used in "createLinePerSegment()" to create one line for each segment with the prototypepoints of the corresponding segment
 */
struct linearRegression
{
  float2* points;
  float2* lines;
  float2* relevantPoints;
  int binCount;
  float thresholdM;
  float thresholdMSmall;
  float thresholdB;

  /*!
     * \param p: pointer to 2D-points array
     * \param l: pointer to lines-array
     * \param bs: bins per segment
     * \param tM: threshold for the m in lines
     * \param tMS: lower threshold for the m in lines
     * \param tB: threshold for the b in lines
     */
  linearRegression(float2* p, float2* l, float2* rP, int bc, float tM, float tMS, float tB)
  {
    points = p;
    lines = l;
    relevantPoints = rP;
    binCount = bc;
    thresholdM = tM;
    thresholdMSmall = tMS;
    thresholdB = tB;

  }

  /*!
     * \param a: index of the segment to calculate the line for
     */
  __host__ __device__
  void operator() (int a)
  {
    int start = a * binCount;
    float2 avg;
    float2 mq;
    avg = make_float2(0, 0);

    //used to represent a line
    float m;
    float b;

    //iterative linear regression
    int relevantPointCounter = 0;
    for(int j = 0; j < binCount; j++)
    {
      float2 nextPoint = points[start + j];
      relevantPoints[relevantPointCounter] = nextPoint;

      //at least 2 poinst needed for a line
      if(relevantPointCounter < 2)
      {
        relevantPointCounter++;
        continue;
      }
      //calculate the average of all points in both dimensions
      int avgDivider = 0;
      for(int i = 0; i < relevantPointCounter + 1; i++)
      {
        float2 p = relevantPoints[i];
        if(p.x < 0.001f && p.y < 0.001f)
        {
          continue;
        }
        else
        {
          avg.x += p.x;
          avg.y += p.y;
          avgDivider++;
        }
      }

      avg.x = avg.x / avgDivider;
      avg.y = avg.y / avgDivider;

      mq = make_float2(0.0f, 0.0f);
      //iterativly add one point after another to the line and check
      //whether the new line satisfies the thresholds. if not, keep the
      //old line
      for(int i = 0; i < relevantPointCounter + 1; i++)
      {
        float2 p = relevantPoints[i];
        if(p.x < 0.001f && p.y < 0.001f)
        {
          continue;
        }
        float dx = p.x - avg.x;
        float dy = p.y - avg.y;

        mq.x += dx * dy;
        mq.y += dx * dx;
      }

      m = mq.x / mq.y;
      b = avg.y - (m * avg.x);

      if(m < thresholdM)
      {
        if(m > thresholdMSmall || (m < thresholdMSmall && b < thresholdB))
        {
          relevantPointCounter++;
        }
      }
    }

    // add line as result
    lines[a] = make_float2(m, b);
  }

};

/*!
 * used to take the first element out of a tuple. for ouput purposes
 */
struct takeFirstFromTuple
{
  __host__ __device__
  thrust::tuple<int, int> operator() (thrust::tuple<int, int> a)
  {
    return a;
  }
};

/*!
 * decide whether a integer tuple is smaller than another
 * true if tuple.first < tuple.second
 */
struct tupleIsSmallerThen
{
  __host__ __device__
  bool operator() (thrust::tuple<int, int> a, thrust::tuple<int, int> b)
  {
    if(a.get<0>() < b.get<0>())
    {
      return a.get<0>() < b.get<0>();
    }
    else if(a.get<0>() == b.get<0>())
    {
      return a.get<1>() < b.get<1>();
    }
    else
    {
      return false;
    }
  }
};

/*!
 *
 */
struct intIsSmallerThen
{
  __host__ __device__
  bool operator() (int a, int b)
  {
    return a < b;
  }
};

/*!
 * used in "labelPointsAsGround()" to determine for each point if it belongs to the ground
 */
struct isGroundOperator
{
  float2 *lines;
  float thresholdDistance;

  /*!
     * \param l: pointer to lines-array
     * \param tD: maximum distance to line threshold
     */
  isGroundOperator(float2* l, float tD)
  {
    lines = l;
    thresholdDistance = tD;
  }

  /*!
     * \param a: tuple <point to check, index of line to check with>
     */
  __host__ __device__
  bool operator() (thrust::tuple<Vector3f, int> a)
  {
    if(distancePointToSegmentLine(a.get<0>(), lines[a.get<1>()]) < thresholdDistance)
    {
      return true;
    }
    return false;
  }

private:
  __host__ __device__
  float distancePointToSegmentLine(Vector3f p, float2 l)
  {
    float distanceFromCenter = sqrt(p.x * p.x + p.y * p.y);
    float y = l.x * distanceFromCenter + l.y;

    return abs(y - p.z);

  }
};

/*!
 * used to partition the output vector
 */
struct GroundPointsPartitionPredicate
{
  __host__ __device__
  bool operator() (thrust::tuple<Vector3f, bool> a)
  {
    return a.get<1>();
  }
};

struct TupleConversion
{
  __host__ __device__
  Vector3f operator() (thrust::tuple<Vector3f, bool> a)
  {
    return a.get<0>();
  }
};


/*! calculates for each point in which segment it belongs.
 */
void GroundPlaneSegmentationHelper::calcSegments()
{
  segmentCalculation func(segmentRadian, segmentCount);
  thrust::transform(pointcloud_d.begin(), pointcloud_d.end(), segmentIndices_d.begin(), func);
}

/*! calculates for each point in which bin it belongs in its segment.
 */
void GroundPlaneSegmentationHelper::calcBins()
{
  binCalculation func(thrust::raw_pointer_cast(&binRanges_d[0]), binRanges_d.size());
  thrust::transform(pointcloud_d.begin(), pointcloud_d.end(), binIndices_d.begin(), func);

  int count = thrust::count(binIndices_d.begin(),
                            binIndices_d.end(),
                            -1);
}

/*! project each point to a plane which divides the segment of the point in half
 */
void GroundPlaneSegmentationHelper::convertTo2D()
{
  convertPointTo2D func;
  thrust::transform(pointcloud_d.begin(), pointcloud_d.end(), pointCloud2D_d.begin(), func);
}

/*! extracts representative points for each bin
 */
void GroundPlaneSegmentationHelper::createPrototypePoints()
{
  tupleIsSmallerThen tupleSmaller;

  //collect segmentindices and binindices in one tuple
  thrust::device_vector<thrust::tuple<int, int> > keys(segmentIndices_d.size());
  thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(segmentIndices_d.begin(), binIndices_d.begin())),
               thrust::make_zip_iterator(thrust::make_tuple(segmentIndices_d.end(), binIndices_d.end())),
               keys.begin());

  //sort keys by segment and bins
  thrust::sort_by_key(keys.begin(), keys.end(), pointCloud2D_d.begin(), tupleSmaller);


  prototypePointsPredicate pred;
  prototypePointsZMinCompare comp;

  //find for each bin one representative point
  thrust::pair<thrust::device_vector<thrust::tuple<int, int> >::iterator, thrust::device_vector<float2>::iterator> new_end;
  thrust::device_vector<thrust::tuple<int, int> > keys_out(segmentCount * binPerSegment);
  new_end = thrust::reduce_by_key(keys.begin(),
                                  keys.end(),
                                  pointCloud2D_d.begin(),
                                  keys_out.begin(),
                                  prototypePoints_d.begin(),
                                  pred,
                                  comp);

  takeFirstFromTuple firstFunc;
  thrust::transform(keys_out.begin(), keys_out.end(), thrust::make_zip_iterator(thrust::make_tuple(prototypeSegments_d.begin(), prototypeBins_d.begin())), firstFunc);
}

/*! estimates a line through the prototypepoints
 */
void GroundPlaneSegmentationHelper::createLinePerSegment()
{
  intIsSmallerThen greaterInt;

  //find the segment to each prototypePoint
  thrust::sort_by_key(prototypeSegments_d.begin(),
                      prototypeSegments_d.end(),
                      thrust::make_zip_iterator(thrust::make_tuple(prototypePoints_d.begin(), prototypeBins_d.begin())),
                      greaterInt);

  thrust::device_vector<float2> prototypePointsInOrder(prototypePoints_d.size());
  thrust::device_vector<int> map(prototypePoints_d.size());

  createPrototypeScatterMap prototypePointsScatterMap(binPerSegment);

  //create prototypePointScatterMap
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(prototypeSegments_d.begin(), prototypeBins_d.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(prototypeSegments_d.end(), prototypeBins_d.end())),
                    map.begin(),
                    prototypePointsScatterMap);

  thrust::scatter(prototypePoints_d.begin(), prototypePoints_d.end(), map.begin(), prototypePointsInOrder.begin());

  //calculate the lines per segment
  thrust::device_vector<float2> relevantPoints(binPerSegment);
  thrust::counting_iterator<int> segments(0);
  linearRegression lr(thrust::raw_pointer_cast(&prototypePointsInOrder[0]),
      thrust::raw_pointer_cast(&lines_d[0]),
      thrust::raw_pointer_cast(&relevantPoints[0]),
      binPerSegment, thresholdLineGradient, thresholdHorizontalLineGradient, thresholdLowestGroundHeight);

  thrust::for_each_n(segments, segmentCount, lr);

}

/*! marks whether a point belongs to the ground
 */
void GroundPlaneSegmentationHelper::labelPointsAsGround()
{
  isGroundOperator op(thrust::raw_pointer_cast(&lines_d[0]), thresholdDistancePointToLine);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(pointcloud_d.begin(), segmentIndices_d.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(pointcloud_d.end(), segmentIndices_d.end())),
                    isGround_d.begin(),
                    op);
}

/*! calculates in which distance range each bin covers
 */
void GroundPlaneSegmentationHelper::fillBinRanges()
{
  float binDist = (rangeMax - rangeMin) / binPerSegment;
  LOGGING_DEBUG(GroundSegmentation, "Max: " << rangeMax << " Min: " << rangeMin << " binsPerSegment: " << binPerSegment << " binWidht: " << binDist << endl);
  for(uint i = 0; i < binRanges_h.size(); i++)
  {
    binRanges_h[i] = make_float2(binDist * i + rangeMin, binDist * (i+1) + rangeMin);
  }
}

void GroundPlaneSegmentationHelper::segmentGround(std::vector<Vector3f> ibeo, Vector3f *velodyne, int size)
{
  thrust::device_vector<Vector3f> ibeo_d = thrust::device_vector<Vector3f>(ibeo);
  thrust::device_ptr<Vector3f> velo_device_pointer = thrust::device_pointer_cast(velodyne);

  thrust::device_vector<Vector3f> temp = thrust::device_vector<Vector3f>(ibeo_d.size() + size);
  thrust::copy(ibeo_d.begin(), ibeo_d.end(), temp.begin());
  thrust::copy(velo_device_pointer, velo_device_pointer + size, temp.begin() + ibeo_d.size());

  segmentGround(thrust::raw_pointer_cast(&temp[0]), temp.size());
}

void GroundPlaneSegmentationHelper::segmentGround(Vector3f* pc, int size)
{
  //thrust::device_ptr<Vector3f> pc_device_pointer = thrust::device_pointer_cast(pc);

  initialize(pc, size);

  LOGGING_DEBUG(GroundSegmentation, "Initialized with " << size << " Points. Remaining Points after Cleanup: " << pointcloud_d.size() << endl);


  calcSegments();
  LOGGING_TRACE(GroundSegmentation, "calculated Segments" << endl);

  calcBins();
  LOGGING_TRACE(GroundSegmentation, "calculated Bins" << endl);

  convertTo2D();
  LOGGING_TRACE(GroundSegmentation, "converted to 2D" << endl);

  createPrototypePoints();
  LOGGING_TRACE(GroundSegmentation, "created PrototypePoints" << endl);

  createLinePerSegment();
  LOGGING_TRACE(GroundSegmentation, "created Lines" << endl);

  labelPointsAsGround();
  LOGGING_TRACE(GroundSegmentation, "labeled as ground" << endl);


  //create two poinclouds for output: ground, nonGround
  GroundPointsPartitionPredicate groundPointsPartition;
  thrust::device_vector<thrust::tuple<Vector3f, bool> > tempGround(pointcloud_d.size());
  thrust::device_vector<thrust::tuple<Vector3f, bool> > tempNonGround(pointcloud_d.size());

  thrust::pair<thrust::device_vector<thrust::tuple<Vector3f, bool > >::iterator, thrust::device_vector<thrust::tuple<Vector3f, bool > >::iterator> new_end;

  new_end = thrust::partition_copy(thrust::make_zip_iterator(thrust::make_tuple(pointcloud_d.begin(), isGround_d.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(pointcloud_d.end(), isGround_d.end())),
                                   tempGround.begin(),
                                   tempNonGround.begin(),
                                   groundPointsPartition);

  tempGround.resize(thrust::distance(tempGround.begin(), new_end.first));
  tempNonGround.resize(thrust::distance(tempNonGround.begin(), new_end.second));

  thrust::device_vector<Vector3f> groundPoints_d(tempGround.size());
  thrust::device_vector<Vector3f> nonGroundPoints_d(tempNonGround.size());

  TupleConversion tupleConversion;

  LOGGING_DEBUG(GroundSegmentation, "tempGround Size: " << tempGround.size() << " tempNonGround Size: " << tempNonGround.size() << " groundPoints_d Size: " << groundPoints_d.size() << " nonGroundPoints_d Size: " << nonGroundPoints_d.size() << endl);
  thrust::transform(tempGround.begin(),
                    tempGround.end(),
                    groundPoints_d.begin(),
                    tupleConversion);
  thrust::transform(tempNonGround.begin(),
                    tempNonGround.end(),
                    nonGroundPoints_d.begin(),
                    tupleConversion);


  groundPoints = groundPoints_d;
  nonGroundPoints = nonGroundPoints_d;
}


void GroundPlaneSegmentationHelper::segmentGround(std::vector<Vector3f> pc)
{
  thrust::device_vector<Vector3f> temp = thrust::device_vector<Vector3f>(pc);
  segmentGround(thrust::raw_pointer_cast(&temp[0]), temp.size());

}

int GroundPlaneSegmentationHelper::getGroundSize()
{
  return groundPoints.size();
}

int GroundPlaneSegmentationHelper::getNonGroundSize()
{
  return nonGroundPoints.size();
}

std::vector<Vector3f> GroundPlaneSegmentationHelper::getGround()
{
  std::vector<Vector3f> temp(groundPoints.size());

  for(uint i = 0; i < groundPoints.size(); i++)
  {
    temp[i] = groundPoints[i];
  }

  return temp;
}

std::vector<Vector3f> GroundPlaneSegmentationHelper::getNonGround()
{
  std::vector<Vector3f> temp(nonGroundPoints.size());

  for(uint i = 0; i < nonGroundPoints.size(); i++)
  {
    temp[i] = nonGroundPoints[i];
  }

  return temp;
}


}//end namespace classification
}//end namespace gpu_voxels
