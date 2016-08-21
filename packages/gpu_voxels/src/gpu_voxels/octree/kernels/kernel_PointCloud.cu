// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2014-11-15
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/octree/kernels/kernel_PointCloud.h>

namespace gpu_voxels {
namespace NTree {

__global__
void kernel_transformKinectPoints(gpu_voxels::Vector3f* point_cloud, OctreeVoxelID num_points, Voxel* voxel,
                                  Sensor* sensor, gpu_voxels::Vector3f voxel_dimension)
{
  const OctreeVoxelID id = blockIdx.x * blockDim.x + threadIdx.x;
  const OctreeVoxelID numThreads = gridDim.x * blockDim.x;

  // TODO: filter points
  for (OctreeVoxelID i = id; i < num_points; i += numThreads)
  {
//    // handle NAN values
//    gpu_voxels::Vector3f point = point_cloud[i];

    const gpu_voxels::Vector3f point = sensor->sensorCoordinatesToWorldCoordinates(point_cloud[i]) * 1000.0f; // from meter to mm
    Voxel* v = &voxel[i];
    v->coordinates.x = uint32_t(point.x / voxel_dimension.x);
    v->coordinates.y = uint32_t(point.y / voxel_dimension.y);
    v->coordinates.z = uint32_t(point.z / voxel_dimension.z);
    v->voxelId = morton_code60(v->coordinates.x, v->coordinates.y, v->coordinates.z);
    v->setOccupancy(sensor->sensorModel.applySensorModel(point_cloud[i]));
  }
}

__global__
void kernel_transformKinectPoints_simple(gpu_voxels::Vector3f* point_cloud, const voxel_count num_points,
                                         OctreeVoxelID* voxel, Sensor* sensor, const uint32_t resolution)
{
  for (voxel_count i = blockIdx.x * blockDim.x + threadIdx.x; i < num_points; i += gridDim.x * blockDim.x)
  {
    const gpu_voxels::Vector3f point = sensor->sensorCoordinatesToWorldCoordinates(point_cloud[i]) * 1000.0f; // from meter to mm

    // only positive coordinates are allowed due to morton code
    assert(point.x >= 0.0f && point.y >= 0.0f && point.z >= 0.0f);

    // transform to voxel
    gpu_voxels::Vector3ui c;
    c.x = uint32_t(point.x / resolution);
    c.y = uint32_t(point.y / resolution);
    c.z = uint32_t(point.z / resolution);
    voxel[i] = morton_code60(c);
  }
}

__global__
void kernel_voxelize_finalStep(OctreeVoxelID* voxelInput, const voxel_count numVoxel, const voxel_count num_output_voxel,
                               Voxel* voxel_output, Sensor* sensor)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_output_voxel; i += gridDim.x * blockDim.x)
  {
    Voxel* v = &voxel_output[i];

    // fix end index of last voxel
    if (i == num_output_voxel - 1)
      v->coordinates.y = numVoxel - 1;

    const uint32_t num_point_in_voxel = v->coordinates.y - v->coordinates.x + 1;
    const Probability occupancy = sensor->sensorModel.estimateVoxelProbability(num_point_in_voxel);
    v->setOccupancy(occupancy);
    inv_morton_code60(v->voxelId, v->coordinates);
  }
}

__global__
void kernel_toMortonCode(ulong3* inputVoxel, voxel_count numVoxel, OctreeVoxelID* outputVoxel)
{
  const voxel_count id = blockIdx.x * blockDim.x + threadIdx.x;
  const voxel_count numThreads = gridDim.x * blockDim.x;

  for (voxel_count i = id; i < numVoxel; i += numThreads)
    outputVoxel[i] = morton_code60(inputVoxel[i].x, inputVoxel[i].y, inputVoxel[i].z);
}

__global__ void kernel_countVoxel(Voxel* voxelInput, OctreeVoxelID numVoxel, OctreeVoxelID* countVoxel)
{
  const OctreeVoxelID chunkSize = floor(double(numVoxel) / (gridDim.x * blockDim.x));
  const OctreeVoxelID overhead = numVoxel - chunkSize * gridDim.x * blockDim.x;
  const OctreeVoxelID id = blockIdx.x * blockDim.x + threadIdx.x;
  const OctreeVoxelID from = id * chunkSize + min((unsigned long long) overhead, (unsigned long long) id);
  const OctreeVoxelID to = min((unsigned long long) (from + chunkSize + (id < overhead) ? 1 : 0),
                         (unsigned long long) numVoxel);

  OctreeVoxelID myVoxelCount = 0;
  if (from < numVoxel)
  {
    OctreeVoxelID lastVoxel = voxelInput[from].voxelId;
    myVoxelCount = ((from == 0) || voxelInput[from - 1].voxelId != lastVoxel) ? 1 : 0;
    for (OctreeVoxelID i = from + 1; i < to; ++i)
    {
      myVoxelCount += voxelInput[i].voxelId == lastVoxel ? 0 : 1;
      lastVoxel = voxelInput[i].voxelId;
    }
  }
  countVoxel[id] = myVoxelCount;
}

__global__ void kernel_combineEqualVoxel(Voxel* voxelInput, OctreeVoxelID numVoxel, OctreeVoxelID* countVoxel,
                                         Voxel* outputVoxel, Sensor* sensor)
{
  const OctreeVoxelID chunkSize = floor(double(numVoxel) / (gridDim.x * blockDim.x));
  const OctreeVoxelID overhead = numVoxel - chunkSize * gridDim.x * blockDim.x;
  const OctreeVoxelID id = blockIdx.x * blockDim.x + threadIdx.x;
  const OctreeVoxelID from = id * chunkSize + min((unsigned long long) overhead, (unsigned long long) id);
  const OctreeVoxelID to = min((unsigned long long) (from + chunkSize + (id < overhead) ? 1 : 0),
                         (unsigned long long) numVoxel);

  if (from < numVoxel)
  {
    OctreeVoxelID lastVoxel = voxelInput[from].voxelId;
    OctreeVoxelID destPos = (id == 0) ? 0 : countVoxel[id - 1] - 1;
    bool leftMost = ((from == 0) || voxelInput[from - 1].voxelId != lastVoxel);
    destPos += leftMost ? 0 : 1;
    OctreeVoxelID firstVoxel = from;

//    // scan to next voxel
//    voxel_id i = from + 1;
//    for (; (i < to || leftMost) && i < numVoxel && voxelInput[i - 1].voxel_id == voxelInput[i].voxel_id; ++i)
//      ;
//
//    if (leftMost)
//    {
//      outputVoxel[destPos] = sensorModel.estimateVoxelProbability(&voxelInput[firstVoxel],
//                                                                  &voxelInput[i - 1]);
//    }

    for (OctreeVoxelID i = from + 1; ((i < to) | leftMost) & (i < numVoxel); ++i)
    {
      if (voxelInput[i].voxelId != lastVoxel)
      {
        if (leftMost)
          outputVoxel[destPos].setOccupancy(
              sensor->sensorModel.estimateVoxelProbability(&voxelInput[firstVoxel], &voxelInput[i - 1]));
        leftMost = true;
        firstVoxel = i;
        ++destPos;
        lastVoxel = voxelInput[i].voxelId;
      }
    }
  }
}

__global__
void kernel_toMortonCode(Vector3ui* inputVoxel, voxel_count numVoxel, OctreeVoxelID* outputVoxel)
{
  const voxel_count id = blockIdx.x * blockDim.x + threadIdx.x;
  const voxel_count numThreads = gridDim.x * blockDim.x;

  for (voxel_count i = id; i < numVoxel; i += numThreads)
    outputVoxel[i] = morton_code60(inputVoxel[i].x, inputVoxel[i].y, inputVoxel[i].z);
}

__global__
void kernel_toVoxels(const Vector3f* input_points, size_t num_points, Vector3ui* output_voxels, float voxel_side_length)
{
  const voxel_count id = blockIdx.x * blockDim.x + threadIdx.x;
  const voxel_count numThreads = gridDim.x * blockDim.x;
  for (voxel_count i = id; i < num_points; i += numThreads)
  {
    Vector3f pointf = input_points[i];
    Vector3ui voxel;
    voxel.x = uint32_t(pointf.x / voxel_side_length); // convert to meter and discrete in voxel
    voxel.y = uint32_t(pointf.y / voxel_side_length);  // convert to meter and discrete in voxel
    voxel.z = uint32_t(pointf.z / voxel_side_length);  // convert to meter and discrete in voxel
    output_voxels[i] = voxel;
  }
}


__global__
void kernel_transformDepthImage(DepthData* depth_image, gpu_voxels::Vector3f* d_point_cloud, Sensor* sensor, const DepthData invalid_measure)
{
  const uint32_t width = sensor->data_width;
  const uint32_t height = sensor->data_height;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x)
  {
    const uint32_t idx = i % width;
    const uint32_t idy = i / width;
    d_point_cloud[i] = sensor->sensorMeasureToSensorCoordinates(depth_image[i], idx, idy, invalid_measure);
  }
}

//__global__
//void kernel_preprocessObjectDepthImage(DepthData* d_depth_image, const uint32_t width, const uint32_t height,
//                                       const DepthData noSampleValue, const DepthData shadowValue,
//                                       const DepthData max_sensor_distance)
//{
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x)
//  {
//    const DepthData depth = d_depth_image[i];
//    DepthData new_depth = depth;
//    if ((depth >= max_sensor_distance))
//#ifdef KINECT_SHOW_OBJECT_MAX_RANGE_DATA
//      new_depth = max_sensor_distance;
//#else
//      new_depth = INVALID_DEPTH_DATA;
//#endif
//    else if ((depth == 0) | (depth == noSampleValue) | (depth == shadowValue))
//      new_depth = INVALID_DEPTH_DATA;
//    d_depth_image[i] = new_depth;
//  }
//}
//
//__global__
//void kernel_preprocessFreeSpaceDepthImage(DepthData* d_depth_image, const uint32_t width,
//                                          const uint32_t height, const DepthData noSampleValue,
//                                          const DepthData shadowValue, const DepthData max_sensor_distance)
//{
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x)
//  {
//    const DepthData depth = d_depth_image[i];
//    DepthData new_depth = depth;
//    if ((depth == 0) | (depth == noSampleValue) | (depth == shadowValue))
//    {
//#ifdef KINECT_CUT_FREE_SPACE_X
//      const int x = i % width;
//      if (min(x, width - x) < KINECT_CUT_FREE_SPACE_X)
//      new_depth = INVALID_DEPTH_DATA;
//#endif
//#ifdef KINECT_CUT_FREE_SPACE_Y
//      const int y = int(i / width);
//      if( min(y, height - y) < KINECT_CUT_FREE_SPACE_Y)
//      new_depth = INVALID_DEPTH_DATA;
//#endif
//#ifdef KINECT_FREE_NAN_MEASURES
//      new_depth = max_sensor_distance;
//#else
//      new_depth = INVALID_DEPTH_DATA;
//#endif
//    }
//    else if (depth >= max_sensor_distance)
//      new_depth = max_sensor_distance;
//    d_depth_image[i] = new_depth;
//  }
//}

__global__
void kernel_preprocessDepthImage(DepthData* d_depth_image, const uint32_t width, const uint32_t height,
                                 const SensorDataProcessing arguments)
{
  const DepthData invalid_measure = arguments.m_invalid_measure;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x)
  {
    const DepthData depth = d_depth_image[i];
    DepthData new_depth = depth;
    bool skip = false;

    if (arguments.m_cut_x_boarder > 0)
    {
      const int x = i % width;
      if (min(x, width - x) < arguments.m_cut_x_boarder)
      {
        skip = true;
        new_depth = invalid_measure;
      }
    }

    if (arguments.m_cut_y_boarder > 0)
    {
      const int x = i % width;
      if (min(x, width - x) < arguments.m_cut_y_boarder)
      {
        new_depth = invalid_measure;
        skip = true;
      }
    }

    if (!skip)
    {
      if (arguments.m_use_invalid_measures && depth == invalid_measure)
      {
        new_depth = arguments.m_sensor_range;
        skip = true;
      }

      if (!skip && depth > arguments.m_sensor_range)
      {
        if (arguments.m_remove_max_range_data)
          new_depth = invalid_measure;
        else
          new_depth = arguments.m_sensor_range;
        skip = true;
      }
    }
    d_depth_image[i] = new_depth;
  }
}

}
}

