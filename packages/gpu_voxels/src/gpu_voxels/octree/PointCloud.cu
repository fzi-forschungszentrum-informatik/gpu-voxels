// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2013-11-16
 *
 */
//----------------------------------------------------------------------/*

#define CUB_STDERR
#if __CUDACC_VER_MAJOR__ < 9
#include <thrust/system/cuda/detail/cub.h>
namespace cub = thrust::system::cuda::detail::cub_;
#else // Cuda 9 or higher
#define THRUST_CUB_NS_PREFIX namespace thrust {   namespace cuda_cub {
#define THRUST_CUB_NS_POSTFIX }   }
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX
namespace cub = thrust::cuda_cub::cub;
#endif

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <gpu_voxels/octree/PointCloud.h>
#include <gpu_voxels/octree/kernels/kernel_PointCloud.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/cuda_handling.h>

#include <icl_core_performance_monitor/PerformanceMonitor.h>

#include <algorithm>    // std::random_shuffle

using namespace std;

namespace gpu_voxels {
namespace NTree {

OctreeVoxelID transformKinectPointCloud(gpu_voxels::Vector3f* point_cloud, voxel_count num_points,
                                  thrust::device_vector<Voxel>& voxel, Sensor& sensor,
                                  gpu_voxels::Vector3f voxel_dimension)
{
  timespec time = getCPUTime();

  thrust::device_vector<Voxel> d_tmp_voxel(num_points);

  // copy to GPU
  thrust::device_vector<gpu_voxels::Vector3f> d_point_cloud = thrust::host_vector<gpu_voxels::Vector3f>(
      point_cloud, point_cloud + num_points);
  thrust::host_vector<Sensor> h_sensor(1);
  h_sensor[0] = sensor;
  thrust::device_vector<Sensor> d_sensor = h_sensor;

  LOGGING_INFO(OctreeLog, "copy to gpu: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
  time = getCPUTime();

  kernel_transformKinectPoints<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(D_PTR(d_point_cloud), num_points, D_PTR(d_tmp_voxel), D_PTR(d_sensor), voxel_dimension);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  LOGGING_INFO(OctreeLog, "kernel_transformKinectPoints: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
  time = getCPUTime();

  thrust::sort(d_tmp_voxel.begin(), d_tmp_voxel.end());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  LOGGING_INFO(OctreeLog, "thrust::sort: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
  time = getCPUTime();

  thrust::device_vector<OctreeVoxelID> count_voxel(NUM_BLOCKS * NUM_THREADS_PER_BLOCK);
  kernel_countVoxel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(D_PTR(d_tmp_voxel), num_points, D_PTR(count_voxel));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  LOGGING_INFO(OctreeLog, "kernel_countVoxel: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
  time = getCPUTime();

  thrust::inclusive_scan(count_voxel.begin(), count_voxel.end(), count_voxel.begin());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  LOGGING_INFO(OctreeLog, "thrust::inclusive_scan: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
  time = getCPUTime();

  OctreeVoxelID num_voxel = count_voxel.back();
  voxel.resize(num_voxel);

  kernel_combineEqualVoxel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(D_PTR(d_tmp_voxel), num_voxel, D_PTR(count_voxel), D_PTR(voxel), D_PTR(d_sensor));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  LOGGING_INFO(OctreeLog, "kernel_combineEqualVoxel: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
  time = getCPUTime();

  return num_voxel;
}

voxel_count transformKinectPointCloud_simple(gpu_voxels::Vector3f* d_point_cloud,
                                             const voxel_count num_points,
                                             thrust::device_vector<Voxel>& d_voxel, Sensor* d_sensor,
                                             const uint32_t resolution)
{
#define SORT_ON_GPU true
#define SORT_WITH_CUB true

  const string prefix = __FUNCTION__;
  const string temp_timer = prefix + "_temp";
  PERF_MON_START(prefix);
  PERF_MON_START(temp_timer);

  uint32_t num_threads = 128;
  uint32_t num_blocks = num_points / num_threads + 1;

  // transform point cloud from sensor coordinates to world coordinates and return these as morton code
  thrust::device_vector<OctreeVoxelID> d_tmp_voxel_id(num_points);
  kernel_transformKinectPoints_simple<<<num_blocks, num_threads>>>(d_point_cloud, num_points,
                                                                   D_PTR(d_tmp_voxel_id),
                                                                   d_sensor,
                                                                   resolution);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "ToWorldCoordinates", prefix);

  // TODO: try to get rid of this sorting step since it consumes the most time of this task; sorting this small sets is quite slow

  // sort on CPU
  if (SORT_ON_GPU)
  {
    if (SORT_WITH_CUB)
    {
      if (num_points != 0)
      {
        // use CUB since it's nearly twice as fast as thrust
        thrust::device_vector<OctreeVoxelID> d_tmp_voxel_id2(num_points);
        int num_items = (int) num_points;
        OctreeVoxelID *d_key_buf = D_PTR(d_tmp_voxel_id);
        OctreeVoxelID *d_key_alt_buf = D_PTR(d_tmp_voxel_id2);
        cub::DoubleBuffer<OctreeVoxelID> d_keys(d_key_buf, d_key_alt_buf);
        // Determine temporary device storage requirements
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items);
        // Allocate temporary storage
        HANDLE_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        // Run sorting operation
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items);
        HANDLE_CUDA_ERROR(cudaFree(d_temp_storage));
        if (d_keys.Current() != d_key_buf)
          d_tmp_voxel_id2.swap(d_tmp_voxel_id);
      }
    }
    else
      thrust::sort(d_tmp_voxel_id.begin(), d_tmp_voxel_id.end());
  }
  else
  {
    thrust::host_vector<OctreeVoxelID> h_tmp_voxel_id = d_tmp_voxel_id;
    thrust::sort(h_tmp_voxel_id.begin(), h_tmp_voxel_id.end());
    d_tmp_voxel_id = h_tmp_voxel_id;
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "SortByMorton", prefix);

  num_threads = 32; //have to use max. 32 threads for kernel_voxelize()
  num_blocks = num_points / num_threads + 1;

  thrust::device_vector<voxel_count> count_voxel(num_blocks + 1, 0);
  kernel_voxelize<true> <<<num_blocks, 32>>>(D_PTR(d_tmp_voxel_id), num_points, D_PTR(count_voxel), NULL);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  thrust::exclusive_scan(count_voxel.begin(), count_voxel.end(), count_voxel.begin());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  voxel_count num_voxel = count_voxel.back();

  //voxel = thrust::device_vector<Voxel>(num_voxel);
  d_voxel.resize(num_voxel);
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  HANDLE_CUDA_ERROR(cudaMemset(D_PTR(voxel), 0, num_voxel * sizeof(Voxel)));
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  kernel_voxelize<false> <<<num_blocks, 32>>>(D_PTR(d_tmp_voxel_id), num_points, D_PTR(count_voxel), D_PTR(d_voxel));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  num_threads = 128; //have to use max. 32 threads for kernel_voxelize()
  num_blocks = num_voxel / num_threads + 1;
  kernel_voxelize_finalStep<<<num_blocks, num_threads>>>(D_PTR(d_tmp_voxel_id),
  num_points,
  num_voxel,
  D_PTR(d_voxel),
  d_sensor);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  PERF_MON_PRINT_INFO_P(temp_timer, "Voxelize", prefix);
  PERF_MON_ADD_DATA_NONTIME_P("NumVoxel", num_voxel, prefix);
  PERF_MON_PRINT_INFO_P(prefix, "", prefix);
  return num_voxel;
#undef SORT_ON_GPU
}

/**
 * Calculates the minimal x, y and z value in the list.
 */
void min_max_XYZ_of_list(Vector3f* min_xyz, Vector3f* max_xyz, std::vector<Vector3f>& points,
                         uint32_t num_points)
{
  Vector3f temp_min, temp_max;
  temp_min = temp_max = points[0];

  for (size_t i = 1; i < num_points; i++)
  {
    temp_min.x = std::min(temp_min.x, points[i].x);
    temp_min.y = std::min(temp_min.y, points[i].y);
    temp_min.z = std::min(temp_min.z, points[i].z);

    temp_max.x = std::max(temp_max.x, points[i].x);
    temp_max.y = std::max(temp_max.y, points[i].y);
    temp_max.z = std::max(temp_max.z, points[i].z);
  }
  *min_xyz = temp_min;
  *max_xyz = temp_max;
}

void transformPoints(std::vector<Vector3f>& points, uint3comp* out_points, uint32_t num_points,
                     Vector3f offset, float scaling)
{
  for (size_t i = 0; i < num_points; i++)
  {
    uint3comp coordinates;
    coordinates.x = (uint32_t) (floor((points[i].x + offset.x) * scaling));
    coordinates.y = (uint32_t) (floor((points[i].y + offset.y) * scaling));
    coordinates.z = (uint32_t) (floor((points[i].z + offset.z) * scaling));
    out_points[i] = coordinates;
  }
}

Vector3ui getMapDimensions(std::vector<Vector3f>& point_cloud, Vector3f& offset, float scaling)
{
  Vector3ui map_dimensions;
  uint32_t num_points = point_cloud.size();
  Vector3f min, max;
  min_max_XYZ_of_list(&min, &max, point_cloud, num_points);
  offset.x = -min.x;
  offset.y = -min.y;
  offset.z = -min.z;
  map_dimensions.x = static_cast<uint32_t>(floor((max.x + offset.x) * scaling) + 1);
  map_dimensions.y = static_cast<uint32_t>(floor((max.y + offset.y) * scaling) + 1);
  map_dimensions.z = static_cast<uint32_t>(floor((max.z + offset.z) * scaling) + 1);
  return map_dimensions;
}

//void transformPointCloud(std::vector<Vector3f>& point_cloud, std::vector<Vector3ui>& points,
//                         Vector3ui& map_dimensions, float scaling)
//{
//  Vector3f offset;
//  uint32_t num_points = point_cloud.size();
//  map_dimensions = getMapDimensions(point_cloud, offset, scaling);
//  printf("Map dimensions in Voxel: %u %u %u \n", map_dimensions.x, map_dimensions.y, map_dimensions.z);
//
//  uint3comp* transformed_points = new uint3comp[num_points];
//  transformPoints(point_cloud, transformed_points, num_points, offset, scaling);
//
//  // remove duplicates
//  thrust::host_vector<uint3comp> h_transformed_points_copy = thrust::host_vector<uint3comp>(
//      transformed_points, transformed_points + num_points);
//  thrust::sort(h_transformed_points_copy.begin(), h_transformed_points_copy.end());
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); // sync just like for plain kernel calls
//  h_transformed_points_copy.erase(
//      thrust::unique(h_transformed_points_copy.begin(), h_transformed_points_copy.end()),
//      h_transformed_points_copy.end());
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); // sync just like for plain kernel calls
//
//  uint32_t num_different_voxel = h_transformed_points_copy.size();
//  printf("num_different_voxel: %u\n", num_different_voxel);
//
//  // shuffle otherwise data is sorted and the octree build would have an advantage
//  uint3comp* ptr = h_transformed_points_copy.data();
//  std::random_shuffle(ptr, ptr + num_different_voxel);
//
//  points.resize(num_different_voxel);
//  for (uint32_t i = 0; i < num_different_voxel; ++i)
//    points[i] = Vector3ui(ptr[i].x, ptr[i].y, ptr[i].z);
//  delete[] transformed_points;
//}

// uses the device for some tasks to speed things up
void transformPointCloud(std::vector<Vector3f>& point_cloud, std::vector<Vector3ui>& points,
                         Vector3ui& map_dimensions, float scaling)
{
  Vector3f offset;
  uint32_t num_points = point_cloud.size();
  map_dimensions = getMapDimensions(point_cloud, offset, scaling);
  LOGGING_INFO(OctreeLog, "Map dimensions in Voxel: " << map_dimensions << endl);

  uint3comp* transformed_points = new uint3comp[num_points];
  transformPoints(point_cloud, transformed_points, num_points, offset, scaling);

  // remove duplicates
  thrust::device_vector<uint3comp> d_transformed_points_copy = thrust::host_vector<uint3comp>(
      transformed_points, transformed_points + num_points);
  thrust::sort(d_transformed_points_copy.begin(), d_transformed_points_copy.end());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); // sync just like for plain kernel calls
  delete[] transformed_points;
  d_transformed_points_copy.erase(
      thrust::unique(d_transformed_points_copy.begin(), d_transformed_points_copy.end()),
      d_transformed_points_copy.end());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); // sync just like for plain kernel calls

  uint32_t num_different_voxel = d_transformed_points_copy.size();
  LOGGING_INFO(OctreeLog, "num_different_voxel: " <<  num_different_voxel << endl);

  // shuffle otherwise data is sorted and the octree build would have an advantage
  thrust::host_vector<uint3comp> h_transformed_points = d_transformed_points_copy;
  uint3comp* ptr = h_transformed_points.data();
  std::random_shuffle(ptr, ptr + num_different_voxel);

  points.resize(num_different_voxel);
  for (uint32_t i = 0; i < num_different_voxel; ++i)
    points[i] = Vector3ui(ptr[i].x, ptr[i].y, ptr[i].z);
}

//  pcl::PointCloud<pcl::PointXYZI>::Ptr
//  pcl::OpenNIGrabber::convertToXYZIPointCloud (const boost::shared_ptr<openni_wrapper::IRImage> &ir_image,
//                                               const boost::shared_ptr<openni_wrapper::DepthImage> &depth_image) const
//  {
//    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZI> > cloud (new pcl::PointCloud<pcl::PointXYZI > ());
//
//    cloud->header.frame_id = rgb_frame_id_;
//    cloud->height = depth_height_;
//    cloud->width = depth_width_;
//    cloud->is_dense = false;
//
//    cloud->points.resize (cloud->height * cloud->width);
//
//    //float constant = 1.0f / device_->getImageFocalLength (cloud->width);
//    register float constant_x = 1.0f / device_->getImageFocalLength (cloud->width);
//    register float constant_y = 1.0f / device_->getImageFocalLength (cloud->width);
//    register float centerX = ((float)cloud->width - 1.f) / 2.f;
//    register float centerY = ((float)cloud->height - 1.f) / 2.f;
//
//    if (pcl_isfinite (rgb_focal_length_x_))
//      constant_x = 1.0f / static_cast<float> (rgb_focal_length_x_);
//
//    if (pcl_isfinite (rgb_focal_length_y_))
//      constant_y = 1.0f / static_cast<float> (rgb_focal_length_y_);
//
//    if (pcl_isfinite (rgb_principal_point_x_))
//      centerX = static_cast<float>(rgb_principal_point_x_);
//
//    if (pcl_isfinite (rgb_principal_point_y_))
//      centerY = static_cast<float>(rgb_principal_point_y_);
//
//    register const XnDepthPixel* depth_map = depth_image->getDepthMetaData ().Data ();
//    register const XnIRPixel* ir_map = ir_image->getMetaData ().Data ();
//
//    if (depth_image->getWidth () != depth_width_ || depth_image->getHeight () != depth_height_)
//    {
//      static unsigned buffer_size = 0;
//      static boost::shared_array<unsigned short> depth_buffer ((unsigned short*)(NULL));
//      static boost::shared_array<unsigned short> ir_buffer ((unsigned short*)(NULL));
//
//      if (buffer_size < depth_width_ * depth_height_)
//      {
//        buffer_size = depth_width_ * depth_height_;
//        depth_buffer.reset (new unsigned short [buffer_size]);
//        ir_buffer.reset (new unsigned short [buffer_size]);
//      }
//
//      depth_image->fillDepthImageRaw (depth_width_, depth_height_, depth_buffer.get ());
//      depth_map = depth_buffer.get ();
//
//      ir_image->fillRaw (depth_width_, depth_height_, ir_buffer.get ());
//      ir_map = ir_buffer.get ();
//    }
//
//    register int depth_idx = 0;
//    float bad_point = std::numeric_limits<float>::quiet_NaN ();
//
//    for (unsigned int v = 0; v < depth_height_; ++v)
//    {
//      for (register unsigned int u = 0; u < depth_width_; ++u, ++depth_idx)
//      {
//        pcl::PointXYZI& pt = cloud->points[depth_idx];
//        /// @todo Different values for these cases
//        // Check for invalid measurements
//        if (depth_map[depth_idx] == 0 ||
//            depth_map[depth_idx] == depth_image->getNoSampleValue () ||
//            depth_map[depth_idx] == depth_image->getShadowValue ())
//        {
//          pt.x = pt.y = pt.z = bad_point;
//        }
//        else
//        {
//          pt.z = depth_map[depth_idx] * 0.001f;
//          pt.x = (static_cast<float> (u) - centerX) * pt.z * constant_x;
//          pt.y = (static_cast<float> (v) - centerY) * pt.z * constant_y;
//        }
//
//        pt.data_c[0] = pt.data_c[1] = pt.data_c[2] = pt.data_c[3] = 0;
//        pt.intensity = static_cast<float> (ir_map[depth_idx]);
//      }
//    }
//    cloud->sensor_origin_.setZero ();
//    cloud->sensor_orientation_.w () = 1.0;
//    cloud->sensor_orientation_.x () = 0.0;
//    cloud->sensor_orientation_.y () = 0.0;
//    cloud->sensor_orientation_.z () = 0.0;
//    return (cloud);
//  }

struct Comp_is_valid_point
{
  __host__ __device__
  __forceinline__
  bool operator()(gpu_voxels::Vector3f v)
  {
    return !::isnan(v.x) & !::isnan(v.y) & !::isnan(v.z);
  }
}
;

/**
 * Needs a cudaDeviceSynchronize() afterwards
 */
void removeInvalidPoints(thrust::device_vector<gpu_voxels::Vector3f>& d_depth_image)
{
  thrust::device_vector<gpu_voxels::Vector3f> temp(d_depth_image.size());
  uint32_t new_size = thrust::copy_if(d_depth_image.begin(), d_depth_image.end(), temp.begin(),
                                      Comp_is_valid_point()) - temp.begin();
  temp.resize(new_size);
  temp.swap(d_depth_image);
}

}  // end of ns
}  // end of ns
