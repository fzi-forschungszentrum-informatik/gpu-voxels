// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2015 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Christian Juelg
 * \date    2015-08-18
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXELMAP_DISTANCE_VOXEL_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_DISTANCE_VOXEL_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/common_defines.h>

#include <cstddef>
#include <ostream>
#include <sstream>

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

//forward declaration
namespace gpu_voxels{
  namespace voxelmap {
    __device__ __host__
    Vector3i mapToVoxelsSigned(int linear_id, const Vector3ui &dimensions);
  };
};

namespace gpu_voxels {

/**
 * @brief Voxel holding information about next obstacle and distance
 */
class DistanceVoxel //: public AbstractVoxel
{
public:

  typedef int32_t pba_dist_t;
  typedef uint32_t pba_voxel_t; // This stores the xyz-postiton as 3x 9Bit interger (+ x3 1Bit as "uninitialized" marker)

  __host__ __device__
  const Vector3ui getObstacle() const;

  /**
   * @brief get squared distance (optimization: don't compute square root; distances are not accumulated during DVM computation)
   *
   * To query a single DistanceVoxel's distance to its closest obstacle in host code, use DVM.obstacleDistance / DVM.squaredObstacleDistance.
   *
   * To get all DistanceVoxels obstacle distances, use DVM.extract_distances.
   *
   */
  __host__ __device__
  pba_dist_t squaredObstacleDistance(Vector3i this_position) const;

  __host__ __device__
  DistanceVoxel();

  /**
   * @brief sets obstacle
   */
  __host__ __device__
  DistanceVoxel(const Vector3ui& o);

  /**
   * @brief sets obstacle
   */
  __host__ __device__
  explicit DistanceVoxel(const pba_voxel_t o);

  /**
   * @brief sets obstacle
   */
  __host__ __device__
  DistanceVoxel(const uint x, const uint y, const uint z);

  /**
   * @brief sets obstacle
   */
  __host__ __device__
  DistanceVoxel(const uint3& o);

  /**
   * @brief sets obstacle
   */
  __host__ __device__
  void setObstacle(const Vector3ui& o);

  /**
   * @brief sets obstacle
   */
  __host__ __device__
  void setObstacle(const Vector3i& o);

  /**
   * @brief setPBAUninitialised
   */
  __host__ __device__
  void setPBAUninitialised();

  /**
   * @brief NOP
   */
  __host__ __device__
  void insert(const uint32_t voxel_meaning);

  __host__ __device__
  bool isOccupied(float col_threshold) const;

  /**
   * @brief insert Inserts new data into this voxel
   * @param voxel_type if type is occupied, mark voxel as obstacle
   */
  __host__ __device__
  void insert(const Vector3ui& current_position, const uint32_t voxel_meaning);

  //  /**
  //   * @brief used in combination with thrust::replace_if to replace all obstacle values
  //   */
  //  struct is_not_obstacle
  //  {
  //    __host__ __device__
  //    bool operator()(const DistanceVoxel& a) const {
  //      return a.getDistance() != DISTANCE_OBSTACLE;
  //    }
  //  };

  struct extract_byte_distance
  {
    typedef thrust::tuple<DistanceVoxel, int > tuple_t;
    typedef int8_t free_space_t;

    Vector3i dims;
    int robot_radius;

    __host__ __device__
    extract_byte_distance(Vector3i dims, int robot_radius) : dims(dims), robot_radius(robot_radius) {}

    __host__ __device__
    free_space_t operator()(const tuple_t tuple) const {
      // get pos from zipiterator/tuple

      DistanceVoxel dv = thrust::get<0>(tuple);
      int linear_id = thrust::get<1>(tuple);

      // pos is the position of the voxel dv
      Vector3i pos;
      pos.z = linear_id / (dims.x * dims.y);
      pos.y = (linear_id -= pos.z * (dims.x * dims.y)) / dims.x;
      pos.x = (linear_id -= pos.y * dims.x);

      //TODO: check for negative result
      //TODO: define return value with meaningful name
       //TODO remove
      if (dims.x-pos.x < 0) printf("pos.x is too large: %d, id: %d\n", pos.x, thrust::get<1>(tuple));
      if (dims.y-pos.y < 0) printf("pos.y is too large! %d, id: %d\n", pos.y, thrust::get<1>(tuple));
      if (dims.z-pos.z < 0) printf("pos.z is too large! %d, id: %d\n", pos.z, thrust::get<1>(tuple));

      if (pos.x > (dims.x-robot_radius)) return dims.x-pos.x < 0 ? 0 : dims.x-pos.x;
      if (pos.y > (dims.y-robot_radius)) return dims.y-pos.y < 0 ? 0 : dims.y-pos.y;
      if (pos.z > (dims.z-robot_radius)) return dims.z-pos.z < 0 ? 0 : dims.z-pos.z;

      //TODO check for negative position
      //TODO: define return value with meaningful name
       //TODO remove
      if (pos.x < 0) printf("pos.x is negative! %d\n", pos.x);
      if (pos.y < 0) printf("pos.y is negative! %d\n", pos.y);
      if (pos.z < 0) printf("pos.z is negative! %d\n", pos.z);

      if (pos.x < robot_radius) return pos.x < 0 ? 0 : pos.x;
      if (pos.y < robot_radius) return pos.y < 0 ? 0 : pos.y;
      if (pos.z < robot_radius) return pos.z < 0 ? 0 : pos.z;

      int free_space = sqrtf(dv.squaredObstacleDistance(pos));
      if (free_space > SCHAR_MAX) return SCHAR_MAX;
      if (free_space < 0) printf("free_space is negative! %d\n", pos.x); //TODO remove
      return free_space;
    }
  };

  struct init_floodfill_distance
  {
    typedef int16_t manhattan_dist_t;

    int robot_radius;

    __host__ __device__
    init_floodfill_distance(int robot_radius) : robot_radius(robot_radius) {}

    __host__ __device__
    manhattan_dist_t operator()(const extract_byte_distance::free_space_t dist) const {
      // get pos from zipiterator/tuple

      if (dist <= robot_radius) return MANHATTAN_DISTANCE_TOO_CLOSE;

      //else set uninitialised
      return MANHATTAN_DISTANCE_UNINITIALIZED;
    }
  };

  //TODO: define binop in distancevoxel? round down to
  //TODO: thrust transform pbaDistanceVoxmap->getDeviceDataPtr() to byte[] (round down to 0..255, cap at 255; could even parameterize on robot size and create boolean

    /**
     * @brief used in combination with thrust::count_if to determine number of (un)initialised voxels
     */
    struct is_initialised
    {
      __host__ __device__
      bool operator()(const DistanceVoxel& a) const {
        if (a.getObstacle().x != PBA_UNINITIALISED_COORD) return true;
        if (a.getObstacle().y != PBA_UNINITIALISED_COORD) return true;
        if (a.getObstacle().z != PBA_UNINITIALISED_COORD) return true;

        return false;
      }
    };

  __host__ __device__
  operator uint3() const ;

//  /**
//   * @brief used in combination with thrust::transform to replace all obstacle values
//   */
//  struct pba_transform
//  {
//    __host__ __device__
//    DistanceVoxel operator()(const DistanceVoxel& a) const {
//      if (a.distanceToObstacle() == DISTANCE_OBSTACLE) {
//        DistanceVoxel pba_obstacle_voxel;
//        pba_obstacle_voxel.setObstacle(a.getObstacle(), PBA_OBSTACLE_DISTANCE);
//        return pba_obstacle_voxel;
//      } else {
//        DistanceVoxel pba_uninitialised_voxel;
//        pba_uninitialised_voxel.setPBAUninitialised();
//        return pba_uninitialised_voxel;

//        //TODO: get distance from threadidx, blockidx; protect using #ifdef
//        // doesnt work; binary transform including counting iterator?
//        // this results in linear adresses; to get x/y values, somehow pass in Vector3ui dims, using templates?
//        //-> simpler using kernel that just knows its idx... or just roll this into first phase1 sweep kernel: check for uninitialised; if not, check for distance_obstacle; then set fitting x/y
//        //-> all unneccessary, position information is already present in all obstacles (set bei insert(pos, type))
//      }
//    }
//  };

//  /**
//   * @brief used in combination with thrust::transform to replace all obstacle values
//   */
//  struct obstacle_zero_transform
//  {
//    __host__ __device__
//    DistanceVoxel operator()(const DistanceVoxel& a) const {
//      if (a.getDistance() == DISTANCE_OBSTACLE) {
//        DistanceVoxel pba_obstacle_voxel;
//        pba_obstacle_voxel.setObstacle(a.getObstacle(), PBA_OBSTACLE_DISTANCE);
//        return pba_obstacle_voxel;
//      } else {
//        // don't touch non-obstacles
//        return a;
//      }
//    }
//  };


  /**
   * @brief hold accumulated differences and count of differing voxels
   */
  struct accumulated_diff
  {
    __host__ __device__
    accumulated_diff() : sum(0), count(0), maxerr(0) {}

    __host__ __device__
    accumulated_diff(double d) : sum(d), count((d != 0) ? 1 : 0), maxerr(d) {}

    __host__
    std::string str() {
      std::stringstream ss;
      ss << "sum of differences: " << sum
         << ", number of differences: " << count
         << ", avg error: " << ((sum>0)?(sum/count):0)
         << ", max error: " << maxerr;
      return ss.str();
    }

    double sum;
    uint32_t count;
    double maxerr;
  };

  /**
   * @brief calc absolute difference
   */
  struct diff_op
  {
//    typedef thrust::tuple<thrust::device_ptr<gpu_voxels::DistanceVoxel>, thrust::counting_iterator<int> > tuple_t;
    typedef thrust::tuple<DistanceVoxel, int > tuple_t;
//    typedef thrust::zip_iterator<tuple_t> iterator_t;
    Vector3ui dims;

    diff_op(Vector3ui dims) : dims(dims) {}

    __host__ __device__
    double operator()(const tuple_t ia, const tuple_t ib) const
    {
      DistanceVoxel a = thrust::get<0>(ia);
      int linear_id_a = thrust::get<1>(ia);

      DistanceVoxel b = thrust::get<0>(ib);
      int linear_id_b = thrust::get<1>(ib);

      if (a.getObstacle().x == PBA_UNINITIALISED_COORD) return MAX_OBSTACLE_DISTANCE;
      if (a.getObstacle().y == PBA_UNINITIALISED_COORD) return MAX_OBSTACLE_DISTANCE;
      if (a.getObstacle().z == PBA_UNINITIALISED_COORD) return MAX_OBSTACLE_DISTANCE;
      if (b.getObstacle().x == PBA_UNINITIALISED_COORD) return MAX_OBSTACLE_DISTANCE;
      if (b.getObstacle().y == PBA_UNINITIALISED_COORD) return MAX_OBSTACLE_DISTANCE;
      if (b.getObstacle().z == PBA_UNINITIALISED_COORD) return MAX_OBSTACLE_DISTANCE;

      Vector3i pos_a = voxelmap::mapToVoxelsSigned(linear_id_a, dims);
      int32_t da = a.squaredObstacleDistance(pos_a);

      Vector3i pos_b = voxelmap::mapToVoxelsSigned(linear_id_b, dims);
      int32_t db = b.squaredObstacleDistance(pos_b);

      //getDistance is squared distance; get root before subtracting
      double diff = sqrt((double)da) - sqrt((double)db);

      return (diff < 0) ? -diff : diff; //accumulate absolute differences
    }
  };

  /**
   * @brief accumulate diffs and count
   */
  struct accumulate_op
  {
    __host__ __device__
    accumulated_diff operator()(const accumulated_diff d1, const accumulated_diff d2) const
    {
      accumulated_diff res(d1);
      res.count += d2.count;
      res.sum += d2.sum;
      res.maxerr = (d2.maxerr > d1.maxerr) ? d2.maxerr : d1.maxerr;
      return res;
    }
  };

  /**
   * @brief operator >> Overloaded ostream operator.
   */
  template<typename T>
  __host__
  friend T& operator<<(T& os, const DistanceVoxel& dv)
  {
    os << dv.getObstacle().x << "/" << dv.getObstacle().y << "/" << dv.getObstacle().z;
    return os;
  }

protected:

  pba_voxel_t m_obstacle;
};

} // end of ns

#endif
