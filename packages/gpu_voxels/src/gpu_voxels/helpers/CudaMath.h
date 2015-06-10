// this is for emacs file handling -&- mode: c++; indent-tabs-mode: nil -&-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-08-23
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_CUDAMATH_H_INCLUDED
#define GPU_VOXELS_CUDAMATH_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <vector>

namespace gpu_voxels {


__host__ __device__
inline gpu_voxels::Matrix4f roll(float angle)
{
  gpu_voxels::Matrix4f m;
  m.a11 = 1;
  m.a12 = 0;
  m.a13 = 0;
  m.a14 = 0;
  m.a21 = 0;
  m.a22 = cos(angle);
  m.a23 = sin(angle);
  m.a24 = 0;
  m.a31 = 0;
  m.a32 = -sin(angle);
  m.a33 = cos(angle);
  m.a34 = 0;
  m.a41 = 0;
  m.a42 = 0;
  m.a43 = 0;
  m.a44 = 1;
  return m;
}

__host__ __device__
inline gpu_voxels::Matrix4f pitch(float angle)
{
  gpu_voxels::Matrix4f m;
  m.a11 = cos(angle);
  m.a12 = 0;
  m.a13 = sin(angle);
  m.a14 = 0;
  m.a21 = 0;
  m.a22 = 1;
  m.a23 = 0;
  m.a24 = 0;
  m.a31 = -sin(angle);
  m.a32 = 0;
  m.a33 = cos(angle);
  m.a34 = 0;
  m.a41 = 0;
  m.a42 = 0;
  m.a43 = 0;
  m.a44 = 1;
  return m;
}

__host__ __device__
inline gpu_voxels::Matrix4f yaw(float angle)
{
  gpu_voxels::Matrix4f m;
  m.a11 = cos(angle);
  m.a12 = sin(angle);
  m.a13 = 0;
  m.a14 = 0;
  m.a21 = -sin(angle);
  m.a22 = cos(angle);
  m.a23 = 0;
  m.a24 = 0;
  m.a31 = 0;
  m.a32 = 0;
  m.a33 = 1;
  m.a34 = 0;
  m.a41 = 0;
  m.a42 = 0;
  m.a43 = 0;
  m.a44 = 1;
  return m;
}

__host__ __device__
inline gpu_voxels::Matrix4f rotateYPR(float _yaw, float _pitch, float _roll)
{
  return yaw(_yaw) * (pitch(_pitch) * roll(_roll));
}

__host__ __device__
inline gpu_voxels::Matrix4f rotateRPY(float _yaw, float _pitch, float _roll)
{
  return roll(_roll) * (pitch(_pitch) * yaw(_yaw));
}


/*!
 * \brief CUDA class CudaMath
 *
 * Basic gpu accelerated linear algebra functions
 */
class CudaMath
{
public:

  //! Constructor
  CudaMath();

  /*! Constructor with custom member intialization
   *  \param max_nr_of_devices The maximum number of graphics cards compatible with CUDA
   *  \param max_nr_of_blocks The maximum available number of CUDA blocks on the GPU.
   *  \param max_nr_of_threads_per_block The maximum available number of CUDA threads per block on the GPU.
   */
  CudaMath(unsigned int max_nr_of_devices, unsigned int max_nr_of_blocks, unsigned int max_nr_of_threads_per_block);

  //! Destructor
  ~CudaMath();

  //! For more comfortable matrix output
  void printMatrix(const Matrix4f& matrix);

  //! Device load balancing functions
  void computeLinearLoad(const uint32_t nr_of_items, uint32_t* blocks,
                         uint32_t* threads_per_block);


  //! Reformatting functions (not gpu accelerated)
  void Vec3ToMat4(const Vector3f& vec_in, Matrix4f& mat_out);
  void Mat3ToMat4(const Matrix3d& in, Matrix4f& out);
  void Mat3AndVec3ToMat4(const Matrix3d& mat_in, const Vector3f& vec_in, Matrix4f& out);
  void Mat3AndVec4ToMat4(const Matrix3d& mat_in, const Vector4d& vec_in, Matrix4f& out);

  void Mat3AndVec3ToMat4(const Matrix3f& mat_in, const Vector3f& vec_in, Matrix4f& mat_out);

  //! Transpose a matrix
  void transpose(const Matrix3d& in, Matrix3d& out);
  void transpose(const Matrix4f& in, Matrix4f& out);

  //! Invert a matrix (not gpu accelerated because of low benefit)
  // void invert(const Matrix4f& in, Matrix4f& out);

  /*! Transform Matrix (corresponding to mcal_kinematic::tTransformCoordinates style)
         computes absoultes[] = base    * relatives[]
      or computes absoultes[] = base^-1 * relatives[] if invert_base is true
      warning: the inversion-part is not gpu accelerated
   */
  void transform(unsigned int nr_of_transformations, Matrix4f& base, const Matrix4f* relatives,
                 Matrix4f* absolutes, bool invert_base);

  /*! Interpolate linear between the values \a value1 and \a value2 using the given \a ratio.
   *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
   *  middle.
   */
  static float interpolateLinear(float value1, float value2, float ratio);


  /*! Interpolate linear between the values \a value1 and \a value2 using the given \a ratio.
   *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
   *  middle.
   */
  static double interpolateLinear(double value1, double value2, double ratio);

  /*! Interpolate linear between the robot JointValueMaps \a joint_state1 and \a joint_state2
   *  using the given \a ratio.
   *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
   *  middle.
   */
  static JointValueMap interpolateLinear(const JointValueMap& joint_state1,
                                         const JointValueMap& joint_state2, float ratio);

  /*! Interpolate linear between the robot joint vectors \a joint_state1 and \a joint_state2
   *  using the given \a ratio.
   *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
   *  middle.
   */
  static std::vector<float> interpolateLinear(const std::vector<float>& joint_state1,
                                              const std::vector<float>& joint_state2, float ratio);

  /*! Interpolate linear between the robot joint vectors \a joint_state1 and \a joint_state2
   *  using the given \a ratio.
   *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
   *  middle.
   */
  static std::vector<double> interpolateLinear(const std::vector<double>& joint_state1,
                                               const std::vector<double>& joint_state2, double ratio);

  /*! Transform each point of the point cloud by multiplication of \a matrix with that point.
   */
  static void transform(std::vector<Vector3f> &host_point_cloud, Matrix4f matrix);

private:
  const unsigned int m_max_nr_of_devices;
  const unsigned int m_max_nr_of_blocks;
  const unsigned int m_max_nr_of_threads_per_block;

};

} // end of namespace

#endif
