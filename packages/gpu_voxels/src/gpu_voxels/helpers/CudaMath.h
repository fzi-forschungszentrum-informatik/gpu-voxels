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
#ifndef GPU_VOXELS_HELPERS_CUDAMATH_H_INCLUDED
#define GPU_VOXELS_HELPERS_CUDAMATH_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/MathHelpers.h>

namespace gpu_voxels {

//! Reformatting function (not gpu accelerated)
__host__ __device__
inline void Vec3ToMat4(const Vector3f& vec_in, Matrix4f& mat_out)
{
  mat_out.a14 = vec_in.x;
  mat_out.a24 = vec_in.x;
  mat_out.a34 = vec_in.x;
  mat_out.a44 = 1;
}

//! Reformatting function (not gpu accelerated)
__host__ __device__
inline void Mat3ToMat4(const Matrix3d& mat_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a12;  mat_out.a13 = mat_in.a13;  mat_out.a14 = 0;
  mat_out.a21 = mat_in.a21;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a13;  mat_out.a24 = 0;
  mat_out.a31 = mat_in.a31;  mat_out.a32 = mat_in.a32;  mat_out.a33 = mat_in.a13;  mat_out.a34 = 0;
  mat_out.a41 = 0;           mat_out.a42 = 0;           mat_out.a43 = 0;           mat_out.a44 = 1;
}

//! Reformatting function (not gpu accelerated)
__host__ __device__
inline void Mat3AndVec3ToMat4(const Matrix3d& mat_in, const Vector3f& vec_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a12;  mat_out.a13 = mat_in.a13;  mat_out.a14 = vec_in.x;
  mat_out.a21 = mat_in.a21;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a23;  mat_out.a24 = vec_in.y;
  mat_out.a31 = mat_in.a31;  mat_out.a32 = mat_in.a32;  mat_out.a33 = mat_in.a33;  mat_out.a34 = vec_in.z;
  mat_out.a41 = 0;           mat_out.a42 = 0;           mat_out.a43 = 0;           mat_out.a44 = 1;
}

//! Reformatting function (not gpu accelerated)
__host__ __device__
inline void Mat3AndVec3ToMat4(const Matrix3f& mat_in, const Vector3f& vec_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a12;  mat_out.a13 = mat_in.a13;  mat_out.a14 = vec_in.x;
  mat_out.a21 = mat_in.a21;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a23;  mat_out.a24 = vec_in.y;
  mat_out.a31 = mat_in.a31;  mat_out.a32 = mat_in.a32;  mat_out.a33 = mat_in.a33;  mat_out.a34 = vec_in.z;
  mat_out.a41 = 0;           mat_out.a42 = 0;           mat_out.a43 = 0;           mat_out.a44 = 1;
}

//! Reformatting function (not gpu accelerated)
__host__ __device__
inline void Mat3AndVec4ToMat4(const Matrix3d& mat_in, const Vector4d& vec_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a12;  mat_out.a13 = mat_in.a13;  mat_out.a14 = vec_in.x;
  mat_out.a21 = mat_in.a21;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a23;  mat_out.a24 = vec_in.y;
  mat_out.a31 = mat_in.a31;  mat_out.a32 = mat_in.a32;  mat_out.a33 = mat_in.a33;  mat_out.a34 = vec_in.z;
  mat_out.a41 = 0;           mat_out.a42 = 0;           mat_out.a43 = 0;           mat_out.a44 = vec_in.w;
}

//! Transpose a matrix
__host__ __device__
inline void transpose(const Matrix4f& mat_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a21;  mat_out.a13 = mat_in.a31;  mat_out.a14 = mat_in.a41;
  mat_out.a21 = mat_in.a12;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a32;  mat_out.a24 = mat_in.a42;
  mat_out.a31 = mat_in.a13;  mat_out.a32 = mat_in.a23;  mat_out.a33 = mat_in.a33;  mat_out.a34 = mat_in.a43;
  mat_out.a41 = mat_in.a14;  mat_out.a42 = mat_in.a24;  mat_out.a43 = mat_in.a34;  mat_out.a44 = mat_in.a44;
}


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

/*!
 * \brief rotateYPR Constructs a rotation matrix
 * \param _yaw
 * \param _pitch
 * \param _roll
 * \return
 */
__host__ __device__
inline gpu_voxels::Matrix4f rotateYPR(float _yaw, float _pitch, float _roll)
{
  return yaw(_yaw) * (pitch(_pitch) * roll(_roll));
}

/*!
 * \brief rotateRPY Constructs a rotation matrix
 * \param _yaw
 * \param _pitch
 * \param _roll
 * \return
 */
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

  //! Destructor
  ~CudaMath();

  /*! Transform Matrix (corresponding to mcal_kinematic::tTransformCoordinates style)
         computes absoultes[] = base    * relatives[]
      or computes absoultes[] = base^-1 * relatives[] if invert_base is true
      warning: the inversion-part is not gpu accelerated
   */
  void transform(unsigned int nr_of_transformations, Matrix4f& base, const Matrix4f* relatives,
                 Matrix4f* absolutes, bool invert_base);

};

} // end of namespace

#endif
