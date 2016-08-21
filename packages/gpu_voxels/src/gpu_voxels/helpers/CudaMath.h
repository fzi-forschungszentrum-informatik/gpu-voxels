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

namespace detail {

/*!
 * \brief swapLine Swaps two rows in a NxM matrix
 * Taken from http://www.virtual-maxim.de/matrix-invertieren-in-c-plus-plus/
 * \param mat
 * \param line1 Index of line1 to swap
 * \param line2 Index of line2 to swap
 * \return false, if line1 or line2 is out of bounds
 */
template <size_t N,size_t M>
__host__ __device__
inline bool swapLine(double mat[N][M], size_t line1, size_t line2)
{
    if(line1 >= N || line2 >= N)
        return false;

    for(size_t i = 0; i < M; ++i)
    {
        double t = mat[line1][i];
        mat[line1][i] = mat[line2][i];
        mat[line2][i] = t;
    }

    return true;
}



/*!
 * \brief invertMatrix inverts a NxM matrix via Gauß-Jordan-Algorithm
 * Taken from http://www.virtual-maxim.de/matrix-invertieren-in-c-plus-plus/
 * \param mat Input matrix
 * \param inv Inverted output matrix
 * \return false, if matrix can not be inverted
 */
template <size_t N>
__host__ __device__
inline bool invertMatrix(const double mat[N][N], double inv[N][N])
{
    // Eine Nx2N Matrix für den Gauß-Jordan-Algorithmus aufbauen
    double A[N][2*N];
    for(size_t i = 0; i < N; ++i)
    {
        for(size_t j = 0; j < N; ++j)
            A[i][j] = mat[i][j];
        for(size_t j = N; j < 2*N; ++j)
            A[i][j] = (i==j-N) ? 1.0 : 0.0;
    }

    // Gauß-Algorithmus.
    for(size_t k = 0; k < N-1; ++k)
    {
        // Zeilen vertauschen, falls das Pivotelement eine Null ist
        if(A[k][k] == 0.0)
        {
            for(size_t i = k+1; i < N; ++i)
            {
                if(A[i][k] != 0.0)
                {
                    swapLine<N, 2*N>(A,k,i);
                    break;
                }
                else if(i==N-1)
                    return false; // Es gibt kein Element != 0
            }
        }

        // Einträge unter dem Pivotelement eliminieren
        for(size_t i = k+1; i < N; ++i)
        {
            double p = A[i][k]/A[k][k];
            for(size_t j = k; j < 2*N; ++j)
                A[i][j] -= A[k][j]*p;
        }
    }

    // Determinante der Matrix berechnen
    double det = 1.0;
    for(size_t k = 0; k < N; ++k)
        det *= A[k][k];

    if(det == 0.0)  // Determinante ist =0 -> Matrix nicht invertierbar
        return false;

    // Jordan-Teil des Algorithmus durchführen
    for(size_t k = N-1; k > 0; --k)
    {
        for(int i = k-1; i >= 0; --i)
        {
            double p = A[i][k]/A[k][k];
            for(size_t j = k; j < 2*N; ++j)
                A[i][j] -= A[k][j]*p;
        }
    }

    // Einträge in der linker Matrix auf 1 normieren und in inv schreiben
    for(size_t i = 0; i < N; ++i)
    {
        const double f = A[i][i];
        for(size_t j = N; j < 2*N; ++j)
            inv[i][j-N] = A[i][j]/f;
    }

    return true;
}

} //end namespace detail

//! Reformatting function (not gpu accelerated)
__host__ __device__
inline void Vec3ToMat4(const Vector3f& vec_in, Matrix4f& mat_out)
{
  mat_out.a14 = vec_in.x;
  mat_out.a24 = vec_in.y;
  mat_out.a34 = vec_in.z;
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

/*!
 * \brief Mat4ToRPY Get the matrix represented as RPY angles
 * \param mat_in Rotation matrix
 * \param rpy_out rpy with roll around X axis, pitch around Y axis and Yaw around Z axis
 * \param solution_number Decides if 1st or 2nd solution is returned
 */
__host__ __device__
inline void Mat4ToRPY(const Matrix4f& mat_in, Vector3f& rpy_out,
                      unsigned int solution_number = 1)
{
  Vector3f out1; //first solution
  Vector3f out2; //second solution
  //get the pointer to the raw data

  // Check that pitch is not at a singularity
  if (1.0 - fabs(mat_in.a31) < 0.00001)
  {
    out1.z = 0;
    out2.z = 0;

    // From difference of angles formula
    if (mat_in.a31 < 0)  //gimbal locked down
    {
      float delta = atan2(mat_in.a12,mat_in.a13);
      out1.y = M_PI_2;
      out2.y = M_PI_2;
      out1.x = delta;
      out2.x = delta;
    }
    else // gimbal locked up
    {
      float delta = atan2(-mat_in.a12,-mat_in.a13);
      out1.y = -M_PI_2;
      out2.y = -M_PI_2;
      out1.x = delta;
      out2.x = delta;
    }
  }
  else
  {
    out1.y = - asin(mat_in.a31);
    out2.y = M_PI - out1.y;

    out1.x = atan2(mat_in.a32/cos(out1.y),
      mat_in.a33/cos(out1.y));
    out2.x = atan2(mat_in.a32/cos(out2.y),
      mat_in.a33/cos(out2.y));

    out1.z = atan2(mat_in.a21/cos(out1.y),
      mat_in.a11/cos(out1.y));
    out2.z = atan2(mat_in.a21/cos(out2.y),
      mat_in.a11/cos(out2.y));
  }

  rpy_out = (solution_number == 1) ? out1 : out2;

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
  m.a23 = -sin(angle);
  m.a24 = 0;
  m.a31 = 0;
  m.a32 = sin(angle);
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
  m.a12 = -sin(angle);
  m.a13 = 0;
  m.a14 = 0;
  m.a21 = sin(angle);
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
 * \param _roll
 * \param _pitch
 * \param _yaw
 * \return
 */
__host__ __device__
inline gpu_voxels::Matrix4f rotateYPR(float _roll, float _pitch, float _yaw)
{
  return roll(_roll) * pitch(_pitch) * yaw(_yaw);
}

/*!
 * \brief rotateRPY Constructs a rotation matrix
 * \param _roll
 * \param _pitch
 * \param _yaw
 * \return
 */
__host__ __device__
inline gpu_voxels::Matrix4f rotateRPY(float _roll, float _pitch, float _yaw)
{
  return yaw(_yaw) * pitch(_pitch) * roll(_roll);
}

/*!
 * \brief rotateYPR Constructs a rotation matrix
 * \param rpy Vector of Roll Pitch and Yaw
 * \return
 */
__host__ __device__
inline gpu_voxels::Matrix4f rotateYPR(gpu_voxels::Vector3f rpy)
{
  return roll(rpy.x)* pitch(rpy.y) * yaw(rpy.z) ;
}

/*!
 * \brief rotateRPY Constructs a rotation matrix by first rotating around Roll, then around Pitch and finaly around Yaw.
 * This acts in the same way as ROS TF Quaternion.setRPY().
 * \param rpy Vector of Roll Pitch and Yaw
 * \return Matrix where rotation is set.
 */
__host__ __device__
inline gpu_voxels::Matrix4f rotateRPY(gpu_voxels::Vector3f rpy)
{
  return yaw(rpy.z) * pitch(rpy.y) * roll(rpy.x);
}


/*!
 * \brief invertMatrix Gets the inverse of a matrix via Gauß-Jordan-Algorithm
 * \param input_matrix
 * \param inverse_matrix
 * \return true if successful
 */
__host__ __device__
inline bool invertMatrix(const gpu_voxels::Matrix4f& input_matrix, gpu_voxels::Matrix4f& inverse_matrix)
{
  double matrix[4][4];
  double inverse[4][4];

  matrix[0][0] = input_matrix.a11;
  matrix[0][1] = input_matrix.a12;
  matrix[0][2] = input_matrix.a13;
  matrix[0][3] = input_matrix.a14;

  matrix[1][0] = input_matrix.a21;
  matrix[1][1] = input_matrix.a22;
  matrix[1][2] = input_matrix.a23;
  matrix[1][3] = input_matrix.a24;

  matrix[2][0] = input_matrix.a31;
  matrix[2][1] = input_matrix.a32;
  matrix[2][2] = input_matrix.a33;
  matrix[2][3] = input_matrix.a34;

  matrix[3][0] = input_matrix.a41;
  matrix[3][1] = input_matrix.a42;
  matrix[3][2] = input_matrix.a43;
  matrix[3][3] = input_matrix.a44;

  if(!detail::invertMatrix(matrix,inverse))
  {
    return false;
  }

  inverse_matrix.a11 = inverse[0][0];
  inverse_matrix.a12 = inverse[0][1];
  inverse_matrix.a13 = inverse[0][2];
  inverse_matrix.a14 = inverse[0][3];

  inverse_matrix.a21 = inverse[1][0];
  inverse_matrix.a22 = inverse[1][1];
  inverse_matrix.a23 = inverse[1][2];
  inverse_matrix.a24 = inverse[1][3];

  inverse_matrix.a31 = inverse[2][0];
  inverse_matrix.a32 = inverse[2][1];
  inverse_matrix.a33 = inverse[2][2];
  inverse_matrix.a34 = inverse[2][3];

  inverse_matrix.a41 = inverse[3][0];
  inverse_matrix.a42 = inverse[3][1];
  inverse_matrix.a43 = inverse[3][2];
  inverse_matrix.a44 = inverse[3][3];

  return true;
}

/*!
 * \brief angleBetween calculates the angle between two vectors
 * \param a First vector
 * \param b Second vector
 * \return The angle between a and b in rad
 */
inline float angleBetween(const Vector3f &a, const Vector3f &b)
{
  float rad = a.normalized().dot(b.normalized());
  if (rad < -1.0)
    rad = -1.0;
  else if (rad >  1.0)
    rad = 1.0;
  return acos(rad);
}


inline Vector3f orientationMatrixDiff(const gpu_voxels::Matrix4f &a, const gpu_voxels::Matrix4f &b)
{
   Vector3f d, tmp1, tmp2;

   tmp1 = Vector3f(a.a11, a.a21, a.a31);
   tmp2 = Vector3f(b.a11, b.a21, b.a31);
   d    = tmp1.cross(tmp2);

   tmp1 = Vector3f(a.a12, a.a22, a.a32);
   tmp2 = Vector3f(b.a12, b.a22, b.a32);
   d    = d + tmp1.cross(tmp2);

   tmp1 = Vector3f(a.a13, a.a23, a.a33);
   tmp2 = Vector3f(b.a13, b.a23, b.a33);
   d    = d + tmp1.cross(tmp2);

   return Vector3f(asin(d.x / 2.0), asin(d.y / 2.0), asin(d.z / 2.0));
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
