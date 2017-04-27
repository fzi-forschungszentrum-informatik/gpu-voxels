// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

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
 * \author  Florian Drews
 * \author  Christian Juelg
 * \date    2012-06-22
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_CUDA_MATRICES_H_INCLUDED
#define GPU_VOXELS_CUDA_MATRICES_H_INCLUDED

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdio>

#include "icl_core_logging/ThreadStream.h"

namespace gpu_voxels {

/*
 * As CUDA does not support STL / Eigen data types or similar
 * here are some matrix structs for more comfortable use
 *
 * Adressing within matrices is ROW, COLLUMN
 */


namespace matrix_inversion {

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

} //end namespace matrix_inversion


// *****************  Square Matrices ********************* //
struct Matrix3f
{
  float a11;  float a12;  float a13;
  float a21;  float a22;  float a23;
  float a31;  float a32;  float a33;

  __device__ __host__ Matrix3f() // problematic in device shared memory arrays
  {
    a11 = 0.0f;    a12 = 0.0f;    a13 = 0.0f;
    a21 = 0.0f;    a22 = 0.0f;    a23 = 0.0f;
    a31 = 0.0f;    a32 = 0.0f;    a33 = 0.0f;
  }
  __device__ __host__ Matrix3f(float _a11, float _a12, float _a13,
                               float _a21, float _a22, float _a23,
                               float _a31, float _a32, float _a33)
  {
    a11 = _a11;    a12 = _a12;    a13 = _a13;
    a21 = _a21;    a22 = _a22;    a23 = _a23;
    a31 = _a31;    a32 = _a32;    a33 = _a33;
  }

  __device__ __host__ static Matrix3f createIdentity()
  {
    return Matrix3f(1, 0, 0,
                    0, 1, 0,
                    0, 0, 1);
  }

  __device__ __host__
  inline Matrix3f operator*(const Matrix3f& rhs) const
  {
    Matrix3f result;

    // 1st column
    result.a11 = a11 * rhs.a11 + a12 * rhs.a21 + a13 * rhs.a31;
    result.a21 = a21 * rhs.a11 + a22 * rhs.a21 + a23 * rhs.a31;
    result.a31 = a31 * rhs.a11 + a32 * rhs.a21 + a33 * rhs.a31;

    // 2nd column
    result.a12 = a11 * rhs.a12 + a12 * rhs.a22 + a13 * rhs.a32;
    result.a22 = a21 * rhs.a12 + a22 * rhs.a22 + a23 * rhs.a32;
    result.a32 = a31 * rhs.a12 + a32 * rhs.a22 + a33 * rhs.a32;

    // 3rd column
    result.a13 = a11 * rhs.a13 + a12 * rhs.a23 + a13 * rhs.a33;
    result.a23 = a21 * rhs.a13 + a22 * rhs.a23 + a23 * rhs.a33;
    result.a33 = a31 * rhs.a13 + a32 * rhs.a23 + a33 * rhs.a33;

    return result;
  }

  __host__ __device__
  static Matrix3f createFromRoll(float angle)
  {
    return Matrix3f(1, 0,          0,
                    0, cos(angle), -sin(angle),
                    0, sin(angle), cos(angle));
  }

  __host__ __device__
  static Matrix3f createFromPitch(float angle)
  {
    return Matrix3f(cos(angle),  0, sin(angle),
                    0,           1, 0,
                    -sin(angle), 0, cos(angle));
  }

  __host__ __device__
  static Matrix3f createFromYaw(float angle)
  {
    return Matrix3f(cos(angle), -sin(angle), 0,
                    sin(angle), cos(angle),  0,
                    0,          0,           1);
  }

  /*!
   * \brief createFromYPR Constructs a rotation matrix
   * \param _roll
   * \param _pitch
   * \param _yaw
   * \return
   */
  __host__ __device__
  static Matrix3f createFromYPR(float _roll, float _pitch, float _yaw)
  {
    return createFromRoll(_roll) * createFromPitch(_pitch) * createFromYaw(_yaw);
  }

  /*!
   * \brief createFromRPY Constructs a rotation matrix
   * \param _roll
   * \param _pitch
   * \param _yaw
   * \return
   */
  __host__ __device__
  static Matrix3f createFromRPY(float _roll, float _pitch, float _yaw)
  {
    return createFromYaw(_yaw) * createFromPitch(_pitch) * createFromRoll(_roll);
  }

  /*!
   * \brief createFromYPR Constructs a rotation matrix
   * \param rpy Vector of Roll Pitch and Yaw
   * \return
   */
  __host__ __device__
  static Matrix3f createFromYPR(Vector3f rpy)
  {
    return createFromRoll(rpy.x)* createFromPitch(rpy.y) * createFromYaw(rpy.z) ;
  }

  /*!
   * \brief createFromRPY Constructs a rotation matrix by first rotating around Roll, then around Pitch and finaly around Yaw.
   * This acts in the same way as ROS TF Quaternion.setRPY().
   * \param rpy Vector of Roll Pitch and Yaw
   * \return Matrix where rotation is set.
   */
  __device__ __host__ static Matrix3f createFromRPY(Vector3f rpy)
  {
    return createFromYaw(rpy.z) * createFromPitch(rpy.y) * createFromRoll(rpy.x);
  }

  /*!
   * \brief Mat3ToRPY Get the matrix represented as RPY angles
   * \param solution_number Decides if 1st or 2nd solution is returned
   * \return rpy with roll around X axis, pitch around Y axis and Yaw around Z axis
   */
  __host__ __device__
  Vector3f toRPY(unsigned int solution_number = 1)
  {
    Vector3f out1; //first solution
    Vector3f out2; //second solution

    // Check that pitch is not at a singularity
    if (1.0 - fabs(a31) < 0.00001)
    {
      out1.z = 0;
      out2.z = 0;

      // From difference of angles formula
      if (a31 < 0)  //gimbal locked down
      {
        float delta = atan2(a12, a13);
        out1.y = M_PI_2;
        out2.y = M_PI_2;
        out1.x = delta;
        out2.x = delta;
      }
      else // gimbal locked up
      {
        float delta = atan2(-a12, -a13);
        out1.y = -M_PI_2;
        out2.y = -M_PI_2;
        out1.x = delta;
        out2.x = delta;
      }
    }
    else
    {
      out1.y = - asin(a31);
      out2.y = M_PI - out1.y;

      out1.x = atan2(a32/cos(out1.y), a33/cos(out1.y));
      out2.x = atan2(a32/cos(out2.y), a33/cos(out2.y));

      out1.z = atan2(a21/cos(out1.y), a11/cos(out1.y));
      out2.z = atan2(a21/cos(out2.y), a11/cos(out2.y));
    }

    return (solution_number == 1) ? out1 : out2;
  }

  /*!
   * \brief orientationMatrixDiff calculates the angles between this and the other rotation matrix.
   * \param other Rotation matrix
   * \return The difference angles on all three axis
   */
  __host__ __device__
  Vector3f orientationMatrixDiff(const gpu_voxels::Matrix3f &other) const
  {
     Vector3f d, tmp1, tmp2;

     tmp1 = Vector3f(a11, a21, a31);
     tmp2 = Vector3f(other.a11, other.a21, other.a31);
     d    = tmp1.cross(tmp2);

     tmp1 = Vector3f(a12, a22, a32);
     tmp2 = Vector3f(other.a12, other.a22, other.a32);
     d    = d + tmp1.cross(tmp2);

     tmp1 = Vector3f(a13, a23, a33);
     tmp2 = Vector3f(other.a13, other.a23, other.a33);
     d    = d + tmp1.cross(tmp2);

     return Vector3f(asin(d.x / 2.0), asin(d.y / 2.0), asin(d.z / 2.0));
  }

  /*!
   * \brief invertMatrix Gets the inverse of a matrix via Gauß-Jordan-Algorithm
   * \param inverse_matrix
   * \return true if successful
   */
  __host__ __device__
  inline bool invertMatrix(gpu_voxels::Matrix3f& inverse_matrix)
  {
    double matrix[3][3];
    double inverse[3][3];

    matrix[0][0] = a11;
    matrix[0][1] = a12;
    matrix[0][2] = a13;

    matrix[1][0] = a21;
    matrix[1][1] = a22;
    matrix[1][2] = a23;

    matrix[2][0] = a31;
    matrix[2][1] = a32;
    matrix[2][2] = a33;

    if(!matrix_inversion::invertMatrix(matrix,inverse))
    {
      return false;
    }

    inverse_matrix.a11 = inverse[0][0];
    inverse_matrix.a12 = inverse[0][1];
    inverse_matrix.a13 = inverse[0][2];

    inverse_matrix.a21 = inverse[1][0];
    inverse_matrix.a22 = inverse[1][1];
    inverse_matrix.a23 = inverse[1][2];

    inverse_matrix.a31 = inverse[2][0];
    inverse_matrix.a32 = inverse[2][1];
    inverse_matrix.a33 = inverse[2][2];

    return true;
  }

  __host__
  friend std::ostream& operator<<(std::ostream& out, const Matrix3f& matrix)
  {
    out.precision(3);
    out << "\n" << std::fixed <<
        "[" << std::setw(10) << matrix.a11 << ", " << std::setw(10) << matrix.a12 << ", " << std::setw(10) << matrix.a13 << ",\n"
        " " << std::setw(10) << matrix.a21 << ", " << std::setw(10) << matrix.a22 << ", " << std::setw(10) << matrix.a23 << ",\n"
        " " << std::setw(10) << matrix.a31 << ", " << std::setw(10) << matrix.a32 << ", " << std::setw(10) << matrix.a33 << "]"
        << std::endl;
    return out;
  }

  __host__
  friend icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Matrix3f& matrix)
  {
    out << "\n" <<
        "[" << matrix.a11 << ", " << matrix.a12 << ", " << matrix.a13 << ",\n"
        " " << matrix.a21 << ", " << matrix.a22 << ", " << matrix.a23 << ",\n"
        " " << matrix.a31 << ", " << matrix.a32 << ", " << matrix.a33 << "]"
        << icl_core::logging::endl;
    return out;
  }

};

struct Matrix3d
{
  double a11;  double a12;  double a13;
  double a21;  double a22;  double a23;
  double a31;  double a32;  double a33;
};

struct Matrix4f
{
  float a11;  float a12;  float a13;  float a14;
  float a21;  float a22;  float a23;  float a24;
  float a31;  float a32;  float a33;  float a34;
  float a41;  float a42;  float a43;  float a44;

  __device__ __host__ Matrix4f() // problematic in device shared memory arrays
  {
    a11 = 0.0f;    a12 = 0.0f;    a13 = 0.0f;    a14 = 0.0f;
    a21 = 0.0f;    a22 = 0.0f;    a23 = 0.0f;    a24 = 0.0f;
    a31 = 0.0f;    a32 = 0.0f;    a33 = 0.0f;    a34 = 0.0f;
    a41 = 0.0f;    a42 = 0.0f;    a43 = 0.0f;    a44 = 0.0f;
  }
  __device__ __host__ Matrix4f(float _a11, float _a12, float _a13, float _a14,
                               float _a21, float _a22, float _a23, float _a24,
                               float _a31, float _a32, float _a33, float _a34,
                               float _a41, float _a42, float _a43, float _a44)
  {
    a11 = _a11;    a12 = _a12;    a13 = _a13;    a14 = _a14;
    a21 = _a21;    a22 = _a22;    a23 = _a23;    a24 = _a24;
    a31 = _a31;    a32 = _a32;    a33 = _a33;    a34 = _a34;
    a41 = _a41;    a42 = _a42;    a43 = _a43;    a44 = _a44;
  }


  __device__ __host__ Matrix4f& operator=(const Matrix4f& other)
  {
    a11 = other.a11;    a12 = other.a12;    a13 = other.a13;    a14 = other.a14;
    a21 = other.a21;    a22 = other.a22;    a23 = other.a23;    a24 = other.a24;
    a31 = other.a31;    a32 = other.a32;    a33 = other.a33;    a34 = other.a34;
    a41 = other.a41;    a42 = other.a42;    a43 = other.a43;    a44 = other.a44;
    return *this;
  }

  __device__ __host__ static Matrix4f createIdentity()
  {
    return Matrix4f(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
  }

  __device__ __host__ static Matrix4f createFromRotationAndTranslation(const Matrix3f& rot, const Vector3f& trans)
  {
    return Matrix4f(rot.a11,  rot.a12,  rot.a13,  trans.x,
                    rot.a21,  rot.a22,  rot.a23,  trans.y,
                    rot.a31,  rot.a32,  rot.a33,  trans.z,
                    0,        0,        0,        1);
  }

  __device__ __host__ Matrix3f getRotation() const
  {
    return Matrix3f(a11,  a12,  a13,
                    a21,  a22,  a23,
                    a31,  a32,  a33);
  }

  __device__ __host__ Vector3f getTranslation() const
  {
    return Vector3f(a14, a24, a34);
  }

  __device__ __host__ void setRotation(const Matrix3f& rot)
  {
    a11 = rot.a11;  a12 = rot.a12;  a13 = rot.a13;
    a21 = rot.a21;  a22 = rot.a22;  a23 = rot.a23;
    a31 = rot.a31;  a32 = rot.a32;  a33 = rot.a33;
  }

  __device__ __host__ void setTranslation(const Vector3f& trans)
  {
    a14 = trans.x;
    a24 = trans.y;
    a34 = trans.z;
  }

  //! Transpose a matrix
  __host__ __device__
  Matrix4f transpose() const
  {
    return Matrix4f(a11,  a21,  a31, a41,
                    a12,  a22,  a32, a42,
                    a13,  a23,  a33, a43,
                    a14,  a24,  a34, a44);
  }

  __device__ __host__ Matrix4f& leftMultiply(const Matrix4f& other)
  {
    // 1st column of y
    float l_a11 = a11 * other.a11 + a12 * other.a21 + a13 * other.a31 + a14 * other.a41;
    float l_a21 = a21 * other.a11 + a22 * other.a21 + a23 * other.a31 + a24 * other.a41;
    float l_a31 = a31 * other.a11 + a32 * other.a21 + a33 * other.a31 + a34 * other.a41;
    float l_a41 = a41 * other.a11 + a42 * other.a21 + a43 * other.a31 + a44 * other.a41;

    // 2nd column of y
    float l_a12 = a11 * other.a12 + a12 * other.a22 + a13 * other.a32 + a14 * other.a42;
    float l_a22 = a21 * other.a12 + a22 * other.a22 + a23 * other.a32 + a24 * other.a42;
    float l_a32 = a31 * other.a12 + a32 * other.a22 + a33 * other.a32 + a34 * other.a42;
    float l_a42 = a41 * other.a12 + a42 * other.a22 + a43 * other.a32 + a44 * other.a42;

    // 3rd column of y
    float l_a13 = a11 * other.a13 + a12 * other.a23 + a13 * other.a33 + a14 * other.a43;
    float l_a23 = a21 * other.a13 + a22 * other.a23 + a23 * other.a33 + a24 * other.a43;
    float l_a33 = a31 * other.a13 + a32 * other.a23 + a33 * other.a33 + a34 * other.a43;
    float l_a43 = a41 * other.a13 + a42 * other.a23 + a43 * other.a33 + a44 * other.a43;

    // 4th column of y
    float l_a14 = a11 * other.a14 + a12 * other.a24 + a13 * other.a34 + a14 * other.a44;
    float l_a24 = a21 * other.a14 + a22 * other.a24 + a23 * other.a34 + a24 * other.a44;
    float l_a34 = a31 * other.a14 + a32 * other.a24 + a33 * other.a34 + a34 * other.a44;
    float l_a44 = a41 * other.a14 + a42 * other.a24 + a43 * other.a34 + a44 * other.a44;

    a11 = l_a11;    a12 = l_a12;    a13 = l_a13;    a14 = l_a14;
    a21 = l_a21;    a22 = l_a22;    a23 = l_a23;    a24 = l_a24;
    a31 = l_a31;    a32 = l_a32;    a33 = l_a33;    a34 = l_a34;
    a41 = l_a41;    a42 = l_a42;    a43 = l_a43;    a44 = l_a44;

    return *this;
  }

  __device__ __host__
  void setIdentity()
  {
    a11 = 1;    a12 = 0;    a13 = 0;    a14 = 0;
    a21 = 0;    a22 = 1;    a23 = 0;    a24 = 0;
    a31 = 0;    a32 = 0;    a33 = 1;    a34 = 0;
    a41 = 0;    a42 = 0;    a43 = 0;    a44 = 1;
  }

  __device__ __host__
  inline bool operator==(const Matrix4f& b) const
  {
    return a11 == b.a11 && a12 == b.a12 && a13 == b.a13 && a14 == b.a14 && /**/
    a21 == b.a21 && a22 == b.a22 && a23 == b.a23 && a24 == b.a24 && /**/
    a31 == b.a31 && a32 == b.a32 && a33 == b.a33 && a34 == b.a34 && /**/
    a41 == b.a41 && a42 == b.a42 && a43 == b.a43 && a44 == b.a44;/**/
  }

  __device__ __host__
  inline bool apprx_equal(const Matrix4f& b, double epsilon) const
  {
    return (
    (fabs(a11 - b.a11) < epsilon) &&
    (fabs(a12 - b.a12) < epsilon) &&
    (fabs(a13 - b.a13) < epsilon) &&
    (fabs(a14 - b.a14) < epsilon) &&
    (fabs(a21 - b.a21) < epsilon) &&
    (fabs(a22 - b.a22) < epsilon) &&
    (fabs(a23 - b.a23) < epsilon) &&
    (fabs(a24 - b.a24) < epsilon) &&
    (fabs(a31 - b.a31) < epsilon) &&
    (fabs(a32 - b.a32) < epsilon) &&
    (fabs(a33 - b.a33) < epsilon) &&
    (fabs(a34 - b.a34) < epsilon) &&
    (fabs(a41 - b.a41) < epsilon) &&
    (fabs(a42 - b.a42) < epsilon) &&
    (fabs(a43 - b.a43) < epsilon) &&
    (fabs(a44 - b.a44) < epsilon));
  }

  /*!
   * \brief invertMatrix Gets the inverse of a matrix via Gauß-Jordan-Algorithm
   * \param inverse_matrix
   * \return true if successful
   */
  __host__ __device__
  inline bool invertMatrix(gpu_voxels::Matrix4f& inverse_matrix) const
  {
    double matrix[4][4];
    double inverse[4][4];

    matrix[0][0] = a11;
    matrix[0][1] = a12;
    matrix[0][2] = a13;
    matrix[0][3] = a14;

    matrix[1][0] = a21;
    matrix[1][1] = a22;
    matrix[1][2] = a23;
    matrix[1][3] = a24;

    matrix[2][0] = a31;
    matrix[2][1] = a32;
    matrix[2][2] = a33;
    matrix[2][3] = a34;

    matrix[3][0] = a41;
    matrix[3][1] = a42;
    matrix[3][2] = a43;
    matrix[3][3] = a44;

    if(!matrix_inversion::invertMatrix(matrix,inverse))
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

  __device__ __host__
  void print() const
  {
    printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n",   a11, a12, a13, a14);
    printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n",   a21, a22, a23, a24);
    printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n",   a31, a32, a33, a34);
    printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n\n", a41, a42, a43, a44);
  }


  __host__
  friend std::ostream& operator<<(std::ostream& out, const Matrix4f& matrix)
  {
    out.precision(3);
    out << "\n" << std::fixed <<
        "[" << std::setw(10) << matrix.a11 << ", " << std::setw(10) << matrix.a12 << ", " << std::setw(10) << matrix.a13 << ", " << std::setw(10) << matrix.a14 << ",\n"
        " " << std::setw(10) << matrix.a21 << ", " << std::setw(10) << matrix.a22 << ", " << std::setw(10) << matrix.a23 << ", " << std::setw(10) << matrix.a24 << ",\n"
        " " << std::setw(10) << matrix.a31 << ", " << std::setw(10) << matrix.a32 << ", " << std::setw(10) << matrix.a33 << ", " << std::setw(10) << matrix.a34 << ",\n"
        " " << std::setw(10) << matrix.a41 << ", " << std::setw(10) << matrix.a42 << ", " << std::setw(10) << matrix.a43 << ", " << std::setw(10) << matrix.a44 << "]"
        << std::endl;
    return out;
  }

  __host__
  friend icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Matrix4f& matrix)
  {
    out << "\n" <<
        "[" << matrix.a11 << ", " << matrix.a12 << ", " << matrix.a13 << ", " << matrix.a14 << ",\n"
        " " << matrix.a21 << ", " << matrix.a22 << ", " << matrix.a23 << ", " << matrix.a24 << ",\n"
        " " << matrix.a31 << ", " << matrix.a32 << ", " << matrix.a33 << ", " << matrix.a34 << ",\n"
        " " << matrix.a41 << ", " << matrix.a42 << ", " << matrix.a43 << ", " << matrix.a44 << "]"
        << icl_core::logging::endl;
    return out;
  }

};

struct Matrix4d
{
  __device__ __host__ Matrix4d() // problematic in device shared memory arrays
  {
    a11 = 0;    a12 = 0;    a13 = 0;    a14 = 0;
    a21 = 0;    a22 = 0;    a23 = 0;    a24 = 0;
    a31 = 0;    a32 = 0;    a33 = 0;    a34 = 0;
    a41 = 0;    a42 = 0;    a43 = 0;    a44 = 0;
  }

  __device__ __host__ Matrix4d(double _a11, double _a12, double _a13, double _a14,
                               double _a21, double _a22, double _a23, double _a24,
                               double _a31, double _a32, double _a33, double _a34,
                               double _a41, double _a42, double _a43, double _a44)
  {
    a11 = _a11;    a12 = _a12;    a13 = _a13;    a14 = _a14;
    a21 = _a21;    a22 = _a22;    a23 = _a23;    a24 = _a24;
    a31 = _a31;    a32 = _a32;    a33 = _a33;    a34 = _a34;
    a41 = _a41;    a42 = _a42;    a43 = _a43;    a44 = _a44;
  }

  __device__ __host__
  void setIdentity()
  {
    a11 = 1;    a12 = 0;    a13 = 0;    a14 = 0;
    a21 = 0;    a22 = 1;    a23 = 0;    a24 = 0;
    a31 = 0;    a32 = 0;    a33 = 1;    a34 = 0;
    a41 = 0;    a42 = 0;    a43 = 0;    a44 = 1;
  }

  double a11;  double a12;  double a13;  double a14;
  double a21;  double a22;  double a23;  double a24;
  double a31;  double a32;  double a33;  double a34;
  double a41;  double a42;  double a43;  double a44;

  __device__ __host__
  void print() const
  {
    printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n",   a11, a12, a13, a14);
    printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n",   a21, a22, a23, a24);
    printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n",   a31, a32, a33, a34);
    printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n\n", a41, a42, a43, a44);
  }
};

// *********** Some operations on data types above ********//
__device__ __host__
inline Vector3f operator*(const Matrix3f& m, const Vector3f& v)
{
  return Vector3f(m.a11 * v.x + m.a12 * v.y + m.a13 * v.z,
                  m.a21 * v.x + m.a22 * v.y + m.a23 * v.z,
                  m.a31 * v.x + m.a32 * v.y + m.a33 * v.z);
}

__device__ __host__
inline Vector3f operator*(const Matrix4f& m, const Vector3f& v)
{
  return Vector3f(m.a11 * v.x + m.a12 * v.y + m.a13 * v.z + m.a14,
                  m.a21 * v.x + m.a22 * v.y + m.a23 * v.z + m.a24,
                  m.a31 * v.x + m.a32 * v.y + m.a33 * v.z + m.a34);
}

__device__ __host__
   inline Vector3f operator*(const Matrix4f& m, const Vector4f& v)
{
  Vector3f result;
  result.x = m.a11 * v.x + m.a12 * v.y + m.a13 * v.z + m.a14 * v.w;
  result.y = m.a21 * v.x + m.a22 * v.y + m.a23 * v.z + m.a24 * v.w;
  result.z = m.a31 * v.x + m.a32 * v.y + m.a33 * v.z + m.a34 * v.w;
  return result;
}


__device__ __host__
inline Matrix4f operator-(const Matrix4f& a, const Matrix4f& b)
{
  Matrix4f result;

  result.a11 = a.a11 - b.a11; result.a12 = a.a12 - b.a12; result.a13 = a.a13 - b.a13; result.a14 = a.a14 - b.a14;
  result.a21 = a.a21 - b.a21; result.a22 = a.a22 - b.a22; result.a23 = a.a23 - b.a23; result.a24 = a.a24 - b.a24;
  result.a31 = a.a31 - b.a31; result.a32 = a.a32 - b.a32; result.a33 = a.a33 - b.a33; result.a34 = a.a34 - b.a34;
  result.a41 = a.a41 - b.a41; result.a42 = a.a42 - b.a42; result.a43 = a.a43 - b.a43; result.a44 = a.a44 - b.a44;

  return result;
}

__device__ __host__
   inline Matrix4f operator*(const Matrix4f& x, const Matrix4f& y)
{
  Matrix4f result;

  // 1st column of y
  result.a11 = x.a11 * y.a11 + x.a12 * y.a21 + x.a13 * y.a31 + x.a14 * y.a41;
  result.a21 = x.a21 * y.a11 + x.a22 * y.a21 + x.a23 * y.a31 + x.a24 * y.a41;
  result.a31 = x.a31 * y.a11 + x.a32 * y.a21 + x.a33 * y.a31 + x.a34 * y.a41;
  result.a41 = x.a41 * y.a11 + x.a42 * y.a21 + x.a43 * y.a31 + x.a44 * y.a41;

  // 2nd column of y
  result.a12 = x.a11 * y.a12 + x.a12 * y.a22 + x.a13 * y.a32 + x.a14 * y.a42;
  result.a22 = x.a21 * y.a12 + x.a22 * y.a22 + x.a23 * y.a32 + x.a24 * y.a42;
  result.a32 = x.a31 * y.a12 + x.a32 * y.a22 + x.a33 * y.a32 + x.a34 * y.a42;
  result.a42 = x.a41 * y.a12 + x.a42 * y.a22 + x.a43 * y.a32 + x.a44 * y.a42;

  // 3rd column of y
  result.a13 = x.a11 * y.a13 + x.a12 * y.a23 + x.a13 * y.a33 + x.a14 * y.a43;
  result.a23 = x.a21 * y.a13 + x.a22 * y.a23 + x.a23 * y.a33 + x.a24 * y.a43;
  result.a33 = x.a31 * y.a13 + x.a32 * y.a23 + x.a33 * y.a33 + x.a34 * y.a43;
  result.a43 = x.a41 * y.a13 + x.a42 * y.a23 + x.a43 * y.a33 + x.a44 * y.a43;

  // 4th column of y
  result.a14 = x.a11 * y.a14 + x.a12 * y.a24 + x.a13 * y.a34 + x.a14 * y.a44;
  result.a24 = x.a21 * y.a14 + x.a22 * y.a24 + x.a23 * y.a34 + x.a24 * y.a44;
  result.a34 = x.a31 * y.a14 + x.a32 * y.a24 + x.a33 * y.a34 + x.a34 * y.a44;
  result.a44 = x.a41 * y.a14 + x.a42 * y.a24 + x.a43 * y.a34 + x.a44 * y.a44;

  return result;
}


__device__ __host__
   inline Matrix4d operator*(const Matrix4d& x, const Matrix4d& y)
{
  Matrix4d result;

  // 1st column of y
  result.a11 = x.a11 * y.a11 + x.a12 * y.a21 + x.a13 * y.a31 + x.a14 * y.a41;
  result.a21 = x.a21 * y.a11 + x.a22 * y.a21 + x.a23 * y.a31 + x.a24 * y.a41;
  result.a31 = x.a31 * y.a11 + x.a32 * y.a21 + x.a33 * y.a31 + x.a34 * y.a41;
  result.a41 = x.a41 * y.a11 + x.a42 * y.a21 + x.a43 * y.a31 + x.a44 * y.a41;

  // 2nd column of y
  result.a12 = x.a11 * y.a12 + x.a12 * y.a22 + x.a13 * y.a32 + x.a14 * y.a42;
  result.a22 = x.a21 * y.a12 + x.a22 * y.a22 + x.a23 * y.a32 + x.a24 * y.a42;
  result.a32 = x.a31 * y.a12 + x.a32 * y.a22 + x.a33 * y.a32 + x.a34 * y.a42;
  result.a42 = x.a41 * y.a12 + x.a42 * y.a22 + x.a43 * y.a32 + x.a44 * y.a42;

  // 3rd column of y
  result.a13 = x.a11 * y.a13 + x.a12 * y.a23 + x.a13 * y.a33 + x.a14 * y.a43;
  result.a23 = x.a21 * y.a13 + x.a22 * y.a23 + x.a23 * y.a33 + x.a24 * y.a43;
  result.a33 = x.a31 * y.a13 + x.a32 * y.a23 + x.a33 * y.a33 + x.a34 * y.a43;
  result.a43 = x.a41 * y.a13 + x.a42 * y.a23 + x.a43 * y.a33 + x.a44 * y.a43;

  // 4th column of y
  result.a14 = x.a11 * y.a14 + x.a12 * y.a24 + x.a13 * y.a34 + x.a14 * y.a44;
  result.a24 = x.a21 * y.a14 + x.a22 * y.a24 + x.a23 * y.a34 + x.a24 * y.a44;
  result.a34 = x.a31 * y.a14 + x.a32 * y.a24 + x.a33 * y.a34 + x.a34 * y.a44;
  result.a44 = x.a41 * y.a14 + x.a42 * y.a24 + x.a43 * y.a34 + x.a44 * y.a44;

  return result;
}

} // end of namespace
#endif
