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
 * \date    2012-06-22
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_CUDA_DATATYPES_H_INCLUDED
#define GPU_VOXELS_CUDA_DATATYPES_H_INCLUDED

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <stdint.h> // for fixed size datatypes
#include <vector>
#include <map>
#include <string>

namespace gpu_voxels {

/*
 * As CUDA does not support STL / Eigen data types or similar
 * here are some structs for more comfortable use
 */

/*! ---------------- Vectors ---------------- */
struct Vector3i
{
  __device__ __host__ Vector3i()
  {
    x = y = z = 0;
  }
  __device__ __host__ Vector3i(int32_t _x, int32_t _y, int32_t _z)
  {
    x = _x;
    y = _y;
    z = _z;
  }

  int32_t x;
  int32_t y;
  int32_t z;

  __device__ __host__
  inline Vector3i& operator+=(const Vector3i& b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }

  __device__ __host__
  inline Vector3i& operator-=(const Vector3i& b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }

  __device__ __host__
  inline bool operator==(const Vector3i& b) const
  {
    return (x == b.x && y == b.y && z == b.z);
  }

  __device__ __host__
  inline bool operator!=(const Vector3i& b) const
  {
    return (x != b.x || y != b.y || z != b.z);
  }
};

struct Vector3ui
{
  __device__ __host__ Vector3ui()
  {
    x = y = z = 0;
  }
  __device__ __host__ Vector3ui(uint32_t _x)
  {
    x = y = z = _x;
  }
  __device__ __host__ Vector3ui(uint32_t _x, uint32_t _y, uint32_t _z)
  {
    x = _x;
    y = _y;
    z = _z;
  }

  uint32_t x;
  uint32_t y;
  uint32_t z;

  __device__ __host__
  inline Vector3ui& operator+=(const Vector3ui& b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }

  __device__ __host__
  inline Vector3ui& operator-=(const Vector3ui& b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }

  __device__ __host__
  inline bool operator==(const Vector3ui& b) const
  {
    return (x == b.x && y == b.y && z == b.z);
  }

  __device__ __host__
  inline bool operator!=(const Vector3ui& b) const
  {
    return (x != b.x || y != b.y || z != b.z);
  }
};

struct Vector3f
{
  __device__ __host__ Vector3f()
  {
    x = y = z = 0;
  }
  __device__ __host__ Vector3f(float _x)
  {
    x = y = z = _x;
  }
  __device__ __host__ Vector3f(float _x, float _y, float _z)
  {
    x = _x;
    y = _y;
    z = _z;
  }

  float x;
  float y;
  float z;

  __host__
     friend std::ostream& operator<<(std::ostream& out, const Vector3f& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << std::endl;
    return out;
  }

  __device__ __host__
  inline Vector3f& operator+=(const Vector3f& b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }

  __device__ __host__
  inline Vector3f& operator-=(const Vector3f& b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }
};

struct Vector3d
{
  __device__ __host__ Vector3d()
  {
    x = y = z = 0;
  }
  __device__ __host__ Vector3d(double _x, double _y, double _z)
  {
    x = _x;
    y = _y;
    z = _z;
  }

  double x;
  double y;
  double z;

  __device__ __host__
  inline Vector3d& operator+=(const Vector3d& b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }

  __device__ __host__
  inline Vector3d& operator-=(const Vector3d& b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }
};

struct Vector4i
{
  __device__ __host__ Vector4i()
  {
    x = y = z = w = 0;
  }
  __device__ __host__ Vector4i(int32_t _x, int32_t _y, int32_t _z, int32_t _w)
  {
    x = _x;
    y = _y;
    z = _z;
    w = _w;
  }

  int32_t x;
  int32_t y;
  int32_t z;
  int32_t w;
};

struct Vector4f
{
  __device__ __host__ Vector4f()
  {
    x = y = z = w = 0;
  }
  __device__ __host__ Vector4f(float _x, float _y, float _z, float _w)
  {
    x = _x;
    y = _y;
    z = _z;
    w = _w;
  }

  float x;
  float y;
  float z;
  float w;
};

struct Vector4d
{
  __device__ __host__ Vector4d()
  {
    x = y = z = w = 0;
  }
  __device__ __host__ Vector4d(double _x, double _y, double _z, double _w)
  {
    x = _x;
    y = _y;
    z = _z;
    w = _w;
  }

  double x;
  double y;
  double z;
  double w;
};

// *****************  Square Matrices ********************* //
struct Matrix3f
{
  float a11;  float a12;  float a13;
  float a21;  float a22;  float a23;
  float a31;  float a32;  float a33;
};

struct Matrix3d
{
  double a11;  double a12;  double a13;
  double a21;  double a22;  double a23;
  double a31;  double a32;  double a33;
};

struct Matrix4f
{
  __device__ __host__ Matrix4f()
  {
    a11 = 0.0f;
    a12 = 0.0f;
    a13 = 0.0f;
    a14 = 0.0f;
    a21 = 0.0f;
    a22 = 0.0f;
    a23 = 0.0f;
    a24 = 0.0f;
    a31 = 0.0f;
    a32 = 0.0f;
    a33 = 0.0f;
    a34 = 0.0f;
    a41 = 0.0f;
    a42 = 0.0f;
    a43 = 0.0f;
    a44 = 0.0f;
  }
  __device__ __host__ Matrix4f(float _a11, float _a12, float _a13, float _a14, float _a21, float _a22,
                               float _a23, float _a24, float _a31, float _a32, float _a33, float _a34,
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

  float a11;  float a12;  float a13;  float a14;
  float a21;  float a22;  float a23;  float a24;
  float a31;  float a32;  float a33;  float a34;
  float a41;  float a42;  float a43;  float a44;

  __device__ __host__
  void setIdentity()
  {
    a11 = 1;    a12 = 0;    a13 = 0;    a14 = 0;
    a21 = 0;    a22 = 1;    a23 = 0;    a24 = 0;
    a31 = 0;    a32 = 0;    a33 = 1;    a34 = 0;
    a41 = 0;    a42 = 0;    a43 = 0;    a44 = 1;
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

};

struct Matrix4d
{
  __device__ __host__ Matrix4d()
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
};

// *********** Some operations on data types above ********//
__device__ __host__
   inline Vector3i operator+(const Vector3i& a, const Vector3i& b)
{
  Vector3i result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

__device__ __host__
   inline Vector3ui operator+(const Vector3ui& a, const Vector3ui& b)
{
  Vector3ui result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

__device__ __host__
   inline Vector3ui operator-(const Vector3ui& a, const Vector3ui& b)
{
  Vector3ui result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;
  return result;
}

__device__ __host__
   inline Vector3ui operator%(const Vector3ui& a, const Vector3ui& b)
{
  Vector3ui result;
  result.x = a.x % b.x;
  result.y = a.y % b.y;
  result.z = a.z % b.z;
  return result;
}

__device__ __host__
   inline Vector3ui operator/(const Vector3ui& a, const Vector3ui& b)
{
  Vector3ui result;
  result.x = a.x / b.x;
  result.y = a.y / b.y;
  result.z = a.z / b.z;
  return result;
}

__device__ __host__
   inline Vector3ui operator*(const Vector3ui& a, const Vector3ui& b)
{
  Vector3ui result;
  result.x = a.x * b.x;
  result.y = a.y * b.y;
  result.z = a.z * b.z;
  return result;
}

__device__ __host__
   inline Vector3f operator*(const float& a, const Vector3ui& b)
{
  Vector3f result;
  result.x = a * b.x;
  result.y = a * b.y;
  result.z = a * b.z;
  return result;
}

__device__ __host__
   inline Vector3f operator*(const Vector3ui& b, const float& a)
{
  Vector3f result;
  result.x = a * b.x;
  result.y = a * b.y;
  result.z = a * b.z;
  return result;
}

__device__ __host__
   inline Vector3ui minVec(const Vector3ui& a, const Vector3ui& b)
{
  Vector3ui result;
  result.x = a.x < b.x ? a.x : b.x;
  result.y = a.y < b.y ? a.y : b.y;
  result.z = a.z < b.z ? a.z : b.z;
  return result;
}

__device__ __host__
   inline Vector3ui maxVec(const Vector3ui& a, const Vector3ui& b)
{
  Vector3ui result;
  result.x = a.x < b.x ? b.x : a.x;
  result.y = a.y < b.y ? b.y : a.y;
  result.z = a.z < b.z ? b.z : a.z;
  return result;
}

__device__ __host__
   inline Vector3f minVec(const Vector3f& a, const Vector3f& b)
{
  Vector3f result;
  result.x = a.x < b.x ? a.x : b.x;
  result.y = a.y < b.y ? a.y : b.y;
  result.z = a.z < b.z ? a.z : b.z;
  return result;
}

__device__ __host__
   inline Vector3f maxVec(const Vector3f& a, const Vector3f& b)
{
  Vector3f result;
  result.x = a.x < b.x ? b.x : a.x;
  result.y = a.y < b.y ? b.y : a.y;
  result.z = a.z < b.z ? b.z : a.z;
  return result;
}


__device__ __host__
   inline Vector3f operator+(const Vector3f& a, const Vector3f& b)
{
  Vector3f result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

__device__ __host__
   inline Vector3d operator+(const Vector3d& a, const Vector3d& b)
{
  Vector3d result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

__device__ __host__
   inline Vector4i operator+(const Vector4i& a, const Vector4i& b)
{
  Vector4i result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  result.w = a.w + b.w;
  return result;
}

__device__ __host__
   inline Vector4f operator+(const Vector4f& a, const Vector4f& b)
{
  Vector4f result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  result.w = a.w + b.w;
  return result;
}

__device__ __host__
   inline Vector4d operator+(const Vector4d& a, const Vector4d& b)
{
  Vector4d result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  result.w = a.w + b.w;
  return result;
}

__device__ __host__
   inline Vector3f operator*(const Matrix4f& m, const Vector3f& v)
{
  Vector3f result;
  result.x = m.a11 * v.x + m.a12 * v.y + m.a13 * v.z + m.a14;
  result.y = m.a21 * v.x + m.a22 * v.y + m.a23 * v.z + m.a24;
  result.z = m.a31 * v.x + m.a32 * v.y + m.a33 * v.z + m.a34;
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
__forceinline__ bool operator<=(const Vector3ui& a, const Vector3ui& b)
{
  return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

__device__ __host__
__forceinline__ bool operator>=(const Vector3ui& a, const Vector3ui& b)
{
  return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}

__device__ __host__
__forceinline__ bool operator<(const Vector3ui& a, const Vector3ui& b)
{
  return a.x < b.x && a.y < b.y && a.z < b.z;
}

__device__ __host__
__forceinline__ bool operator>(const Vector3ui& a, const Vector3ui& b)
{
  return a.x > b.x && a.y > b.y && a.z > b.z;
}

__device__ __host__
   __forceinline__ Vector3ui operator>>(const Vector3ui& a, const uint32_t shift)
{
  return Vector3ui(a.x >> shift, a.y >> shift, a.z >> shift);
}

__device__ __host__
   __forceinline__ Vector3ui operator<<(const Vector3ui& a, const uint32_t shift)
{
  return Vector3ui(a.x << shift, a.y << shift, a.z << shift);
}

__device__ __host__
   __forceinline__ Vector3ui operator&(const Vector3ui& a, const uint32_t value)
{
  return Vector3ui(a.x & value, a.y & value, a.z & value);
}

__device__ __host__
   inline Vector3f operator*(const Vector3f& a, const float b)
{
  return Vector3f(a.x * b, a.y * b, a.z * b);
}

__device__ __host__
   inline Vector3f operator/(const Vector3f& a, const float b)
{
  return Vector3f(a.x / b, a.y / b, a.z / b);
}

__device__ __host__
   inline Vector3f operator-(const Vector3f& a, const Vector3f& b)
{
  return Vector3f(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__
   inline bool operator==(const Vector3ui& a, const Vector3ui& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

// ##################################################################

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

__device__ __host__
inline void applyTransform(const Matrix4f& m, const Vector3f& v, Vector3f& result)
{
  result.x = m.a11 * v.x + m.a12 * v.y + m.a13 * v.z + m.a14;
  result.y = m.a21 * v.x + m.a22 * v.y + m.a23 * v.z + m.a24;
  result.z = m.a31 * v.x + m.a32 * v.y + m.a33 * v.z + m.a34;
}

/*************** Sensor ********************/

struct Sensor
{
  __host__ __device__ Sensor()
  {
  }

  __host__ __device__ Sensor(Vector3f _position, Matrix3f _orientation, uint32_t _data_width,
                             uint32_t _data_height) :
      position(_position), orientation(_orientation), data_width(_data_width), data_height(_data_height), data_size(
          _data_width * _data_height)
  {
  }

  __host__ __device__ Sensor(const Sensor& other) :
      position(other.position), orientation(other.orientation), data_width(other.data_width), data_height(
          other.data_height), data_size(other.data_size)
  {
  }

  Vector3f position;
  Matrix3f orientation;
  uint32_t data_width;
  uint32_t data_height;
  uint32_t data_size;
};


struct MetaPointCloudStruct
{
  uint16_t num_clouds;
  uint32_t accumulated_cloud_size;
  uint32_t *cloud_sizes;
  Vector3f** clouds_base_addresses;

  __device__ __host__
  MetaPointCloudStruct()
    : num_clouds(0),
      cloud_sizes(0),
      clouds_base_addresses(0)
    {
    }
};

typedef std::map<std::string, float > JointValueMap;

} // end of namespace
#endif
