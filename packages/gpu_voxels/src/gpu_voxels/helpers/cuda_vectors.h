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
#ifndef GPU_VOXELS_CUDA_VECTORS_H_INCLUDED
#define GPU_VOXELS_CUDA_VECTORS_H_INCLUDED

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include "icl_core_logging/ThreadStream.h"

namespace gpu_voxels {

/*
 * As CUDA does not support STL / Eigen data types or similar
 * here are some vector structs for more comfortable use
 */


struct Vector3ui
{
  __device__ __host__ Vector3ui() : x(), y(), z() {} // problematic in device shared memory arrays

  __device__ __host__ explicit Vector3ui(uint32_t _x) : x(_x), y(_x), z(_x) {}

  __device__ __host__ Vector3ui(uint32_t _x, uint32_t _y, uint32_t _z) : x(_x), y(_y), z(_z) {}

  __device__ __host__ Vector3ui(uint3 t) : x(t.x), y(t.y), z(t.z) {}

  __device__ __host__
  operator uint3() const { uint3 t; t.x = x; t.y = y; t.z = z; return t; }

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

  __host__
  friend std::ostream& operator<<(std::ostream& out, const Vector3ui& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << std::endl;
    return out;
  }

  __host__
  friend icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Vector3ui& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << icl_core::logging::endl;
    return out;
  }
};


struct Vector3i
{
  __device__ __host__ Vector3i() : x(), y(), z() {} // problematic in device shared memory arrays

  __device__ __host__ explicit Vector3i(int32_t _x) : x(_x), y(_x), z(_x) {}

  __device__ __host__ Vector3i(int32_t _x, int32_t _y, int32_t _z) : x(_x), y(_y), z(_z) {}

  int32_t x;
  int32_t y;
  int32_t z;

  __device__ __host__ explicit Vector3i(const Vector3ui& _v)
  {
    x = _v.x;
    y = _v.y;
    z = _v.z;
  }

  __device__ __host__
  inline Vector3i operator-(const Vector3i& b)
  {
    return Vector3i(x - b.x, y - b.y, z - b.z);
  }

  __device__ __host__
  inline Vector3i operator*(const Vector3i& b)
  {
    return Vector3i(x * b.x, y * b.y, z * b.z);
  }

  __device__ __host__
  inline Vector3i& operator=(const Vector3ui& _v)
  {
    x = _v.x;
    y = _v.y;
    z = _v.z;
    return *this;
  }


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

  __host__
  friend std::ostream& operator<<(std::ostream& out, const Vector3i& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << std::endl;
    return out;
  }

  __host__
  friend icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Vector3i& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << icl_core::logging::endl;
    return out;
  }
};


struct Vector3f
{
  __device__ __host__ Vector3f() : x(), y(), z() {} //problematic in device shared memory arrays

  __device__ __host__ explicit Vector3f(float _x) : x(_x), y(_x), z(_x) {}

  __device__ __host__ Vector3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

  float x;
  float y;
  float z;

  __device__ __host__
  inline Vector3f operator+(const Vector3f& rhs)
  {
    return Vector3f(x + rhs.x, y + rhs.y, z + rhs.z);
  }

  __device__ __host__
  inline Vector3f operator-(const Vector3f& rhs)
  {
    return Vector3f(x - rhs.x, y - rhs.y, z - rhs.z);
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

  __device__ __host__
  inline Vector3f& operator*=(const float& b)
  {
    x *= b;
    y *= b;
    z *= b;
    return *this;
  }

  __device__ __host__
  inline Vector3f& operator/=(const float& b)
  {
    x /= b;
    y /= b;
    z /= b;
    return *this;
  }

  __device__ __host__
  inline bool operator==(const Vector3f& b) const
  {
    return (x == b.x && y == b.y && z == b.z);
  }

  __device__ __host__
  inline bool operator!=(const Vector3f& b) const
  {
    return (x != b.x || y != b.y || z != b.z);
  }

  __device__ __host__
  inline static bool compVec(const Vector3f& i, const Vector3f& j)
  {
    return i.x < j.x || (i.x == j.x && i.y < j.y) || (i.x == j.x && i.y == j.y && i.z < j.z);
  }

  __device__ __host__
  inline static bool eqlVec(const Vector3f& i, const Vector3f& j)
  {
    return (i.x == j.x && i.y == j.y && i.z == j.z);
  }

  __device__ __host__
  inline bool apprx_equal(const Vector3f& b, double epsilon) const
  {
    return (
    (fabs(x - b.x) < epsilon) &&
    (fabs(y - b.y) < epsilon) &&
    (fabs(z - b.z) < epsilon));
  }


  __device__ __host__
  inline float length() const
  {
    return sqrt((x * x) + (y * y) + (z * z));
  }


  /*!
   * \brief normalize normalizes this vector to unit length
   */
  __device__ __host__
  inline void normalize()
  {
    x = x / length();
    y = y / length();
    z = z / length();
  }

  __device__ __host__
  inline Vector3f abs() const
  {
    return Vector3f(fabs(x), fabs(y), fabs(z));
  }

  /*!
   * \brief normalized does not affect the vector it is called on but returns a normalized copy of it
   * which has unit length
   * \return normalized copy of this vector
   */
  __device__ __host__
  inline Vector3f normalized() const
  {
    return Vector3f(x / length(), y / length(), z / length());
  }

  __device__ __host__
  inline float dot(const Vector3f &other) const
  {
    return x * other.x + y * other.y + z * other.z;
  }

  __device__ __host__
  inline Vector3f cross(const Vector3f &other) const
  {
    return Vector3f(y * other.z - z * other.y,
                    z * other.x - x * other.z,
                    x * other.y - y * other.x);
  }

  /*!
   * \brief angleBetween calculates the angle between this and an other vector
   * \param other Second vector
   * \return The angle between this and other in rad
   */
  float angleBetween(const Vector3f &other) const
  {
    float rad = this->normalized().dot(other.normalized());
    if (rad < -1.0)
      rad = -1.0;
    else if (rad >  1.0)
      rad = 1.0;
    return acos(rad);
  }

  __host__
  friend std::ostream& operator<<(std::ostream& out, const Vector3f& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << std::endl;
    return out;
  }

  __host__
  friend icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Vector3f& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << icl_core::logging::endl;
    return out;
  }
};

struct Vector3d
{
  __device__ __host__ Vector3d() : x(), y(), z() {} //problematic in device shared memory arrays

  __device__ __host__ explicit Vector3d(double _x) : x(_x), y(_x), z(_x) {}

  __device__ __host__ Vector3d(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

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

  __device__ __host__
  inline bool operator==(const Vector3d& b) const
  {
    return (x == b.x && y == b.y && z == b.z);
  }

  __host__
  friend std::ostream& operator<<(std::ostream& out, const Vector3d& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << std::endl;
    return out;
  }

  __host__
  friend icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Vector3d& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << icl_core::logging::endl;
    return out;
  }
};

struct Vector4i
{
  __device__ __host__ Vector4i() : x(), y(), z(), w() {} //problematic in device shared memory arrays

  __device__ __host__ Vector4i(int32_t _x, int32_t _y, int32_t _z, int32_t _w) : x(_x), y(_y), z(_z), w(_w) {}

  int32_t x;
  int32_t y;
  int32_t z;
  int32_t w;

  __host__
  friend std::ostream& operator<<(std::ostream& out, const Vector4i& vector)
  {
    out << "(x, y, z, w) = (" << vector.x << ", " << vector.y << ", " << vector.z << ", " << vector.w << ")" << std::endl;
    return out;
  }

  __host__
  friend icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Vector4i& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ")" << icl_core::logging::endl;
    return out;
  }
};

struct Vector4f
{
  __device__ __host__ Vector4f() : x(), y(), z(), w() {} //problematic in device shared memory arrays

  __device__ __host__ Vector4f(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}


  float x;
  float y;
  float z;
  float w;

  __host__
  friend std::ostream& operator<<(std::ostream& out, const Vector4f& vector)
  {
    out << "(x, y, z, w) = (" << vector.x << ", " << vector.y << ", " << vector.z << ", " << vector.w << ")" << std::endl;
    return out;
  }

  __host__
  friend icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Vector4f& vector)
  {
    out << "(x, y, z) = (" << vector.x << ", " << vector.y << ", " << vector.z << ", " << vector.w << ")" << icl_core::logging::endl;
    return out;
  }
};

struct Vector4d
{
  __device__ __host__ Vector4d() : x(), y(), z(), w() {} //problematic in device shared memory arrays

  __device__ __host__ Vector4d(double _x, double _y, double _z, double _w) : x(_x), y(_y), z(_z), w(_w) {}

  double x;
  double y;
  double z;
  double w;
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

// Special case for adding negative offsets to collisions:
__device__ __host__
   inline Vector3ui operator+(const Vector3ui& a, const Vector3i& b)
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
   inline Vector3f operator*(const float& a, const Vector3i& b)
{
  Vector3f result;
  result.x = a * b.x;
  result.y = a * b.y;
  result.z = a * b.z;
  return result;
}

__device__ __host__
   inline Vector3f operator*(const Vector3f& a, const Vector3f& b)
{
  return Vector3f(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__
   inline Vector3f operator*(const Vector3i& b, const float& a)
{
  Vector3f result;
  result.x = a * b.x;
  result.y = a * b.y;
  result.z = a * b.z;
  return result;
}

__device__ __host__
   inline Vector3i operator*(const int32_t& a, const Vector3i& b)
{
  Vector3i result;
  result.x = a * b.x;
  result.y = a * b.y;
  result.z = a * b.z;
  return result;
}

__device__ __host__
   inline Vector3i operator*(const Vector3i& b, const int32_t& a)
{
  Vector3i result;
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
   inline Vector3f operator/(const Vector3i& a, const float b)
{
  return Vector3f(a.x / b, a.y / b, a.z / b);
}

} // end of namespace
#endif
