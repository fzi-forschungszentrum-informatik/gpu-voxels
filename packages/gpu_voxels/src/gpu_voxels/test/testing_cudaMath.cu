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
 * \author  Matthias Wagner <mwagner@fzi.de>
 * \date    2014-06-13
 *
 */
//----------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include "gpu_voxels/helpers/cuda_datatypes.h"
#include <gpu_voxels/test/testing_fixtures.hpp>

using namespace gpu_voxels;


BOOST_FIXTURE_TEST_SUITE(cudaMath, ArgsFixture)

const Matrix4f matrix(0.9751700, -0.218711, -0.0347626, 10, /**/
                      0.1976770, 0.930432, -0.3085770, 11,/**/
                      0.0998334, 0.294044, 0.9505640, 12,/**/
                      0, 0, 0, 1);/**/


BOOST_AUTO_TEST_CASE(matrix_equality)
{
  PERF_MON_START("matrix_equality");
  for(int i = 0; i < iterationCount; i++)
  {
    BOOST_CHECK(matrix == matrix);
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("matrix_equality", "matrix_equality", "cudaMath");
  }
}

BOOST_AUTO_TEST_CASE(matrix_inequality)
{
  PERF_MON_START("matrix_inequality");
  for(int i = 0; i < iterationCount; i++)
  {
    Matrix4f a = matrix;
    Matrix4f b = matrix;
    b.a13 += 0.0001;
    BOOST_CHECK(!(a == b));
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("matrix_inequality", "matrix_inequality", "cudaMath");
  }
}

BOOST_AUTO_TEST_CASE(matrix_apprx_equality)
{
  PERF_MON_START("matrix_apprx_equality");
  for(int i = 0; i < iterationCount; i++)
  {
    Matrix4f a = matrix;
    Matrix4f b = matrix;
    b.a22 -= 0.00001;
    BOOST_CHECK(a.apprx_equal(b, 0.000011));

    a = matrix;
    b = matrix;
    b.a22 += 0.00001;
    BOOST_CHECK(a.apprx_equal(b, 0.000011));
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("matrix_apprx_equality", "matrix_apprx_equality", "cudaMath");
  }
}

BOOST_AUTO_TEST_CASE(matrix_apprx_inequality)
{
  PERF_MON_START("matrix_apprx_inequality");
  for(int i = 0; i < iterationCount; i++)
  {
    Matrix4f a = matrix;
    Matrix4f b = matrix;
    b.a22 += 0.000013;
    BOOST_CHECK(!a.apprx_equal(b, 0.000011));
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("matrix_apprx_inequality", "matrix_apprx_inequality", "cudaMath");
  }
}

BOOST_AUTO_TEST_CASE(matrix_transpose)
{
  PERF_MON_START("matrix_transpose");
  for(int i = 0; i < iterationCount; i++)
  {
    Matrix4f result = matrix.transpose();
    Matrix4f correct_result = Matrix4f(0.9751700, 0.1976770, 0.0998334, 0, /**/
                                       -0.218711, 0.930432, 0.294044, 0,/**/
                                       -0.0347626, -0.3085770, 0.9505640, 0,/**/
                                       10, 11, 12, 1);/**/

    BOOST_CHECK(result == correct_result);
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("matrix_transpose", "matrix_transpose", "cudaMath");
  }
}
BOOST_AUTO_TEST_CASE(matrix_multiply)
{
  PERF_MON_START("matrix_multiply");
  for(int i = 0; i < iterationCount; i++)
  {
    Matrix4f result;
    Matrix4f a = Matrix4f(1, 2, 3, 4, 4, 3, 2, 1, 2, 1, 3, 4, 2, 3, 1, 4);
    Matrix4f b = Matrix4f(7, 4, 5, 6, 6, 5, 7, 4, 4, 6, 5, 7, 4, 5, 6, 7);
    Matrix4f r = Matrix4f(47, 52, 58, 63, 58, 48, 57, 57, 48, 51, 56, 65, 52, 49, 60, 59);

    result = a * b;

    BOOST_CHECK(result == r);
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("matrix_multiply", "matrix_multiply", "cudaMath");
  }
}

BOOST_AUTO_TEST_CASE(matrix_inverse)
{
  PERF_MON_START("matrix_inverse");
  for(int i = 0; i < iterationCount; i++)
  {
    Matrix4f inverse;
    if(matrix.invertMatrix(inverse))
    {
      Matrix4f result = matrix * inverse;
      Matrix4f identity = Matrix4f::createIdentity();

      BOOST_CHECK(identity.apprx_equal(result, 0.00000011));
    }else{
      BOOST_CHECK_MESSAGE(false, "Error in matrix inversion.");
    }
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("matrix_inverse", "matrix_inverse", "cudaMath");
  }
}



BOOST_AUTO_TEST_CASE(matrix_rpy)
{
  PERF_MON_START("matrix_rpy");
  for(int i = 0; i < iterationCount; i++)
  {
    gpu_voxels::Vector3f rot_a; // Input given in RPY
    gpu_voxels::Vector3f rot_b; // Generated. Given in RPY
    gpu_voxels::Vector3f rot_c; // Generated. Given in RPY

    Matrix3f a; // holds rot_a in form of a matrix
    Matrix3f b; // holds rot_b in form of a matrix
    Matrix3f c; // holds rot_c in form of a matrix


    boost::mt19937 rng;
    boost::uniform_real<> plus_minus_two_pi(-2.0 * M_PI, 2.0 * M_PI);
    boost::variate_generator< boost::mt19937, boost::uniform_real<> > gen(rng, plus_minus_two_pi);

    // Calculate two RPY representations (rot_b, rot_c) from the matrix a, then convert them into matrices (b, c).
    // Derive the rotation difference betwee a & b and a & c and see, if they are small enough (< 0.005 due to float rounding errors).
    for(size_t n = 0; n < 1000; n++)
    {
      rot_a = gpu_voxels::Vector3f(gen(), gen(), gen());
      a = Matrix3f::createFromRPY(rot_a);

      rot_b = a.toRPY(0);
      rot_c = a.toRPY(1);

      b = Matrix3f::createFromRPY(rot_b);
      c = Matrix3f::createFromRPY(rot_c);

      if(!(a.orientationMatrixDiff(b).apprx_equal(gpu_voxels::Vector3f(0.0), 0.005)))
        if(!(a.orientationMatrixDiff(b).apprx_equal(gpu_voxels::Vector3f(0.0), 0.005)))
        {
          std::cout << "a = " << a << " b = " << b << " Diff = " << a.orientationMatrixDiff(b) << std::endl;
        }
      if(!(a.orientationMatrixDiff(c).apprx_equal(gpu_voxels::Vector3f(0.0), 0.005)))
      {
        std::cout << "a = " << a << " c = " << c << " Diff = " << a.orientationMatrixDiff(c) << std::endl;
      }


      BOOST_CHECK_MESSAGE(a.orientationMatrixDiff(b).apprx_equal(gpu_voxels::Vector3f(0.0), 0.005), "Difference between input and first reconstructed RPY is zero.");
      BOOST_CHECK_MESSAGE(a.orientationMatrixDiff(c).apprx_equal(gpu_voxels::Vector3f(0.0), 0.005), "Difference between input and second reconstructed RPY is zero.");
    }
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("matrix_rpy", "matrix_rpy", "cudaMath");
  }
}

BOOST_AUTO_TEST_SUITE_END()

