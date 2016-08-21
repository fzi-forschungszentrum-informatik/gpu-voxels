// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Matthias Wagner <mwagner@fzi.de>
 * \date    2014-06-13
 *
 */
//----------------------------------------------------------------------
#include <gpu_voxels/helpers/CudaMath.h>

#include <boost/test/unit_test.hpp>
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

using namespace gpu_voxels;

BOOST_AUTO_TEST_SUITE(cudaMath)

const Matrix4f matrix(0.9751700, -0.218711, -0.0347626, 10, /**/
                      0.1976770, 0.930432, -0.3085770, 11,/**/
                      0.0998334, 0.294044, 0.9505640, 12,/**/
                      0, 0, 0, 1);/**/


BOOST_AUTO_TEST_CASE(matrix_equality)
{
  BOOST_CHECK(matrix == matrix);
}

BOOST_AUTO_TEST_CASE(matrix_inequality)
{
  Matrix4f a = matrix;
  Matrix4f b = matrix;
  b.a13 += 0.0001;
  BOOST_CHECK(!(a == b));
}

BOOST_AUTO_TEST_CASE(matrix_apprx_equality)
{
  Matrix4f a = matrix;
  Matrix4f b = matrix;
  b.a22 -= 0.00001;
  BOOST_CHECK(a.apprx_equal(b, 0.000011));

  a = matrix;
  b = matrix;
  b.a22 += 0.00001;
  BOOST_CHECK(a.apprx_equal(b, 0.000011));
}

BOOST_AUTO_TEST_CASE(matrix_apprx_inequality)
{
  Matrix4f a = matrix;
  Matrix4f b = matrix;
  b.a22 += 0.000013;
  BOOST_CHECK(!a.apprx_equal(b, 0.000011));
}

BOOST_AUTO_TEST_CASE(matrix_transpose)
{
  Matrix4f result;
  Matrix4f correct_result = Matrix4f(0.9751700, 0.1976770, 0.0998334, 0, /**/
                                     -0.218711, 0.930432, 0.294044, 0,/**/
                                     -0.0347626, -0.3085770, 0.9505640, 0,/**/
                                     10, 11, 12, 1);/**/

  transpose(matrix, result);
  BOOST_CHECK(result == correct_result);

}
BOOST_AUTO_TEST_CASE(matrix_multiply)
{

  Matrix4f result;
  Matrix4f a = Matrix4f(1, 2, 3, 4, 4, 3, 2, 1, 2, 1, 3, 4, 2, 3, 1, 4);
  Matrix4f b = Matrix4f(7, 4, 5, 6, 6, 5, 7, 4, 4, 6, 5, 7, 4, 5, 6, 7);
  Matrix4f r = Matrix4f(47, 52, 58, 63, 58, 48, 57, 57, 48, 51, 56, 65, 52, 49, 60, 59);

  result = a * b;

  BOOST_CHECK(result == r);

}

BOOST_AUTO_TEST_CASE(matrix_inverse)
{

  Matrix4f inverse;
  invertMatrix(matrix, inverse);

  Matrix4f result = matrix * inverse;

  Matrix4f identity;
  identity.setIdentity();

  BOOST_CHECK(identity.apprx_equal(result, 0.00000011));
}



BOOST_AUTO_TEST_CASE(matrix_rpy)
{
  gpu_voxels::Vector3f rot_a; // Input given in RPY
  gpu_voxels::Vector3f rot_b; // Generated. Given in RPY
  gpu_voxels::Vector3f rot_c; // Generated. Given in RPY

  Matrix4f a; // holds rot_a in form of a matrix
  Matrix4f b; // holds rot_b in form of a matrix
  Matrix4f c; // holds rot_c in form of a matrix


  boost::mt19937 rng;
  boost::uniform_real<> plus_minus_two_pi(-2.0 * M_PI, 2.0 * M_PI);
  boost::variate_generator< boost::mt19937, boost::uniform_real<> > gen(rng, plus_minus_two_pi);

  // Calculate two RPY representations (rot_b, rot_c) from the matrix a, then convert them into matrices (b, c).
  // Derive the rotation difference betwee a & b and a & c and see, if they are small enough (< 0.005 due to float rounding errors).
  for(size_t n = 0; n < 1000; n++)
  {
    rot_a = gpu_voxels::Vector3f(gen(), gen(), gen());
    a = rotateRPY(rot_a);
    Mat4ToRPY(a, rot_b, 0);
    Mat4ToRPY(a, rot_c, 1);

    b = rotateRPY(rot_b);
    c = rotateRPY(rot_c);

    if(!(orientationMatrixDiff(a, b).apprx_equal(gpu_voxels::Vector3f(0.0), 0.005)))
    {
       std::cout << "a = " << a << " b = " << b << " Diff = " << orientationMatrixDiff(a, b) << std::endl;
    }
    if(!(orientationMatrixDiff(a, c).apprx_equal(gpu_voxels::Vector3f(0.0), 0.005)))
    {
       std::cout << "a = " << a << " c = " << c << " Diff = " << orientationMatrixDiff(a, c) << std::endl;
    }


    BOOST_CHECK_MESSAGE(orientationMatrixDiff(a, b).apprx_equal(gpu_voxels::Vector3f(0.0), 0.005), "Difference between input and first reconstructed RPY is zero.");
    BOOST_CHECK_MESSAGE(orientationMatrixDiff(a, c).apprx_equal(gpu_voxels::Vector3f(0.0), 0.005), "Difference between input and second reconstructed RPY is zero.");
  }
}

BOOST_AUTO_TEST_SUITE_END()

