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

using namespace gpu_voxels;

bool matrix_equal(Matrix4f a, Matrix4f b)
{
  return a.a11 == b.a11 && a.a12 == b.a12 && a.a13 == b.a13 && a.a14 == b.a14 && /**/
  a.a21 == b.a21 && a.a22 == b.a22 && a.a23 == b.a23 && a.a24 == b.a24 && /**/
  a.a31 == b.a31 && a.a32 == b.a32 && a.a33 == b.a33 && a.a34 == b.a34 && /**/
  a.a41 == b.a41 && a.a42 == b.a42 && a.a43 == b.a43 && a.a44 == b.a44;/**/
}

BOOST_AUTO_TEST_SUITE(cudaMath)

BOOST_AUTO_TEST_CASE(MatrixEqual)
{
  Matrix4f matrix = Matrix4f(0.9751700, -0.218711, -0.0347626, 10, /**/
                             0.1976770, 0.930432, -0.3085770, 11,/**/
                             0.0998334, 0.294044, 0.9505640, 12,/**/
                             0, 0, 0, 1);/**/
  BOOST_CHECK(matrix_equal(matrix, matrix));
}

BOOST_AUTO_TEST_CASE(transpose)
{
  Matrix4f result;
  Matrix4f matrix = Matrix4f(0.9751700, -0.218711, -0.0347626, 10, /**/
                             0.1976770, 0.930432, -0.3085770, 11,/**/
                             0.0998334, 0.294044, 0.9505640, 12,/**/
                             0, 0, 0, 1);/**/

  Matrix4f correct_result = Matrix4f(0.9751700, 0.1976770, 0.0998334, 0, /**/
                                     -0.218711, 0.930432, 0.294044, 0,/**/
                                     -0.0347626, -0.3085770, 0.9505640, 0,/**/
                                     10, 11, 12, 1);/**/
  CudaMath* cuda_math = new CudaMath();

  cuda_math->transpose(matrix, result);

  BOOST_CHECK(matrix_equal(result, correct_result));

}
BOOST_AUTO_TEST_CASE(multiply)
{

  Matrix4f result;
  Matrix4f a = Matrix4f(1, 2, 3, 4, 4, 3, 2, 1, 2, 1, 3, 4, 2, 3, 1, 4);
  Matrix4f b = Matrix4f(7, 4, 5, 6, 6, 5, 7, 4, 4, 6, 5, 7, 4, 5, 6, 7);
  Matrix4f r = Matrix4f(47, 52, 58, 63, 58, 48, 57, 57, 48, 51, 56, 65, 52, 49, 60, 59);

  result = a * b;

  BOOST_CHECK(matrix_equal(result, r));

}
BOOST_AUTO_TEST_SUITE_END()

