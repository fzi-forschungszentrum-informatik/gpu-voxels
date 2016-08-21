// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann <hermann@fzi.de>
 * \date    2016-06-20
 *
 */
//----------------------------------------------------------------------

#include <boost/test/unit_test.hpp>


#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/BitVector.h>

using namespace gpu_voxels;

BOOST_AUTO_TEST_SUITE(bitvector)

BOOST_AUTO_TEST_CASE(bitvector_bitshift)
{
  for(size_t shift = 3; shift < 20; shift += 10)
  {
    BitVector<BIT_VECTOR_LENGTH> my_bitvector;
    my_bitvector.setBit(eBVM_FREE);
    my_bitvector.setBit(eBVM_OCCUPIED);
    my_bitvector.setBit(eBVM_COLLISION);
    my_bitvector.setBit(eBVM_UNKNOWN);
    my_bitvector.setBit(eBVM_SWEPT_VOLUME_START);
    my_bitvector.setBit(eBVM_SWEPT_VOLUME_START + 10); // This would slide into eBVM_OCCUPIED
    my_bitvector.setBit(eBVM_SWEPT_VOLUME_START + 20);
    my_bitvector.setBit(eBVM_SWEPT_VOLUME_END);
    my_bitvector.setBit(eBVM_UNDEFINED);



    BOOST_MESSAGE("-------- Shifting " << shift << " Bits ----------");

    BOOST_MESSAGE("Before shift " << my_bitvector);
    performLeftShift(my_bitvector, shift);
    BOOST_MESSAGE("After shift " << my_bitvector);

    // Non BV Bits should be gone:
    BOOST_CHECK_MESSAGE(!my_bitvector.getBit(eBVM_FREE), "Unset eBVM_FREE.");
    BOOST_CHECK_MESSAGE(!my_bitvector.getBit(eBVM_OCCUPIED), "Unset eBVM_OCCUPIED.");
    BOOST_CHECK_MESSAGE(!my_bitvector.getBit(eBVM_COLLISION), "Unset eBVM_COLLISION.");
    BOOST_CHECK_MESSAGE(!my_bitvector.getBit(eBVM_UNKNOWN), "Unset eBVM_UNKNOWN.");

    // The original bits should not be set any more:
    BOOST_CHECK_MESSAGE(!my_bitvector.getBit(eBVM_SWEPT_VOLUME_START), "Unset eBVM_SWEPT_VOLUME_START.");
    BOOST_CHECK_MESSAGE(!my_bitvector.getBit(eBVM_SWEPT_VOLUME_START + 10), "Unset eBVM_SWEPT_VOLUME_START + 10.");
    BOOST_CHECK_MESSAGE(!my_bitvector.getBit(eBVM_SWEPT_VOLUME_START + 20), "Unset eBVM_SWEPT_VOLUME_START + 20.");
    BOOST_CHECK_MESSAGE(!my_bitvector.getBit(eBVM_SWEPT_VOLUME_END), "Unset eBVM_SWEPT_VOLUME_END.");
    BOOST_CHECK_MESSAGE(!my_bitvector.getBit(eBVM_UNDEFINED), "Unset eBVM_UNDEFINED.");

    // Shifted bits (in range) should be there:
    if((eBVM_SWEPT_VOLUME_START + 20 - shift) > eBVM_UNKNOWN)
    {
      BOOST_CHECK_MESSAGE(my_bitvector.getBit((eBVM_SWEPT_VOLUME_START + 20) - shift), "Set eBVM_SWEPT_VOLUME_START + 20 - shift");
    } else {
      BOOST_CHECK_MESSAGE(!my_bitvector.getBit((eBVM_SWEPT_VOLUME_START + 20) - shift), "Set eBVM_SWEPT_VOLUME_START + 20 - shift");
    }
    BOOST_CHECK_MESSAGE(my_bitvector.getBit(eBVM_SWEPT_VOLUME_END - shift), "Set eBVM_SWEPT_VOLUME_END - shift");
    BOOST_CHECK_MESSAGE(my_bitvector.getBit(eBVM_UNDEFINED - shift), "Set eBVM_UNKNOWN - shift");

    BOOST_MESSAGE("-------- Shifted " << shift << " Bits ----------");
  }
}



//BOOST_AUTO_TEST_CASE(bitvector_bitshift_collision)
//{

//}

BOOST_AUTO_TEST_SUITE_END()


