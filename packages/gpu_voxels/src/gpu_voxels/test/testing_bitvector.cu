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
 * \author  Andreas Hermann <hermann@fzi.de>
 * \date    2016-06-20
 *
 */
//----------------------------------------------------------------------

#include <boost/test/unit_test.hpp>


#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/BitVector.h>
#include <gpu_voxels/test/testing_fixtures.hpp>

using namespace gpu_voxels;


BOOST_FIXTURE_TEST_SUITE(bitvector, ArgsFixture)

BOOST_AUTO_TEST_CASE(bitvector_logic_operations)
{
  PERF_MON_START("bitvector_logic_operations");
  for(int i = 0; i < iterationCount; i++)
  {
    BitVector<BIT_VECTOR_LENGTH> a;
    BitVector<BIT_VECTOR_LENGTH> not_a;
    BitVector<BIT_VECTOR_LENGTH> b;
    BitVector<BIT_VECTOR_LENGTH> c;
    BitVector<BIT_VECTOR_LENGTH> same_a;
    BitVector<BIT_VECTOR_LENGTH> tmp;

    std::stringstream a_strstr("1011100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011");
    std::stringstream same_a_strstr("1011100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011");
    std::stringstream not_a_strstr("0100011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100");
    std::stringstream b_strstr("1011100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011");


    a_strstr >> a;
    same_a_strstr >> same_a;
    not_a_strstr >> not_a;
    b_strstr >> b;

    BOOST_CHECK_MESSAGE(tmp.isZero(), "isZero() operator failed.");
    BOOST_CHECK_MESSAGE(!a.isZero(), "isZero() operator failed.");

    BOOST_CHECK_MESSAGE(a == same_a, "EQUALS operator failed.");
    BOOST_CHECK_MESSAGE(!(not_a == a), "EQUALS operator failed.");

    tmp = a;
    BOOST_CHECK_MESSAGE(a == tmp, "SET operator failed.");

    tmp.clear();
    BOOST_CHECK_MESSAGE(tmp.isZero(), "clear() operator failed.");
    tmp = BitVector<BIT_VECTOR_LENGTH>(a);
    BOOST_CHECK_MESSAGE(a == tmp, "CopyCTor operator failed.");

    c = ~a;
    BOOST_CHECK_MESSAGE(c == not_a, "NOT operator failed.");
    BOOST_CHECK_MESSAGE((a & not_a).isZero(), "AND operator failed.");
    BOOST_CHECK_MESSAGE( (~(a | not_a)).isZero(), "OR operator failed.");

    tmp.clear();
    tmp.setBit(eBVM_FREE);
    tmp.setBit(eBVM_COLLISION);
    tmp.setBit(eBVM_UNKNOWN);
    tmp.setBit(eBVM_SWEPT_VOLUME_START);
    tmp.setBit(BitVoxelMeaning(111));
    tmp.setBit(eBVM_SWEPT_VOLUME_END);
    tmp.setBit(eBVM_UNDEFINED);
    BOOST_CHECK_MESSAGE(tmp == b, "setBit() operator failed.");

    for(size_t i = 0; i < BIT_VECTOR_LENGTH; i++)
    {
      bool myBit = b.getBit(i);
      switch(i)
      {
        case eBVM_FREE:
        case eBVM_COLLISION:
        case eBVM_UNKNOWN:
        case eBVM_SWEPT_VOLUME_START:
        case 111:
        case eBVM_SWEPT_VOLUME_END:
        case eBVM_UNDEFINED:
             BOOST_CHECK_MESSAGE(myBit, "getBit() operator failed.");
             break;
        default:
             BOOST_CHECK_MESSAGE(!myBit, "getBit() operator failed.");
       }
    }

    tmp.clear();
    BOOST_CHECK_MESSAGE(tmp.isZero(), "clear() operator failed.");
    tmp.setByte( 0*8, uint8_t(29));
    tmp.setByte(13*8, uint8_t(128));
    tmp.setByte(31*8, uint8_t(192));
    BOOST_CHECK_MESSAGE(tmp == b, "setByte() operator failed.");


    b.clearBit(eBVM_COLLISION);
    BOOST_CHECK_MESSAGE(!(b.getBit(eBVM_COLLISION)), "clearBit() operator failed.");


    // a OR not_a ==> All bits set. Invert it ==> No bits set.
    a |= not_a;
    BOOST_CHECK_MESSAGE((~a).isZero(), "SET OR operator failed.");

    tmp.clear();
    BOOST_CHECK_MESSAGE(tmp.isZero(), "clear() operator failed.");
    BOOST_CHECK_MESSAGE(tmp.noneButEmpty(), "noneButEmpty() operator failed.");
    tmp.setBit(eBVM_FREE);
    BOOST_CHECK_MESSAGE(tmp.noneButEmpty(), "noneButEmpty() operator failed.");
    tmp.setBit(123);
    BOOST_CHECK_MESSAGE(!tmp.noneButEmpty(), "noneButEmpty() operator failed.");

    // The remaining operators are tested in the VoxelMap:
    // bitMarginCollisionCheck()
    // performLeftShift()
    // <<
    // >>
  }
}

BOOST_AUTO_TEST_CASE(bitvector_bitshift)
{
  PERF_MON_START("bitvector_bitshift");
  for(int i = 0; i < iterationCount; i++)
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



      BOOST_TEST_MESSAGE("-------- Shifting " << shift << " Bits ----------");

      BOOST_TEST_MESSAGE("Before shift " << my_bitvector);
      performLeftShift(my_bitvector, shift);
      BOOST_TEST_MESSAGE("After shift " << my_bitvector);

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

      BOOST_TEST_MESSAGE("-------- Shifted " << shift << " Bits ----------");
    }
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvector_bitshift", "bitvector_bitshift", "bitvector");
  }
}



//BOOST_AUTO_TEST_CASE(bitvector_bitshift_collision)
//{

//}

BOOST_AUTO_TEST_SUITE_END()


