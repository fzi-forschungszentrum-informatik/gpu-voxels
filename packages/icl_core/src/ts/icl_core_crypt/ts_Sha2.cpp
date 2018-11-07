// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2012-01-19
 *
 */
//----------------------------------------------------------------------

#include <icl_core_crypt/Sha2.h>

#include <boost/test/unit_test.hpp>

using ::icl_core::crypt::Sha224;
using ::icl_core::crypt::Sha256;
using ::icl_core::crypt::Sha384;
using ::icl_core::crypt::Sha512;

BOOST_AUTO_TEST_SUITE(ts_Sha2)

//----------------------------------------------------------------------
// Empty strings
//----------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(EmptyStringDigestSha224)
{
  Sha224 sha;
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f");
}

BOOST_AUTO_TEST_CASE(EmptyStringDigestSha256)
{
  Sha256 sha;
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

BOOST_AUTO_TEST_CASE(EmptyStringDigestSha384)
{
  Sha384 sha;
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b");
}

BOOST_AUTO_TEST_CASE(EmptyStringDigestSha512)
{
  Sha512 sha;
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
}

//----------------------------------------------------------------------
// The quick brown fox
//----------------------------------------------------------------------

const char *teststring1 = "The quick brown fox jumps over the lazy dog";
const char *teststring2 = "The quick brown fox jumps over the lazy dog.";

BOOST_AUTO_TEST_CASE(SimpleStringDigestSha224)
{
  Sha224 sha;
  sha.process(teststring1);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "730e109bd7a8a32b1cb9d9a09aa2325d2430587ddbc0c38bad911525");
  sha.clear();
  sha.process(teststring2);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "619cba8e8e05826e9b8c519c0a5c68f4fb653e8a3d8aa04bb2c8cd4c");
}

BOOST_AUTO_TEST_CASE(SimpleStringDigestSha256)
{
  Sha256 sha;
  sha.process(teststring1);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592");
  sha.clear();
  sha.process(teststring2);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "ef537f25c895bfa782526529a9b63d97aa631564d5d789c2b765448c8635fb6c");
}

BOOST_AUTO_TEST_CASE(SimpleStringDigestSha384)
{
  Sha384 sha;
  sha.process(teststring1);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "ca737f1014a48f4c0b6dd43cb177b0afd9e5169367544c494011e3317dbf9a509cb1e5dc1e85a941bbee3d7f2afbc9b1");
  sha.clear();
  sha.process(teststring2);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "ed892481d8272ca6df370bf706e4d7bc1b5739fa2177aae6c50e946678718fc67a7af2819a021c2fc34e91bdb63409d7");
}

BOOST_AUTO_TEST_CASE(SimpleStringDigestSha512)
{
  Sha512 sha;
  sha.process(teststring1);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "07e547d9586f6a73f73fbac0435ed76951218fb7d0c8d788a309d785436bbb642e93a252a954f23912547d1e8a3b5ed6e1bfd7097821233fa0538f3db854fee6");
  sha.clear();
  sha.process(teststring2);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "91ea1245f20d46ae9a037a989f54f1f790f0a47607eeb8a14d12890cea77a1bbc6c7ed9cf205e67b7f2b8fd4c7dfd3a7a8617e45f3c463d481c7e586c39ac1ed");
}

//----------------------------------------------------------------------
// Longer string consisting of more than one message block
//----------------------------------------------------------------------

const char *longstring =
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam "
  "lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam "
  "viverra nec consectetur ante hendrerit. Donec et mollis "
  "dolor. Praesent et diam eget libero egestas mattis sit amet vitae "
  "augue. Nam tincidunt congue enim, ut porta lorem lacinia "
  "consectetur. Donec ut libero sed arcu vehicula ultricies a non "
  "tortor. Lorem ipsum dolor sit amet, consectetur adipiscing "
  "elit. Aenean ut gravida lorem. Ut turpis felis, pulvinar a semper "
  "sed, adipiscing id dolor. Pellentesque auctor nisi id magna "
  "consequat sagittis. Curabitur dapibus enim sit amet elit pharetra "
  "tincidunt feugiat nisl imperdiet. Ut convallis libero in urna "
  "ultrices accumsan. Donec sed odio eros. Donec viverra mi quis quam "
  "pulvinar at malesuada arcu rhoncus. Cum sociis natoque penatibus et "
  "magnis dis parturient montes, nascetur ridiculus mus. In rutrum "
  "accumsan ultricies. Mauris vitae nisi at sem facilisis semper ac in "
  "est.\n"
  "Vivamus fermentum semper porta. Nunc diam velit, adipiscing ut "
  "tristique vitae, sagittis vel odio. Maecenas convallis ullamcorper "
  "ultricies. Curabitur ornare, ligula semper consectetur sagittis, "
  "nisi diam iaculis velit, id fringilla sem nunc vel mi. Nam dictum, "
  "odio nec pretium volutpat, arcu ante placerat erat, non tristique "
  "elit urna et turpis. Quisque mi metus, ornare sit amet fermentum "
  "et, tincidunt et orci. Fusce eget orci a orci congue vestibulum. Ut "
  "dolor diam, elementum et vestibulum eu, porttitor vel "
  "elit. Curabitur venenatis pulvinar tellus gravida ornare. Sed et "
  "erat faucibus nunc euismod ultricies ut id justo. Nullam cursus "
  "suscipit nisi, et ultrices justo sodales nec. Fusce venenatis "
  "facilisis lectus ac semper. Aliquam at massa ipsum. Quisque "
  "bibendum purus convallis nulla ultrices ultricies. Nullam aliquam, "
  "mi eu aliquam tincidunt, purus velit laoreet tortor, viverra "
  "pretium nisi quam vitae mi. Fusce vel volutpat elit. Nam sagittis "
  "nisi dui.\n";

BOOST_AUTO_TEST_CASE(LongStringDigestSha224)
{
  Sha224 sha;
  sha.process(longstring);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "9319b8ec0551910b30009ba504103bbdbb1e02ed8e9247320e01dccf");
}

BOOST_AUTO_TEST_CASE(LongStringDigestSha256)
{
  Sha256 sha;
  sha.process(longstring);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "93ba6b27717ba945b8f43e9ad1d44dc02123b13f9000eb4b25b8a153422b6677");
}

BOOST_AUTO_TEST_CASE(LongStringDigestSha384)
{
  Sha384 sha;
  sha.process(longstring);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "866e65e74170721a59ced427e884d1e25a1d62862387af2004d30549e3e05a34875283c2d1fd2aa13989fff8b6ecad9f");
}

BOOST_AUTO_TEST_CASE(LongStringDigestSha512)
{
  Sha512 sha;
  sha.process(longstring);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "aff2b51a8c5555603914381005f6f46bec85793610e89cf9a3517ff3b9e98c7e5e1a636f0b46b1288478d30ec83c434d5e07f86ca6ed8507a9a8b03d63d1f5c7");
}

//----------------------------------------------------------------------
// String ending exactly one byte before the padding boundary
//----------------------------------------------------------------------

const char *padstring256_440bit = "....:....|....:....|....:....|....:....|....:....|....:";
const char *padstring512_888bit = "....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|.";

BOOST_AUTO_TEST_CASE(Pad440Sha224)
{
  Sha224 sha;
  sha.process(padstring256_440bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "a526362dae0b4aa0fdd4966ed84d51f6ab3dbff574f75613e8f21ba2");
}

BOOST_AUTO_TEST_CASE(Pad440Sha256)
{
  Sha256 sha;
  sha.process(padstring256_440bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "d0c06dbf5fec4c438d6cacf1113435e7e504ed45c04efb5cb70776f1d391e4d9");
}

BOOST_AUTO_TEST_CASE(Pad888Sha384)
{
  Sha384 sha;
  sha.process(padstring512_888bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "ac807c831aa023556137b6aeaf0b1ec33ff24c99a858ffe97a1926dfd99f3f4cb81b0afd61ff29f01ea5f8851a22ee28");
}

BOOST_AUTO_TEST_CASE(Pad888Sha512)
{
  Sha512 sha;
  sha.process(padstring512_888bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "d3818ef0619071407cab75933bee598cf8b26976d3c51cade7bd86ebc951e5ce53cd18d3784f152bb2536622c6538331f42c2f44f811dd9a75ae345ec7dd060e");
}

//----------------------------------------------------------------------
// String ending exactly on the padding boundary
//----------------------------------------------------------------------

const char *padstring256_448bit = "....:....|....:....|....:....|....:....|....:....|....:.";
const char *padstring512_896bit = "....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|..";

BOOST_AUTO_TEST_CASE(Pad448Sha224)
{
  Sha224 sha;
  sha.process(padstring256_448bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "a95081d421a9576d6507b21bab994a139dab99241cc70a1bc4db8763");
}

BOOST_AUTO_TEST_CASE(Pad448Sha256)
{
  Sha256 sha;
  sha.process(padstring256_448bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "5f237fc93336874639ea7fe7a2f55e58b954946f90ab3254f6a93ad33562279f");
}

BOOST_AUTO_TEST_CASE(Pad896Sha384)
{
  Sha384 sha;
  sha.process(padstring512_896bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "c2343fcfe20cedf0ff1314b03002d710d963928687bcc29ca66ed4891bc8ccecb9601771fa8119241f756a7d55b4117b");
}

BOOST_AUTO_TEST_CASE(Pad896Sha512)
{
  Sha512 sha;
  sha.process(padstring512_896bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "ada0d27f302ed4fec585ee259b0590454e159d023befd97782c6fad093284463bc0c366acfed361359d6f63de07291e8039abbe0366c89ee4ad3d67e244fdd04");
}

//----------------------------------------------------------------------
// String ending exactly one byte after the padding boundary
//----------------------------------------------------------------------

const char *padstring256_456bit = "....:....|....:....|....:....|....:....|....:....|....:..";
const char *padstring512_904bit = "....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|....:....|...";

BOOST_AUTO_TEST_CASE(Pad456Sha224)
{
  Sha224 sha;
  sha.process(padstring256_456bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "d6c17098edf2463cd98f9f97c58fca8a881d0e83802193c81742129a");
}

BOOST_AUTO_TEST_CASE(Pad456Sha256)
{
  Sha256 sha;
  sha.process(padstring256_456bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "a1d509b8f4d8fac578e40b57273c8d665e2716fe2446b32e25d2d10eb218bd0e");
}

BOOST_AUTO_TEST_CASE(Pad904Sha384)
{
  Sha384 sha;
  sha.process(padstring512_904bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "392bc91f4da2a2f545f19a911fcf5072c8fa7a8909ed1ee9175d17af587f213a02c3a0acf8ee6868129b383c70bb2cc6");
}

BOOST_AUTO_TEST_CASE(Pad904Sha512)
{
  Sha512 sha;
  sha.process(padstring512_904bit);
  sha.finalize();
  BOOST_CHECK_EQUAL(sha.getHexDigest(), "73ce0c347018a80cb9781cc0eb302e43b35384db671c512f31c369c89e2b4ad6ea1cfe59abcc3698b7c7bc01c776e67e804f93823730244a9620d6526baee3da");
}

BOOST_AUTO_TEST_SUITE_END()
