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
#ifndef ICL_CORE_CRYPT_SHA2_IMPL_HPP_INCLUDED
#define ICL_CORE_CRYPT_SHA2_IMPL_HPP_INCLUDED

#include <sstream>
#include <iomanip>
#include <string.h>

namespace icl_core {
namespace crypt {

#define TEMPLATEM template <typename T, T t_h0, T t_h1, T t_h2, T t_h3, T t_h4, T t_h5, T t_h6, T t_h7, size_t t_len>
#define CLASSM Sha2Impl<T, t_h0, t_h1, t_h2, t_h3, t_h4, t_h5, t_h6, t_h7, t_len>

TEMPLATEM
CLASSM::Sha2Impl()
{
  clear();
}

TEMPLATEM
void CLASSM::clear()
{
  m_message_size = 0;
  m_buffer_fill = 0;
  m_digest[0] = t_h0;
  m_digest[1] = t_h1;
  m_digest[2] = t_h2;
  m_digest[3] = t_h3;
  m_digest[4] = t_h4;
  m_digest[5] = t_h5;
  m_digest[6] = t_h6;
  m_digest[7] = t_h7;
}

TEMPLATEM
::icl_core::String CLASSM::getHexDigest() const
{
  std::stringstream ss;
  for (size_t i = 0; i < t_len; ++i)
  {
    ss << std::hex << std::setw(8) << std::setfill('0') << m_digest[i];
  }
  return ss.str();
}

#define bswaparr(buf, T, i) ((static_cast<T>(buf[(i)*4+3])      ) |     \
                             (static_cast<T>(buf[(i)*4+2]) <<  8) |     \
                             (static_cast<T>(buf[(i)*4+1]) << 16) |     \
                             (static_cast<T>(buf[(i)*4+0]) << 24))

#define bswaparr64(buf, T, i) ((static_cast<T>(buf[(i)*8+7])      ) |   \
                               (static_cast<T>(buf[(i)*8+6]) <<  8) |   \
                               (static_cast<T>(buf[(i)*8+5]) << 16) |   \
                               (static_cast<T>(buf[(i)*8+4]) << 24) |   \
                               (static_cast<T>(buf[(i)*8+3]) << 32) |   \
                               (static_cast<T>(buf[(i)*8+2]) << 40) |   \
                               (static_cast<T>(buf[(i)*8+1]) << 48) |   \
                               (static_cast<T>(buf[(i)*8+0]) << 56))

//! FIPS PUB 180-3 "Ch" operation
#define Ch(x, y, z)  ((z) ^ ((x) & ((y) ^ (z))))
//! FIPS PUB 180-3 "Maj" operation
#define Maj(x, y, z) (((x) & (y)) | ((z) & ((x) ^ (y))))
//! Right rotation
#define Rotr(x, n, nbits) (((x) >> (n)) | ((x) << ((nbits)-(n))))
//! Right shift
#define Shr(x, n) ((x) >> (n))
//! Flexible variable mapping to avoid unnecessary copying.
#define wv(i) v##i

#define Sigma0_256(x) (Rotr((x),  2, 32) ^ Rotr((x), 13, 32) ^ Rotr((x), 22, 32))
#define Sigma1_256(x) (Rotr((x),  6, 32) ^ Rotr((x), 11, 32) ^ Rotr((x), 25, 32))
#define sigma0_256(x) (Rotr((x),  7, 32) ^ Rotr((x), 18, 32) ^  Shr((x),  3))
#define sigma1_256(x) (Rotr((x), 17, 32) ^ Rotr((x), 19, 32) ^  Shr((x), 10))

//----------------------------------------------------------------------
// Implementation specifics for 32-bit words
//----------------------------------------------------------------------

static const uint32_t k256[64] = {
  0x428a2f98ul, 0x71374491ul, 0xb5c0fbcful, 0xe9b5dba5ul, 0x3956c25bul, 0x59f111f1ul, 0x923f82a4ul, 0xab1c5ed5ul,
  0xd807aa98ul, 0x12835b01ul, 0x243185beul, 0x550c7dc3ul, 0x72be5d74ul, 0x80deb1feul, 0x9bdc06a7ul, 0xc19bf174ul,
  0xe49b69c1ul, 0xefbe4786ul, 0x0fc19dc6ul, 0x240ca1ccul, 0x2de92c6ful, 0x4a7484aaul, 0x5cb0a9dcul, 0x76f988daul,
  0x983e5152ul, 0xa831c66dul, 0xb00327c8ul, 0xbf597fc7ul, 0xc6e00bf3ul, 0xd5a79147ul, 0x06ca6351ul, 0x14292967ul,
  0x27b70a85ul, 0x2e1b2138ul, 0x4d2c6dfcul, 0x53380d13ul, 0x650a7354ul, 0x766a0abbul, 0x81c2c92eul, 0x92722c85ul,
  0xa2bfe8a1ul, 0xa81a664bul, 0xc24b8b70ul, 0xc76c51a3ul, 0xd192e819ul, 0xd6990624ul, 0xf40e3585ul, 0x106aa070ul,
  0x19a4c116ul, 0x1e376c08ul, 0x2748774cul, 0x34b0bcb5ul, 0x391c0cb3ul, 0x4ed8aa4aul, 0x5b9cca4ful, 0x682e6ff3ul,
  0x748f82eeul, 0x78a5636ful, 0x84c87814ul, 0x8cc70208ul, 0x90befffaul, 0xa4506cebul, 0xbef9a3f7ul, 0xc67178f2ul
};

#define round_0_15(a, b, c, d, e, f, g, h, K, W, t)                     \
  T1 = wv(h) + Sigma1_256(wv(e)) + Ch(wv(e), wv(f), wv(g)) + K[t] + W[t&15]; \
  T2 = Sigma0_256(wv(a)) + Maj(wv(a), wv(b), wv(c));                    \
  wv(d) += T1;                                                          \
  wv(h) = T1 + T2;

#define round_16_63(a, b, c, d, e, f, g, h, K, W, t)                    \
  W[(t)&15] += sigma1_256(W[(t+14)&15]) + W[(t+9)&15] + sigma0_256(W[(t+1)&15]); \
  T1 = wv(h) + Sigma1_256(wv(e)) + Ch(wv(e), wv(f), wv(g)) + K[t] + W[t&15]; \
  T2 = Sigma0_256(wv(a)) + Maj(wv(a), wv(b), wv(c));                    \
  wv(d) += T1;                                                          \
  wv(h) = T1 + T2;

TEMPLATEM
void CLASSM::processBuffer()
{
  T v0 = m_digest[0];
  T v1 = m_digest[1];
  T v2 = m_digest[2];
  T v3 = m_digest[3];
  T v4 = m_digest[4];
  T v5 = m_digest[5];
  T v6 = m_digest[6];
  T v7 = m_digest[7];

  T w[16] = {
    bswaparr(m_buffer, T,  0), bswaparr(m_buffer, T,  1),
    bswaparr(m_buffer, T,  2), bswaparr(m_buffer, T,  3),
    bswaparr(m_buffer, T,  4), bswaparr(m_buffer, T,  5),
    bswaparr(m_buffer, T,  6), bswaparr(m_buffer, T,  7),
    bswaparr(m_buffer, T,  8), bswaparr(m_buffer, T,  9),
    bswaparr(m_buffer, T, 10), bswaparr(m_buffer, T, 11),
    bswaparr(m_buffer, T, 12), bswaparr(m_buffer, T, 13),
    bswaparr(m_buffer, T, 14), bswaparr(m_buffer, T, 15)
  };
  T T1, T2;

  round_0_15(0,1,2,3,4,5,6,7, k256, w,  0);
  round_0_15(7,0,1,2,3,4,5,6, k256, w,  1);
  round_0_15(6,7,0,1,2,3,4,5, k256, w,  2);
  round_0_15(5,6,7,0,1,2,3,4, k256, w,  3);
  round_0_15(4,5,6,7,0,1,2,3, k256, w,  4);
  round_0_15(3,4,5,6,7,0,1,2, k256, w,  5);
  round_0_15(2,3,4,5,6,7,0,1, k256, w,  6);
  round_0_15(1,2,3,4,5,6,7,0, k256, w,  7);

  round_0_15(0,1,2,3,4,5,6,7, k256, w,  8);
  round_0_15(7,0,1,2,3,4,5,6, k256, w,  9);
  round_0_15(6,7,0,1,2,3,4,5, k256, w, 10);
  round_0_15(5,6,7,0,1,2,3,4, k256, w, 11);
  round_0_15(4,5,6,7,0,1,2,3, k256, w, 12);
  round_0_15(3,4,5,6,7,0,1,2, k256, w, 13);
  round_0_15(2,3,4,5,6,7,0,1, k256, w, 14);
  round_0_15(1,2,3,4,5,6,7,0, k256, w, 15);

  round_16_63(0,1,2,3,4,5,6,7, k256, w, 16);
  round_16_63(7,0,1,2,3,4,5,6, k256, w, 17);
  round_16_63(6,7,0,1,2,3,4,5, k256, w, 18);
  round_16_63(5,6,7,0,1,2,3,4, k256, w, 19);
  round_16_63(4,5,6,7,0,1,2,3, k256, w, 20);
  round_16_63(3,4,5,6,7,0,1,2, k256, w, 21);
  round_16_63(2,3,4,5,6,7,0,1, k256, w, 22);
  round_16_63(1,2,3,4,5,6,7,0, k256, w, 23);

  round_16_63(0,1,2,3,4,5,6,7, k256, w, 24);
  round_16_63(7,0,1,2,3,4,5,6, k256, w, 25);
  round_16_63(6,7,0,1,2,3,4,5, k256, w, 26);
  round_16_63(5,6,7,0,1,2,3,4, k256, w, 27);
  round_16_63(4,5,6,7,0,1,2,3, k256, w, 28);
  round_16_63(3,4,5,6,7,0,1,2, k256, w, 29);
  round_16_63(2,3,4,5,6,7,0,1, k256, w, 30);
  round_16_63(1,2,3,4,5,6,7,0, k256, w, 31);

  round_16_63(0,1,2,3,4,5,6,7, k256, w, 32);
  round_16_63(7,0,1,2,3,4,5,6, k256, w, 33);
  round_16_63(6,7,0,1,2,3,4,5, k256, w, 34);
  round_16_63(5,6,7,0,1,2,3,4, k256, w, 35);
  round_16_63(4,5,6,7,0,1,2,3, k256, w, 36);
  round_16_63(3,4,5,6,7,0,1,2, k256, w, 37);
  round_16_63(2,3,4,5,6,7,0,1, k256, w, 38);
  round_16_63(1,2,3,4,5,6,7,0, k256, w, 39);

  round_16_63(0,1,2,3,4,5,6,7, k256, w, 40);
  round_16_63(7,0,1,2,3,4,5,6, k256, w, 41);
  round_16_63(6,7,0,1,2,3,4,5, k256, w, 42);
  round_16_63(5,6,7,0,1,2,3,4, k256, w, 43);
  round_16_63(4,5,6,7,0,1,2,3, k256, w, 44);
  round_16_63(3,4,5,6,7,0,1,2, k256, w, 45);
  round_16_63(2,3,4,5,6,7,0,1, k256, w, 46);
  round_16_63(1,2,3,4,5,6,7,0, k256, w, 47);

  round_16_63(0,1,2,3,4,5,6,7, k256, w, 48);
  round_16_63(7,0,1,2,3,4,5,6, k256, w, 49);
  round_16_63(6,7,0,1,2,3,4,5, k256, w, 50);
  round_16_63(5,6,7,0,1,2,3,4, k256, w, 51);
  round_16_63(4,5,6,7,0,1,2,3, k256, w, 52);
  round_16_63(3,4,5,6,7,0,1,2, k256, w, 53);
  round_16_63(2,3,4,5,6,7,0,1, k256, w, 54);
  round_16_63(1,2,3,4,5,6,7,0, k256, w, 55);

  round_16_63(0,1,2,3,4,5,6,7, k256, w, 56);
  round_16_63(7,0,1,2,3,4,5,6, k256, w, 57);
  round_16_63(6,7,0,1,2,3,4,5, k256, w, 58);
  round_16_63(5,6,7,0,1,2,3,4, k256, w, 59);
  round_16_63(4,5,6,7,0,1,2,3, k256, w, 60);
  round_16_63(3,4,5,6,7,0,1,2, k256, w, 61);
  round_16_63(2,3,4,5,6,7,0,1, k256, w, 62);
  round_16_63(1,2,3,4,5,6,7,0, k256, w, 63);

  m_digest[0] += v0;
  m_digest[1] += v1;
  m_digest[2] += v2;
  m_digest[3] += v3;
  m_digest[4] += v4;
  m_digest[5] += v5;
  m_digest[6] += v6;
  m_digest[7] += v7;
}

#undef TEMPLATEM
#undef CLASSM

#undef round_0_15
#undef round_16_63

//----------------------------------------------------------------------
// Implementation specifics for 64-bit words
//----------------------------------------------------------------------

static const uint64_t k512[80] = {
  0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull,
  0x3956c25bf348b538ull, 0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull,
  0xd807aa98a3030242ull, 0x12835b0145706fbeull, 0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull,
  0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull, 0xc19bf174cf692694ull,
  0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
  0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull,
  0x983e5152ee66dfabull, 0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull,
  0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull, 0x06ca6351e003826full, 0x142929670a0e6e70ull,
  0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull, 0x53380d139d95b3dfull,
  0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
  0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull,
  0xd192e819d6ef5218ull, 0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull,
  0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull, 0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull,
  0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull, 0x682e6ff3d6b2b8a3ull,
  0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
  0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull,
  0xca273eceea26619cull, 0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull,
  0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull, 0x113f9804bef90daeull, 0x1b710b35131c471bull,
  0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull, 0x431d67c49c100d4cull,
  0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull
};

#define TEMPLATEM template <uint64_t t_h0, uint64_t t_h1, uint64_t t_h2, uint64_t t_h3, uint64_t t_h4, uint64_t t_h5, uint64_t t_h6, uint64_t t_h7, size_t t_len>
#define CLASSM Sha2Impl<uint64_t, t_h0, t_h1, t_h2, t_h3, t_h4, t_h5, t_h6, t_h7, t_len>

#define Sigma0_512(x) (Rotr((x), 28, 64) ^ Rotr((x), 34, 64) ^ Rotr((x), 39, 64))
#define Sigma1_512(x) (Rotr((x), 14, 64) ^ Rotr((x), 18, 64) ^ Rotr((x), 41, 64))
#define sigma0_512(x) (Rotr((x),  1, 64) ^ Rotr((x),  8, 64) ^  Shr((x),  7))
#define sigma1_512(x) (Rotr((x), 19, 64) ^ Rotr((x), 61, 64) ^  Shr((x),  6))

TEMPLATEM
CLASSM::Sha2Impl()
{
  clear();
}

TEMPLATEM
void CLASSM::clear()
{
  m_message_size = 0;
  m_buffer_fill = 0;
  m_digest[0] = t_h0;
  m_digest[1] = t_h1;
  m_digest[2] = t_h2;
  m_digest[3] = t_h3;
  m_digest[4] = t_h4;
  m_digest[5] = t_h5;
  m_digest[6] = t_h6;
  m_digest[7] = t_h7;
}

TEMPLATEM
::icl_core::String CLASSM::getHexDigest() const
{
  std::stringstream ss;
  for (size_t i = 0; i < t_len; ++i)
  {
    ss << std::hex << std::setw(16) << std::setfill('0') << m_digest[i];
  }
  return ss.str();
}

#define round_0_15(a, b, c, d, e, f, g, h, K, W, t)                     \
  T1 = wv(h) + Sigma1_512(wv(e)) + Ch(wv(e), wv(f), wv(g)) + K[t] + W[t&15]; \
  T2 = Sigma0_512(wv(a)) + Maj(wv(a), wv(b), wv(c));                    \
  wv(d) += T1;                                                          \
  wv(h) = T1 + T2;

#define round_16_79(a, b, c, d, e, f, g, h, K, W, t)                    \
  W[(t)&15] += sigma1_512(W[(t+14)&15]) + W[(t+9)&15] + sigma0_512(W[(t+1)&15]); \
  T1 = wv(h) + Sigma1_512(wv(e)) + Ch(wv(e), wv(f), wv(g)) + K[t] + W[t&15]; \
  T2 = Sigma0_512(wv(a)) + Maj(wv(a), wv(b), wv(c));                    \
  wv(d) += T1;                                                          \
  wv(h) = T1 + T2;

TEMPLATEM
void CLASSM::processBuffer()
{
  uint64_t v0 = m_digest[0];
  uint64_t v1 = m_digest[1];
  uint64_t v2 = m_digest[2];
  uint64_t v3 = m_digest[3];
  uint64_t v4 = m_digest[4];
  uint64_t v5 = m_digest[5];
  uint64_t v6 = m_digest[6];
  uint64_t v7 = m_digest[7];

  uint64_t w[16] = {
    bswaparr64(m_buffer, uint64_t,  0), bswaparr64(m_buffer, uint64_t,  1),
    bswaparr64(m_buffer, uint64_t,  2), bswaparr64(m_buffer, uint64_t,  3),
    bswaparr64(m_buffer, uint64_t,  4), bswaparr64(m_buffer, uint64_t,  5),
    bswaparr64(m_buffer, uint64_t,  6), bswaparr64(m_buffer, uint64_t,  7),
    bswaparr64(m_buffer, uint64_t,  8), bswaparr64(m_buffer, uint64_t,  9),
    bswaparr64(m_buffer, uint64_t, 10), bswaparr64(m_buffer, uint64_t, 11),
    bswaparr64(m_buffer, uint64_t, 12), bswaparr64(m_buffer, uint64_t, 13),
    bswaparr64(m_buffer, uint64_t, 14), bswaparr64(m_buffer, uint64_t, 15)
  };
  uint64_t T1, T2;

  round_0_15(0,1,2,3,4,5,6,7, k512, w,  0);
  round_0_15(7,0,1,2,3,4,5,6, k512, w,  1);
  round_0_15(6,7,0,1,2,3,4,5, k512, w,  2);
  round_0_15(5,6,7,0,1,2,3,4, k512, w,  3);
  round_0_15(4,5,6,7,0,1,2,3, k512, w,  4);
  round_0_15(3,4,5,6,7,0,1,2, k512, w,  5);
  round_0_15(2,3,4,5,6,7,0,1, k512, w,  6);
  round_0_15(1,2,3,4,5,6,7,0, k512, w,  7);

  round_0_15(0,1,2,3,4,5,6,7, k512, w,  8);
  round_0_15(7,0,1,2,3,4,5,6, k512, w,  9);
  round_0_15(6,7,0,1,2,3,4,5, k512, w, 10);
  round_0_15(5,6,7,0,1,2,3,4, k512, w, 11);
  round_0_15(4,5,6,7,0,1,2,3, k512, w, 12);
  round_0_15(3,4,5,6,7,0,1,2, k512, w, 13);
  round_0_15(2,3,4,5,6,7,0,1, k512, w, 14);
  round_0_15(1,2,3,4,5,6,7,0, k512, w, 15);

  round_16_79(0,1,2,3,4,5,6,7, k512, w, 16);
  round_16_79(7,0,1,2,3,4,5,6, k512, w, 17);
  round_16_79(6,7,0,1,2,3,4,5, k512, w, 18);
  round_16_79(5,6,7,0,1,2,3,4, k512, w, 19);
  round_16_79(4,5,6,7,0,1,2,3, k512, w, 20);
  round_16_79(3,4,5,6,7,0,1,2, k512, w, 21);
  round_16_79(2,3,4,5,6,7,0,1, k512, w, 22);
  round_16_79(1,2,3,4,5,6,7,0, k512, w, 23);

  round_16_79(0,1,2,3,4,5,6,7, k512, w, 24);
  round_16_79(7,0,1,2,3,4,5,6, k512, w, 25);
  round_16_79(6,7,0,1,2,3,4,5, k512, w, 26);
  round_16_79(5,6,7,0,1,2,3,4, k512, w, 27);
  round_16_79(4,5,6,7,0,1,2,3, k512, w, 28);
  round_16_79(3,4,5,6,7,0,1,2, k512, w, 29);
  round_16_79(2,3,4,5,6,7,0,1, k512, w, 30);
  round_16_79(1,2,3,4,5,6,7,0, k512, w, 31);

  round_16_79(0,1,2,3,4,5,6,7, k512, w, 32);
  round_16_79(7,0,1,2,3,4,5,6, k512, w, 33);
  round_16_79(6,7,0,1,2,3,4,5, k512, w, 34);
  round_16_79(5,6,7,0,1,2,3,4, k512, w, 35);
  round_16_79(4,5,6,7,0,1,2,3, k512, w, 36);
  round_16_79(3,4,5,6,7,0,1,2, k512, w, 37);
  round_16_79(2,3,4,5,6,7,0,1, k512, w, 38);
  round_16_79(1,2,3,4,5,6,7,0, k512, w, 39);

  round_16_79(0,1,2,3,4,5,6,7, k512, w, 40);
  round_16_79(7,0,1,2,3,4,5,6, k512, w, 41);
  round_16_79(6,7,0,1,2,3,4,5, k512, w, 42);
  round_16_79(5,6,7,0,1,2,3,4, k512, w, 43);
  round_16_79(4,5,6,7,0,1,2,3, k512, w, 44);
  round_16_79(3,4,5,6,7,0,1,2, k512, w, 45);
  round_16_79(2,3,4,5,6,7,0,1, k512, w, 46);
  round_16_79(1,2,3,4,5,6,7,0, k512, w, 47);

  round_16_79(0,1,2,3,4,5,6,7, k512, w, 48);
  round_16_79(7,0,1,2,3,4,5,6, k512, w, 49);
  round_16_79(6,7,0,1,2,3,4,5, k512, w, 50);
  round_16_79(5,6,7,0,1,2,3,4, k512, w, 51);
  round_16_79(4,5,6,7,0,1,2,3, k512, w, 52);
  round_16_79(3,4,5,6,7,0,1,2, k512, w, 53);
  round_16_79(2,3,4,5,6,7,0,1, k512, w, 54);
  round_16_79(1,2,3,4,5,6,7,0, k512, w, 55);

  round_16_79(0,1,2,3,4,5,6,7, k512, w, 56);
  round_16_79(7,0,1,2,3,4,5,6, k512, w, 57);
  round_16_79(6,7,0,1,2,3,4,5, k512, w, 58);
  round_16_79(5,6,7,0,1,2,3,4, k512, w, 59);
  round_16_79(4,5,6,7,0,1,2,3, k512, w, 60);
  round_16_79(3,4,5,6,7,0,1,2, k512, w, 61);
  round_16_79(2,3,4,5,6,7,0,1, k512, w, 62);
  round_16_79(1,2,3,4,5,6,7,0, k512, w, 63);

  round_16_79(0,1,2,3,4,5,6,7, k512, w, 64);
  round_16_79(7,0,1,2,3,4,5,6, k512, w, 65);
  round_16_79(6,7,0,1,2,3,4,5, k512, w, 66);
  round_16_79(5,6,7,0,1,2,3,4, k512, w, 67);
  round_16_79(4,5,6,7,0,1,2,3, k512, w, 68);
  round_16_79(3,4,5,6,7,0,1,2, k512, w, 69);
  round_16_79(2,3,4,5,6,7,0,1, k512, w, 70);
  round_16_79(1,2,3,4,5,6,7,0, k512, w, 71);

  round_16_79(0,1,2,3,4,5,6,7, k512, w, 72);
  round_16_79(7,0,1,2,3,4,5,6, k512, w, 73);
  round_16_79(6,7,0,1,2,3,4,5, k512, w, 74);
  round_16_79(5,6,7,0,1,2,3,4, k512, w, 75);
  round_16_79(4,5,6,7,0,1,2,3, k512, w, 76);
  round_16_79(3,4,5,6,7,0,1,2, k512, w, 77);
  round_16_79(2,3,4,5,6,7,0,1, k512, w, 78);
  round_16_79(1,2,3,4,5,6,7,0, k512, w, 79);

  m_digest[0] += v0;
  m_digest[1] += v1;
  m_digest[2] += v2;
  m_digest[3] += v3;
  m_digest[4] += v4;
  m_digest[5] += v5;
  m_digest[6] += v6;
  m_digest[7] += v7;
}

#undef TEMPLATEM
#undef CLASSM

#undef round_0_15
#undef round_16_79
#undef bswap
#undef bswap64
#undef Ch
#undef Maj
#undef Rotr
#undef Shr
#undef wv
#undef Sigma0_256
#undef Sigma1_256
#undef sigma0_256
#undef sigma1_256
#undef Sigma0_512
#undef Sigma1_512
#undef sigma0_512
#undef sigma1_512

}
}

#endif
