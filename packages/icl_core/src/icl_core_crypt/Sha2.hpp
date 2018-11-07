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
#ifndef ICL_CORE_CRYPT_SHA2_HPP_INCLUDED
#define ICL_CORE_CRYPT_SHA2_HPP_INCLUDED

#include <sstream>
#include <iomanip>
#include <string.h>

namespace icl_core {
namespace crypt {

#define TEMPLATEM template <typename T, T t_h0, T t_h1, T t_h2, T t_h3, T t_h4, T t_h5, T t_h6, T t_h7, size_t t_len>
#define CLASSM Sha2<T, t_h0, t_h1, t_h2, t_h3, t_h4, t_h5, t_h6, t_h7, t_len>
#define IMPLM Sha2Impl<T, t_h0, t_h1, t_h2, t_h3, t_h4, t_h5, t_h6, t_h7, t_len>

TEMPLATEM
CLASSM::Sha2()
  : IMPLM()
{
}

TEMPLATEM
CLASSM& CLASSM::process(const char *data)
{
  for (; *data != 0; ++data)
  {
    if (m_buffer_fill == cMESSAGE_BLOCK_SIZE)
    {
      processBuffer();
      m_buffer_fill = 0;
    }
    m_buffer[m_buffer_fill++] = *data;
    ++m_message_size;
  }
  return *this;
}

TEMPLATEM
CLASSM& CLASSM::process(const void *data, size_t size)
{
  const uint8_t *ptr = reinterpret_cast<const uint8_t *>(data);
  size_t rest = size;

  // Fill the buffer completely as many times as possible.
  while (rest >= cMESSAGE_BLOCK_SIZE-m_buffer_fill)
  {
    size_t amount = cMESSAGE_BLOCK_SIZE-m_buffer_fill;
    ::memcpy(&m_buffer[m_buffer_fill], ptr, amount);
    rest -= amount;
    processBuffer();
    ptr += amount;
    m_message_size += amount;
    m_buffer_fill = 0;
  }
  // Partially fill the buffer using the remaining data.
  ::memcpy(&m_buffer[m_buffer_fill], ptr, rest);
  m_message_size += rest;
  m_buffer_fill += rest;
  return *this;
}

TEMPLATEM
CLASSM& CLASSM::finalize()
{
  finalizeBuffer(m_buffer_fill);
  return *this;
}

#define bswap64(i) ((i) >> 56 |                                 \
                    (((i) >> 40) & 0x000000000000ff00ull) |     \
                    (((i) >> 24) & 0x0000000000ff0000ull) |     \
                    (((i) >>  8) & 0x00000000ff000000ull) |     \
                    (((i) <<  8) & 0x000000ff00000000ull) |     \
                    (((i) << 24) & 0x0000ff0000000000ull) |     \
                    (((i) << 40) & 0x00ff000000000000ull) |     \
                    (i) << 56)                                  \

TEMPLATEM
void CLASSM::finalizeBuffer(size_t size)
{
  uint64_t message_size_bits = m_message_size*8;

  // Always pad a "1" bit (i.e., 0x80).
  if (size < cMESSAGE_BLOCK_SIZE)
  {
    m_buffer[size++] = 0x80;
  }
  else
  {
    // Buffer is full, process first and then add the 0x80
    processBuffer();
    m_buffer[0] = 0x80;
    size = 1;
  }

  // Now pad to the padding position.
  if (size <= cMESSAGE_PAD_POSITION)
  {
    for (size_t i = size; i < cMESSAGE_BLOCK_SIZE-8; ++i)
    {
      m_buffer[i] = 0;
    }
    m_buffer[cMESSAGE_BLOCK_SIZE-8] = uint8_t((message_size_bits >> 56) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-7] = uint8_t((message_size_bits >> 48) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-6] = uint8_t((message_size_bits >> 40) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-5] = uint8_t((message_size_bits >> 32) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-4] = uint8_t((message_size_bits >> 24) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-3] = uint8_t((message_size_bits >> 16) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-2] = uint8_t((message_size_bits >>  8) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-1] = uint8_t((message_size_bits      ) & 0xff);
    processBuffer();
  }
  else
  {
    // Pad buffer first and process.  The message size goes into the
    // next block.
    for (size_t i = size; i < cMESSAGE_BLOCK_SIZE; ++i)
    {
      m_buffer[i] = 0;
    }
    processBuffer();

    // Pad the next block.
    for (size_t i = 0; i < cMESSAGE_BLOCK_SIZE-8; ++i)
    {
      m_buffer[i] = 0;
    }
    m_buffer[cMESSAGE_BLOCK_SIZE-8] = uint8_t((message_size_bits >> 56) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-7] = uint8_t((message_size_bits >> 48) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-6] = uint8_t((message_size_bits >> 40) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-5] = uint8_t((message_size_bits >> 32) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-4] = uint8_t((message_size_bits >> 24) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-3] = uint8_t((message_size_bits >> 16) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-2] = uint8_t((message_size_bits >>  8) & 0xff);
    m_buffer[cMESSAGE_BLOCK_SIZE-1] = uint8_t((message_size_bits      ) & 0xff);
    processBuffer();
  }
}

#undef bswap64

#undef TEMPLATEM
#undef CLASSM
#undef IMPLM

}
}

#endif
