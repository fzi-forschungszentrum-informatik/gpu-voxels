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
#ifndef ICL_CORE_CRYPT_SHA2_IMPL_H_INCLUDED
#define ICL_CORE_CRYPT_SHA2_IMPL_H_INCLUDED

#include <icl_core/BaseTypes.h>

namespace icl_core {
namespace crypt {

/*! The implementation internals for SHA-2, following the FIPS PUB
 *  180-3 Secure Hash Standard (SHS).  The implementation works for
 *  32-bit (SHA-224, SHA-256) and 64-bit (SHA-384, SHA-512) data
 *  types.  The algorithmic differences in the 64-bit version are
 *  captured by a template specialization.
 *
 *  \tparam T The type of the message words (uint32_t or uint64_t).
 *  \tparam t_h0 The initial hash value, word 0.
 *  \tparam t_h1 The initial hash value, word 1.
 *  \tparam t_h2 The initial hash value, word 2.
 *  \tparam t_h3 The initial hash value, word 3.
 *  \tparam t_h4 The initial hash value, word 4.
 *  \tparam t_h5 The initial hash value, word 5.
 *  \tparam t_h6 The initial hash value, word 6.
 *  \tparam t_h7 The initial hash value, word 7.
 *  \tparam t_len The length, in words, of the message digest.  This
 *          may be at most 8 (i.e., all eight words of the digest are
 *          used).  Only the first \a t_len words of the digest are
 *          returned otherwise.
 */
template <typename T, T t_h0, T t_h1, T t_h2, T t_h3, T t_h4, T t_h5, T t_h6, T t_h7, size_t t_len>
class Sha2Impl
{
public:
  //! The message block size in bytes.
  static const size_t cMESSAGE_BLOCK_SIZE = 64;
  /*! The position up to which the last block is padded (the remaining
   *  bytes hold the message length).
   */
  static const size_t cMESSAGE_PAD_POSITION = 56;

  //! Initializes the internal state.
  Sha2Impl();

  //! Clears the internal state.  A new hash can then be calculated.
  void clear();

  //! Returns the message digest as a hex string.
  ::icl_core::String getHexDigest() const;

protected:
  /*! Processes the current buffer contents and adds them to the
   *  message digest.
   */
  void processBuffer();

  //! The whole message digest.
  T m_digest[8];
  //! The size of the message so far, in bytes.
  uint64_t m_message_size;
  /*! The internal message buffer.  Stores the data for one message
   *  block.
   */
  uint8_t m_buffer[cMESSAGE_BLOCK_SIZE];
  //! The amount of bytes currently stored in the internal buffer.
  size_t m_buffer_fill;
};

/*! Template specialization for SHA-2 implementation internals for
 *  64-bit word size.
 *  \tparam t_h0 The initial hash value, word 0.
 *  \tparam t_h1 The initial hash value, word 1.
 *  \tparam t_h2 The initial hash value, word 2.
 *  \tparam t_h3 The initial hash value, word 3.
 *  \tparam t_h4 The initial hash value, word 4.
 *  \tparam t_h5 The initial hash value, word 5.
 *  \tparam t_h6 The initial hash value, word 6.
 *  \tparam t_h7 The initial hash value, word 7.
 *  \tparam t_len The length, in words, of the message digest.  This
 *          may be at most 8 (i.e., all eight words of the digest are
 *          used).  Only the first \a t_len words of the digest are
 *          returned otherwise.
 */
template <uint64_t t_h0, uint64_t t_h1, uint64_t t_h2, uint64_t t_h3, uint64_t t_h4, uint64_t t_h5, uint64_t t_h6, uint64_t t_h7, size_t t_len>
class Sha2Impl<uint64_t, t_h0, t_h1, t_h2, t_h3, t_h4, t_h5, t_h6, t_h7, t_len>
{
public:
  //! The message block size in bytes.
  static const size_t cMESSAGE_BLOCK_SIZE = 128;
  /*! The position up to which the last block is padded (the remaining
   *  bytes hold the message length).
   */
  static const size_t cMESSAGE_PAD_POSITION = 112;

  //! Initializes the internal state.
  Sha2Impl();

  //! Clears the internal state.  A new hash can then be calculated.
  void clear();

  //! Returns the message digest as a hex string.
  ::icl_core::String getHexDigest() const;

protected:
  /*! Processes the current buffer contents and adds them to the
   *  message digest.
   */
  void processBuffer();

  /*! Processes the first \a size bytes of the current buffer contents
   *  and finalizes the digest.
   */
  void finalizeBuffer(size_t size);

  //! The whole message digest.
  uint64_t m_digest[8];
  //! The size of the message so far, in bytes.
  uint64_t m_message_size;
  /*! The internal message buffer.  Stores the data for one message
   *  block.
   */
  uint8_t m_buffer[cMESSAGE_BLOCK_SIZE];
  //! The amount of bytes currently stored in the internal buffer.
  size_t m_buffer_fill;
};

}
}

#include "icl_core_crypt/Sha2Impl.hpp"

#endif
