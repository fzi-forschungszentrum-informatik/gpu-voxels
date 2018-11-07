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
#ifndef ICL_CORE_CRYPT_SHA2_H_INCLUDED
#define ICL_CORE_CRYPT_SHA2_H_INCLUDED

#include <icl_core/BaseTypes.h>

#include "icl_core_crypt/Sha2Impl.h"

namespace icl_core {
//! Contains cryptographic functions.
namespace crypt {

/*! An implementation of SHA-2, following the FIPS PUB 180-3 Secure
 *  Hash Standard (SHS).
 *
 *  Instead of instantiating this manually, use the convenience
 *  typedefs Sha224, Sha256, Sha384 or Sha512 instead.
 *
 *  Usage example (using Sha256):
 *
 *  \code
 *  Sha256 sha;
 *  sha.process("An example string.");
 *  sha.process("More data for the same digest.");
 *  sha.finalize();
 *  std::cout << sha.getHexDigest() << std::endl;
 *  \endcode
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
 *  \see Sha224, Sha256, Sha384, Sha512
 */
template <typename T, T t_h0, T t_h1, T t_h2, T t_h3, T t_h4, T t_h5, T t_h6, T t_h7, size_t t_len>
class Sha2 : public Sha2Impl<T, t_h0, t_h1, t_h2, t_h3, t_h4, t_h5, t_h6, t_h7, t_len>
{
public:
  typedef Sha2Impl<T, t_h0, t_h1, t_h2, t_h3, t_h4, t_h5, t_h6, t_h7, t_len> Impl;
  using Impl::cMESSAGE_BLOCK_SIZE;
  using Impl::cMESSAGE_PAD_POSITION;
  using Impl::clear;
  using Impl::getHexDigest;

  //! Initializes the internal state.
  Sha2();

  //! Processes the string \a data.
  Sha2& process(const char *data);
  //! Processes the string \a data.
  inline Sha2& process(const ::icl_core::String& data)
  { return process(data.c_str(), data.size()); }
  //! Processes \a size bytes from the buffer \a data.
  Sha2& process(const void *data, size_t size);

  //! Finalizes the message digest.
  Sha2& finalize();

protected:
  /*! Processes the first \a size bytes of the current buffer contents
   *  and finalizes the digest.
   */
  void finalizeBuffer(size_t size);

  using Impl::processBuffer;
  using Impl::m_digest;
  using Impl::m_message_size;
  using Impl::m_buffer;
  using Impl::m_buffer_fill;
};

}
}

#include "icl_core_crypt/Sha2.hpp"

namespace icl_core {
namespace crypt {

//! The SHA-224 hash algorithm.
typedef Sha2<uint32_t,
             0xc1059ed8ul, 0x367cd507ul, 0x3070dd17ul, 0xf70e5939ul,
             0xffc00b31ul, 0x68581511ul, 0x64f98fa7ul, 0xbefa4fa4ul, 7> Sha224;
//! The SHA-256 hash algorithm.
typedef Sha2<uint32_t,
             0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
             0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul, 8> Sha256;
//! The SHA-384 hash algorithm.
typedef Sha2<uint64_t,
             0xcbbb9d5dc1059ed8ull, 0x629a292a367cd507ull, 0x9159015a3070dd17ull, 0x152fecd8f70e5939ull,
             0x67332667ffc00b31ull, 0x8eb44a8768581511ull, 0xdb0c2e0d64f98fa7ull, 0x47b5481dbefa4fa4ull, 6> Sha384;
//! The SHA-512 hash algorithm.
typedef Sha2<uint64_t,
             0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
             0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull, 8> Sha512;

}
}

#endif
