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
 * \author  Jan Oberlaender
 * \date    2012-11-29
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_VECTOR_WRAPPER_H_INCLUDED
#define ICL_CORE_VECTOR_WRAPPER_H_INCLUDED

/*! Passthrough typedefs for a vector datatype.  This macro is needed
 *  by other macros defined later on.  Do not use it directly.
 */
#define ICL_CORE_WRAP_VECTOR_HEADER(TWrapper, TVector, access, wrapped) \
  typedef TVector::value_type value_type;                               \
  typedef TVector::allocator_type allocator_type;                       \
  typedef TVector::size_type size_type;                                 \
  typedef TVector::difference_type difference_type;                     \
  typedef TVector::reference reference;                                 \
  typedef TVector::const_reference const_reference;                     \
  typedef TVector::pointer pointer;                                     \
  typedef TVector::const_pointer const_pointer;                         \
  typedef TVector::iterator iterator;                                   \
  typedef TVector::const_iterator const_iterator;                       \
  typedef TVector::reverse_iterator reverse_iterator;                   \
  typedef TVector::const_reverse_iterator const_reverse_iterator


/*! Passthrough typedefs for a template vector datatype (with a
 *  typename keyword).  This macro is needed by other macros defined
 *  later on.  Do not use it directly.
 */
#define ICL_CORE_WRAP_VECTOR_HEADER_TYPENAME(TWrapper, TVector, access, wrapped) \
  typedef typename TVector::value_type value_type;                      \
  typedef typename TVector::allocator_type allocator_type;              \
  typedef typename TVector::size_type size_type;                        \
  typedef typename TVector::difference_type difference_type;            \
  typedef typename TVector::reference reference;                        \
  typedef typename TVector::const_reference const_reference;            \
  typedef typename TVector::pointer pointer;                            \
  typedef typename TVector::const_pointer const_pointer;                \
  typedef typename TVector::iterator iterator;                          \
  typedef typename TVector::const_iterator const_iterator;              \
  typedef typename TVector::reverse_iterator reverse_iterator;          \
  typedef typename TVector::const_reverse_iterator const_reverse_iterator


/*! Wrapper which passes through all methods of the wrapped
 *  std::vector to the wrapping class.  This macro is needed by other
 *  macros defined later on.  Do not use it directly.
 */
#define ICL_CORE_WRAP_VECTOR_BODY(TWrapper, TVector, access, wrapped)   \
  void assign(size_type count, const value_type& value)                 \
  { wrapped.assign(count, value); }                                     \
  template <class TInputIterator>                                       \
  void assign(TInputIterator first, TInputIterator last)                \
  { wrapped.template assign<TInputIterator>(first, last); }                      \
  allocator_type get_allocator() const                                  \
  { return wrapped.get_allocator(); }                                   \
                                                                        \
  reference at(size_type pos) { return wrapped.at(pos); }               \
  const_reference at(size_type pos) const { return wrapped.at(pos); }   \
  reference operator [] (size_type pos) { return wrapped[pos]; }        \
  const_reference operator [] (size_type pos) const                     \
  { return wrapped[pos]; }                                              \
                                                                        \
  reference front() { return wrapped.front(); }                         \
  const_reference front() const { return wrapped.front(); }             \
  reference back() { return wrapped.back(); }                           \
  const_reference back() const { return wrapped.back(); }               \
  iterator begin() { return wrapped.begin(); }                          \
  const_iterator begin() const { return wrapped.begin(); }              \
  iterator end() { return wrapped.end(); }                              \
  const_iterator end() const { return wrapped.end(); }                  \
  reverse_iterator rbegin() { return wrapped.rbegin(); }                \
  const_reverse_iterator rbegin() const { return wrapped.rbegin(); }    \
  reverse_iterator rend() { return wrapped.rend(); }                    \
  const_reverse_iterator rend() const { return wrapped.rend(); }        \
                                                                        \
  bool empty() const { return wrapped.empty(); }                        \
  size_type size() const { return wrapped.size(); }                     \
  size_type max_size() const { return wrapped.max_size(); }             \
  void reserve(size_type size) { wrapped.reserve(size); }               \
  size_type capacity() const { return wrapped.capacity(); }             \
                                                                        \
  void clear() { wrapped.clear(); }                                     \
  iterator insert(iterator pos, const value_type& value)                \
  { return wrapped.insert(pos, value); }                                \
  void insert(iterator pos, size_type count, const value_type& value)   \
  { wrapped.insert(pos, count, value); }                                \
  template <class TInputIterator>                                       \
  void insert(iterator pos, TInputIterator first, TInputIterator last)  \
  { wrapped.template insert<TInputIterator>(pos, first, last); }                 \
                                                                        \
  iterator erase(iterator pos) { return wrapped.erase(pos); }           \
  iterator erase(iterator first, iterator last)                         \
  { return wrapped.erase(first, last); }                                \
  void push_back(const value_type& value) { wrapped.push_back(value); } \
  void pop_back() { wrapped.pop_back(); }                               \
  void resize(size_type count, value_type value = value_type())         \
  { wrapped.resize(count, value); }                                     \
  void swap(TWrapper& other) { wrapped.swap(other.wrapped); }           \
access:                                                                 \
  TVector wrapped


/*! Wrapper for the usual std::vector constructors.  This macro is
 *  needed by other macros defined later on.  Do not use it directly.
 */
#define ICL_CORE_WRAP_VECTOR_CTOR(TWrapper, TVector, access, wrapped)   \
  TWrapper(const TVector& other) : wrapped(other) { }                   \
  explicit TWrapper(const allocator_type& alloc = allocator_type())     \
    : wrapped(alloc)                                                    \
  { }                                                                   \
  explicit TWrapper(size_type count,                                    \
                    const value_type& value = value_type())             \
    : wrapped(count, value)                                             \
  { }                                                                   \
  template <class TInputIterator>                                       \
  TWrapper(TInputIterator first, TInputIterator last,                   \
           const allocator_type& alloc = allocator_type())              \
    : wrapped(first, last, alloc)                                       \
  { }


/*! Wrapper for the usual std::vector constructors, with initializer
 *  lists for custom members of the wrapping class.  This macro is
 *  needed by other macros defined later on.  Do not use it directly.
 */
#define ICL_CORE_WRAP_VECTOR_CTOR_INIT(TWrapper, TVector, access, wrapped, ...) \
  TWrapper(const TVector& other)                                        \
    : wrapped(other), __VA_ARGS__                                       \
  { }                                                                   \
  explicit TWrapper(const allocator_type& alloc = allocator_type())     \
    : wrapped(alloc), __VA_ARGS__                                       \
  { }                                                                   \
  explicit TWrapper(size_type count,                                    \
                    const value_type& value = value_type())             \
    : wrapped(count, value), __VA_ARGS__                                \
  { }                                                                   \
  template <class TInputIterator>                                       \
  TWrapper(TInputIterator first, TInputIterator last,                   \
           const allocator_type& alloc = allocator_type())              \
    : wrapped(first, last, alloc), __VA_ARGS__                          \
  { }                                                                   \
                                                                        \


/*! Helper macro to generate a thin wrapper around std::vector.  This
 *  defines all the typedefs and methods provided by std::vector and
 *  passes them on to the wrapped object.  The wrapped vector member
 *  variable is defined as well.
 *
 *  \note Use this version if your TVector does not depend on a
 *        template parameter of your wrapper class, and if you do not
 *        have any member variables of your own which need to be
 *        initialized in the generated constructors.
 *  \note If you define copy constructors and assignment operators,
 *        you are responsible for initializing the wrapped vector
 *        yourself.
 *
 *  \param TWrapper Typename of the wrapper class.
 *  \param TVector Full typename of the wrapped vector template
 *         instance.
 *  \param access Access level for the wrapped vector member variable
 *         (public, protected or private).
 *  \param wrapped Name of the wrapped vector member variable.
 */
#define ICL_CORE_WRAP_VECTOR(TWrapper, TVector, access, wrapped)        \
  ICL_CORE_WRAP_VECTOR_HEADER(TWrapper, TVector, acccess, wrapped);     \
  ICL_CORE_WRAP_VECTOR_CTOR(TWrapper, TVector, access, wrapped)         \
  ICL_CORE_WRAP_VECTOR_BODY(TWrapper, TVector, access, wrapped)


/*! Helper macro to generate a thin wrapper around std::vector<T>
 *  where T is a template parameter of the wrapping class.  This
 *  defines all the typedefs and methods provided by std::vector and
 *  passes them on to the wrapped object.  The wrapped vector member
 *  variable is defined as well.
 *
 *  \note Use this version if your TVector depends on a template
 *        parameter of your wrapper class, and if you do not have any
 *        member variables of your own which need to be initialized in
 *        the generated constructors.
 *  \note If you define copy constructors and assignment operators,
 *        you are responsible for initializing the wrapped vector
 *        yourself.
 *
 *  \param TWrapper Typename of the wrapper class.
 *  \param TVector Full typename of the wrapped vector template
 *         instance.
 *  \param access Access level for the wrapped vector member variable
 *         (public, protected or private).
 *  \param wrapped Name of the wrapped vector member variable.
 */
#define ICL_CORE_WRAP_VECTOR_TYPENAME(TWrapper, TVector, access, wrapped) \
  ICL_CORE_WRAP_VECTOR_HEADER_TYPENAME(TWrapper, TVector, acccess, wrapped); \
  ICL_CORE_WRAP_VECTOR_CTOR(TWrapper, TVector, access, wrapped)         \
  ICL_CORE_WRAP_VECTOR_BODY(TWrapper, TVector, access, wrapped)


/*! Helper macro to for a thin wrapper around std::vector, with custom
 *  member variable initializers.  This defines all the typedefs and
 *  methods provided by std::vector and passes them on to the wrapped
 *  object.  The wrapped vector member variable is defined as well.
 *
 *  \note Use this version if your TVector does not depend on a
 *        template parameter of your wrapper class, and if you have
 *        member variables of your own which need to be initialized in
 *        the generated constructors.
 *  \note If you define copy constructors and assignment operators,
 *        you are responsible for initializing the wrapped vector
 *        yourself.
 *
 *  \param TWrapper Typename of the wrapper class.
 *  \param TVector Full typename of the wrapped vector template
 *         instance.
 *  \param access Access level for the wrapped vector member variable
 *         (public, protected or private).
 *  \param wrapped Name of the wrapped vector member variable.
 *  \param ... Initializer list passed to all constructors (for custom
 *         member variables).
 */
#define ICL_CORE_WRAP_VECTOR_INIT(TWrapper, TVector, access, wrapped, ...) \
  ICL_CORE_WRAP_VECTOR_HEADER(TWrapper, TVector, access, wrapped);      \
  ICL_CORE_WRAP_VECTOR_CTOR_INIT(TWrapper, TVector, access, wrapped, __VA_ARGS__) \
  ICL_CORE_WRAP_VECTOR_BODY(TWrapper, TVector, access, wrapped)


/*! Helper macro to generate a thin wrapper around std::vector<T>
 *  where T is a template parameter of the wrapping class, with custom
 *  member variable initializers.  This defines all the typedefs and
 *  methods provided by std::vector and passes them on to the wrapped
 *  object.  The wrapped vector member variable is defined as well.
 *
 *  \note Use this version if your TVector depends on a template
 *        parameter of your wrapper class, and if you have member
 *        variables of your own which need to be initialized in the
 *        generated constructors.
 *  \note If you define copy constructors and assignment operators,
 *        you are responsible for initializing the wrapped vector
 *        yourself.
 *
 *  \param TWrapper Typename of the wrapper class.
 *  \param TVector Full typename of the wrapped vector template
 *         instance.
 *  \param access Access level for the wrapped vector member variable
 *         (public, protected or private).
 *  \param wrapped Name of the wrapped vector member variable.
 *  \param ... Initializer list passed to all constructors (for custom
 *         member variables).
 */
#define ICL_CORE_WRAP_VECTOR_TYPENAME_INIT(TWrapper, TVector, access, wrapped, ...) \
  ICL_CORE_WRAP_VECTOR_HEADER_TYPENAME(TWrapper, TVector, access, wrapped); \
  ICL_CORE_WRAP_VECTOR_CTOR_INIT(TWrapper, TVector, access, wrapped, __VA_ARGS__) \
  ICL_CORE_WRAP_VECTOR_BODY(TWrapper, TVector, access, wrapped)



#endif // ICL_CORE_VECTOR_WRAPPER_H_INCLUDED
