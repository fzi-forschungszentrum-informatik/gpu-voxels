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
 * \author  Thomas Schamm <schamm@fzi.de>
 * \date    2010-04-08
 *
 * \brief   Contains icl_core::SchemeParser
 *
 * \b icl_core::SchemeParser
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_SCHEME_PARSER_H_INCLUDED
#define ICL_CORE_SCHEME_PARSER_H_INCLUDED

#include "icl_core/BaseTypes.h"
#include "icl_core/ImportExport.h"

#include <vector>

#include <boost/spirit/version.hpp>
#if SPIRIT_VERSION < 0x2000
# include <boost/spirit.hpp>
# define BOOST_SPIRIT_NAMESPACE boost::spirit
#else
// Undefine the Boost Spirit version macros because they are redefined
// in the include below!
# undef SPIRIT_VERSION
# undef SPIRIT_PIZZA_VERSION

# include <boost/spirit/include/classic.hpp>
# define BOOST_SPIRIT_NAMESPACE boost::spirit::classic
#endif

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core
{

struct Query
{
    String name;
    String value;
};

typedef std::vector<Query> QueryList;

enum SchemeType
{
    FileScheme,   //!< file:///path/file?key=value
    HttpScheme,   //!< http://domain.org/path/to/site.ext?key=value
    CameraScheme, //!< camera://source?key=value
    GpsScheme, //!< gps://type (gpsd/oxfs/...)
    OtherScheme
};

struct Scheme
{
    SchemeType scheme_type;
    String scheme_name;
    String specifier;
    String anchor;
    QueryList queries;
};

#ifdef __cplusplus
ICL_CORE_IMPORT_EXPORT std::ostream& operator<<(std::ostream& stream, Scheme const& scheme);
#endif // __cplusplus

/*! Defines an abstract function object, which is called while
 *  parsing.
 */
class AbstractFunctionObject
{
public:
    virtual ~AbstractFunctionObject()
    { }
    virtual void operator () (char const* str, char const* end) const = 0;
};

class SchemeFunction : public AbstractFunctionObject
{
public:
    virtual ~SchemeFunction()
    { }
    virtual void operator () (char const* str, char const* end) const;
    Scheme *m_scheme_handler;
};

class SpecifierFunction : public AbstractFunctionObject
{
public:
    virtual ~SpecifierFunction()
    { }
    virtual void operator () (char const* str, char const* end) const;
    Scheme *m_scheme_handler;
};

class AnchorFunction : public AbstractFunctionObject
{
public:
    virtual ~AnchorFunction()
    { }
    virtual void operator () (char const* str, char const* end) const;
    Scheme *m_scheme_handler;
};

class QueryKeyFunction : public AbstractFunctionObject
{
public:
    virtual ~QueryKeyFunction()
    { }
    virtual void operator () (char const* str, char const* end) const;
    QueryList *m_queries;
};

class QueryValueFunction : public AbstractFunctionObject
{
public:
    virtual ~QueryValueFunction()
    { }
    virtual void operator () (char const* str, char const* end) const;
    QueryList *m_queries;
};

/*! A SchemeParser object will parse a given string.  Yet, this class
 *  is not as generic as possible, meaning that parsing rules are
 *  defined internally, not given from outside.
 */
class ICL_CORE_IMPORT_EXPORT SchemeParser
{
public:
    //! Constructor.
    SchemeParser();

    //! Destructor.
    ~SchemeParser();

    /*! Parse the given string.
   *  \returns \c true if full parse was accomplished.
   */
    bool parseScheme(const String &str);

    /*! Return the information struct holding the details of the last
   *  parse.
   */
    const BOOST_SPIRIT_NAMESPACE::parse_info<> &getParseInfo() const;

    //! Returns the parser result.
    const icl_core::Scheme &getSchemeResult() const;

    /*! Static method, provided for convenience.
   */
    static bool parseScheme(const String &str, Scheme &scheme_handler,
			    BOOST_SPIRIT_NAMESPACE::parse_info<> &info);

    ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

    /*! Parse the given string.
   *  \deprecated Obsolete coding style.
   */
    ICL_CORE_VC_DEPRECATE_STYLE bool ParseScheme(const String &str) ICL_CORE_GCC_DEPRECATE_STYLE;

    /*! Return the information struct holding the details of the last
   *  parse.
   *  \deprecated Obsolete coding style.
   */
    ICL_CORE_VC_DEPRECATE_STYLE const BOOST_SPIRIT_NAMESPACE::parse_info<> &GetParseInfo() const ICL_CORE_GCC_DEPRECATE_STYLE;

    /*! Returns the parser result.
   *  \deprecated Obsolete coding style.
   */
    ICL_CORE_VC_DEPRECATE_STYLE const icl_core::Scheme &GetSchemeResult() const ICL_CORE_GCC_DEPRECATE_STYLE;

    /*! Static method, provided for convenience.
   *  \deprecated Obsolete coding style.
   */
    ICL_CORE_VC_DEPRECATE_STYLE static bool ParseScheme(const String &str, Scheme &scheme_handler,
							BOOST_SPIRIT_NAMESPACE::parse_info<> &info)
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
    /////////////////////////////////////////////////

private:
    BOOST_SPIRIT_NAMESPACE::parse_info<> m_info;

    icl_core::Scheme m_scheme;
};

} // namespace icl_core

#endif // _icl_core_SchemeParser_h_
