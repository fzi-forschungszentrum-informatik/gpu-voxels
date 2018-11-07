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
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-03-12
 *
 * \brief   Contains Getopt.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_GETOPT_PARSER_H_INCLUDED
#define ICL_CORE_CONFIG_GETOPT_PARSER_H_INCLUDED

#include "icl_core/BaseTypes.h"
#include "icl_core/Deprecate.h"
#include "icl_core/List.h"
#include "icl_core/Map.h"
#include "icl_core/Vector.h"
#include "icl_core_config/ImportExport.h"
#include "icl_core_config/GetoptParameter.h"
#include "icl_core_config/GetoptPositionalParameter.h"

namespace icl_core {
namespace config {

/*! \brief Handles commandline parameters.
 *
 *  Getopt reads all commandline parameters and extracts commandline
 *  options (both key/value and simple ones).  All parameters, which
 *  which were not identified as option parameters can be accessed as
 *  non-option parameters.
 *
 *  Commandline options have to be registered with calls to
 *  AddParameter(). Then parsing the commandline is initialized by a
 *  call to Initialize() with the commandline as arguments.
 *
 *  Getopt is implemented as a singleton so that it can be used from
 *  everywhere after it has been initialized once.
 */
class ICL_CORE_CONFIG_IMPORT_EXPORT Getopt
{
public:
  enum ParameterRegistrationCheck
  {
    ePRC_Strict, //!< all options have to be registered
    ePRC_Relaxed //!< options not registered are ignored
  };
  typedef ICL_CORE_VC_DEPRECATE ParameterRegistrationCheck tParameterRegistrationCheck ICL_CORE_GCC_DEPRECATE;

  enum CommandLineCleaning
  {
    eCLC_None,   //!< command line options are left untouched
    eCLC_Cleanup //!< known command line options are removed
  };
  typedef ICL_CORE_VC_DEPRECATE CommandLineCleaning tCommandLineCleaning ICL_CORE_GCC_DEPRECATE;

  struct KeyValue
  {
    KeyValue(const icl_core::String& key, const icl_core::String& value)
      : m_key(key),
        m_value(value)
    { }

    icl_core::String m_key;
    icl_core::String m_value;
  };
  typedef icl_core::List<KeyValue> KeyValueList;

  /*! Get the singleton instance.
   */
  static Getopt& instance();

  /*! Active extra command parameters. They are delimited from regular
   *  commandline parameters using the \a delimiter and run from there
   *  to the end of the commandline.
   */
  void activateExtraCmdParams(const icl_core::String& delimiter = "--");

  /*! Adds a parameter to the list of commandline options.
   */
  void addParameter(const GetoptParameter& parameter);

  /*! Adds a list of parameters to the list of commandline options.
   */
  void addParameter(const GetoptParameterList& parameters);

  /*! Adds a positional parameter to the list of commandline options.
   */
  void addParameter(const GetoptPositionalParameter& parameter);

  /*! Adds a list of positional parameters to the list of commandline
   *  options.
   */
  void addParameter(const GetoptPositionalParameterList& parameters);

  /*! Initializes Getopt with a commandline.
   *
   *  \deprecated Please use Initialize(argc, argv, eCLC_None) or
   *  Initialize(argc, argv, eCLC_Cleanup) instead which provides the
   *  same functionality.
   */
  bool initialize(int& argc, char *argv[], bool remove_read_arguments);

  /*! Initializes Getopt with a commandline.
   *
   * \param argc Number of command line options in argv
   * \param argv Command line options
   * \param cleanup Can be eCLC_None to leave argc and argv untouched
   *        or eCLC_Cleanup to remove known options from argv and
   *        decrease argc appropriately
   * \param registration_check When encountering a not registered
   *        command line option, the value ePRC_Strict causes the
   *        initialization to fail, while ePRC_Relaxed accepts it
   *        anyway
   */
  bool initialize(int& argc, char *argv[], CommandLineCleaning cleanup = eCLC_None,
                  ParameterRegistrationCheck registration_check = ePRC_Strict);

  /*! Returns \c true if Getopt has already been initialized.
   */
  bool isInitialized() const { return m_initialized; }

  //! Get the original argc
  int &argc();

  //! Get the original argv
  char** argv() const;

  /*! Get the extra command parameter at \a index.
   */
  icl_core::String extraCmdParam(size_t index) const;

  /*! Get the number of extra command parameters.
   */
  size_t extraCmdParamCount() const;

  /*! Get the value of the commandline option \a name.
   *
   *  \returns An empty string if the option has not been set,
   *           otherwise the value of the option.
   */
  icl_core::String paramOpt(const icl_core::String& name) const;

  /*! Checks if the option \a name is present.
   */
  bool paramOptPresent(const icl_core::String& name) const;

  /*! Get the list of defined suffixes for the specified \a prefix.
   */
  KeyValueList paramPrefixOpt(const icl_core::String& prefix) const;

  /*! Check in a prefix option is present.
   */
  bool paramPrefixOptPresent(const icl_core::String& prefix) const;

  /*! Get the non-option parameter at the specified \a index.
   *
   *  \returns An empty string if no such parameter exists.
   */
  icl_core::String paramNonOpt(size_t index) const;

  /*! Get the number of non-option parameters.
   */
  size_t paramNonOptCount() const;

  /*! Get the program name.
   */
  icl_core::String programName() const;

  /*! Get the program version.
   */
  icl_core::String programVersion() const;

  /*! Set the program version.
   */
  void setProgramVersion(icl_core::String const & version);

  /*! Set the program description, a short string describing the program's purpose.
   */
  void setProgramDescription(icl_core::String const & description);

  /*! Get the program description.
   */
  icl_core::String programDescription() const;

  /*! Prints the help text.
   */
  void printHelp() const;

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Get the singleton instance.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE static Getopt& Instance() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Active extra command parameters.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void ActivateExtraCmdParams(const icl_core::String& delimiter = "--")
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Adds a parameter to the list of commandline options.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AddParameter(const GetoptParameter& parameter)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Adds a list of parameters to the list of commandline options.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AddParameter(const GetoptParameterList& parameters)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Initializes Getopt with a commandline.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Initialize(int& argc, char *argv[], bool remove_read_arguments)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Initializes Getopt with a commandline.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Initialize(int& argc, char *argv[],
                                              CommandLineCleaning cleanup = eCLC_None,
                                              ParameterRegistrationCheck registration_check = ePRC_Strict)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns \c true if Getopt has already been initialized.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsInitialized() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the extra command parameter at \a index.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String ExtraCmdParam(size_t index) const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the number of extra command parameters.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE size_t ExtraCmdParamCount() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the value of the commandline option \a name.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String ParamOpt(const icl_core::String& name) const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Checks if the option \a name is present.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool ParamOptPresent(const icl_core::String& name) const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the list of defined suffixes for the specified \a prefix.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE KeyValueList ParamPrefixOpt(const icl_core::String& prefix) const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Check in a prefix option is present.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool ParamPrefixOptPresent(const icl_core::String& prefix) const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the non-option parameter at the specified \a index.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String ParamNonOpt(size_t index) const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the number of non-option parameters.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE size_t ParamNonOptCount() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the program name.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String ProgramName() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Prints the help text.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void PrintHelp() const ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  Getopt();

  typedef icl_core::Map<icl_core::String, GetoptParameter> ParameterMap;
  ParameterMap m_parameters;
  ParameterMap m_prefix_parameters;
  ParameterMap m_short_parameters;
  ParameterMap m_short_prefix_parameters;
  GetoptPositionalParameterList m_required_positional_parameters;
  GetoptPositionalParameterList m_optional_positional_parameters;
  bool m_extra_cmd_param_activated;
  icl_core::String m_extra_cmd_param_delimiter;

  bool m_initialized;

  int m_argc;
  char** m_argv;
  icl_core::String m_program_name;
  icl_core::String m_program_version;
  icl_core::String m_program_description;
  icl_core::Vector<icl_core::String> m_param_non_opt;
  icl_core::Map<icl_core::String, icl_core::String> m_param_opt;
  icl_core::Map<icl_core::String, KeyValueList> m_prefix_param_opt;
  icl_core::Vector<icl_core::String> m_extra_cmd_param;
};

}
}
#endif
