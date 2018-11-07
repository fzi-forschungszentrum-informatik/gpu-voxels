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
 */
//----------------------------------------------------------------------
#include <iostream>
#include <stdlib.h>
#include <boost/regex.hpp>
#include <boost/foreach.hpp>

#include "icl_core/List.h"
#include "icl_core_config/GetoptParser.h"

namespace icl_core {
namespace config {

Getopt& Getopt::instance()
{
  static Getopt instance;
  return instance;
}

void Getopt::activateExtraCmdParams(const icl_core::String& delimiter)
{
  m_extra_cmd_param_activated = true;
  m_extra_cmd_param_delimiter = delimiter;
}

void Getopt::addParameter(const GetoptParameter& parameter)
{
  if (parameter.isPrefixOption())
  {
    m_prefix_parameters.insert(std::make_pair(parameter.option(), parameter));
    if (parameter.shortOption() != "")
    {
      m_short_prefix_parameters.insert(std::make_pair(parameter.shortOption(), parameter));
    }
  }
  else
  {
    m_parameters.insert(std::make_pair(parameter.option(), parameter));
    if (parameter.shortOption() != "")
    {
      m_short_parameters.insert(std::make_pair(parameter.shortOption(), parameter));
    }
  }
}

void Getopt::addParameter(const GetoptParameterList& parameters)
{
  for (GetoptParameterList::const_iterator opt_it = parameters.begin();
       opt_it != parameters.end(); ++opt_it)
  {
    addParameter(*opt_it);
  }
}

void Getopt::addParameter(const GetoptPositionalParameter &parameter)
{
  if (parameter.isOptional())
  {
    m_optional_positional_parameters.push_back(parameter);
  }
  else
  {
    m_required_positional_parameters.push_back(parameter);
  }
}

void Getopt::addParameter(const GetoptPositionalParameterList &parameters)
{
  for (GetoptPositionalParameterList::const_iterator opt_it = parameters.begin();
       opt_it != parameters.end(); ++opt_it)
  {
    addParameter(*opt_it);
  }
}

bool Getopt::initialize(int& argc, char *argv[], bool remove_read_arguments)
{
  return initialize(argc, argv, remove_read_arguments ? eCLC_Cleanup : eCLC_None, ePRC_Strict);
}

bool Getopt::initialize(int& argc, char *argv[], CommandLineCleaning cleanup,
                        ParameterRegistrationCheck registration_check)
{
  if (argc == 0)
  {
    return false;
  }

  if (isInitialized())
  {
    std::cerr << "GETOPT WARNING: The commandline option framework is already initialized!" << std::endl;
    return true;
  }

  // Store the full argc and argv
  m_argc = argc;
  m_argv = argv;


  // Store the program name.
  m_program_name = argv[0];

  // Store all parameters in a temporary list.
  icl_core::List<icl_core::String> arguments;
  for (int index = 1; index < argc; ++index)
  {
    arguments.push_back(argv[index]);
  }

  // Scan the commandline parameters and check for
  // registered options.
  size_t positional_parameters_counter = 0;
  bool extra_cmd_params_reached = false;
  boost::regex long_opt_regex("--([^-][^=]*)(=(.*))?");
  boost::regex short_opt_regex("-([^-].*)");
  boost::smatch mres;
  for (icl_core::List<icl_core::String>::const_iterator arg_it = arguments.begin();
       arg_it != arguments.end(); ++arg_it)
  {
    if (extra_cmd_params_reached)
    {
      m_extra_cmd_param.push_back(*arg_it);
    }
    else if (boost::regex_match(*arg_it, mres, long_opt_regex))
    {
      // Found a long option parameter!
      icl_core::String name = mres[1];
      ParameterMap::const_iterator find_it = m_parameters.find(name);
      if (find_it != m_parameters.end())
      {
        if (find_it->second.hasValue())
        {
          // According to the regular expression the value has to be
          // the 3rd (and last) match result.
          if (mres.size() == 4)
          {
            m_param_opt[name] = mres[3];
          }
          else
          {
            std::cerr << "Found option " << *arg_it << " but the value is missing." << std::endl;
            printHelp();
            return false;
          }
        }
        else
        {
          m_param_opt[name] = "yes";
        }
      }
      else
      {
        // Parameter not found in the list of configured parameters.
        // Check if a matching prefix option has been registered.
        bool found = false;
        boost::smatch prefix_res;
        for (ParameterMap::const_iterator prefix_it = m_prefix_parameters.begin();
             !found && prefix_it != m_prefix_parameters.end(); ++prefix_it)
        {
          if (boost::regex_match(name, prefix_res, boost::regex(prefix_it->first + "(.*)")))
          {
            found = true;

            if (prefix_it->second.hasValue())
            {
              if (mres.size() == 4)
              {
                m_prefix_param_opt[prefix_it->first].push_back(KeyValue(prefix_res[1], mres[3]));
              }
              else
              {
                std::cerr << "Found prefix option " << name << " but the value is missing." << std::endl;
                printHelp();
                return false;
              }
            }
            else
            {
              m_prefix_param_opt[prefix_it->first].push_back(KeyValue(prefix_res[1], "yes"));
            }
          }
        }

        // Also not a prefix option!
        if (!found)
        {
          if (registration_check == ePRC_Strict)
          {
            std::cerr << "Found unknown option " << *arg_it << "." << std::endl;
            printHelp();
            return false;
          }
          else
          {
            m_param_non_opt.push_back(*arg_it);
          }
        }
      }
    }
    else if (boost::regex_match(*arg_it, mres, short_opt_regex))
    {
      // Found a short option parameter!
      icl_core::String name = mres[1];
      ParameterMap::const_iterator find_it = m_short_parameters.find(name);
      if (find_it != m_short_parameters.end())
      {
        if (find_it->second.hasValue())
        {
          // The value is the next commandline argument.
          ++arg_it;
          if (arg_it == arguments.end())
          {
            std::cerr << "Found option -" << name << " but the value is missing." << std::endl;
            printHelp();
            return false;
          }
          else
          {
            m_param_opt[find_it->second.option()] = *arg_it;
          }
        }
        else
        {
          m_param_opt[find_it->second.option()] = "yes";
        }
      }
      else
      {
        // Parameter not found in the list of configured parameters.
        // Check if a matching prefix option has been registered.
        bool found = false;
        boost::smatch prefix_res;
        for (ParameterMap::const_iterator prefix_it = m_short_prefix_parameters.begin();
             !found && prefix_it != m_short_prefix_parameters.end(); ++prefix_it)
        {
          if (boost::regex_match(name, prefix_res, boost::regex(prefix_it->first + "(.*)")))
          {
            found = true;

            if (prefix_it->second.hasValue())
            {
              // The value is the next commandline argument.
              ++arg_it;
              if (arg_it == arguments.end())
              {
                std::cerr << "Found prefix option " << name << " but the value is missing." << std::endl;
                printHelp();
                return false;
              }
              else
              {
                m_prefix_param_opt[prefix_it->second.option()].push_back(KeyValue(prefix_res[1], *arg_it));
              }
            }
            else
            {
              m_prefix_param_opt[prefix_it->second.option()].push_back(KeyValue(prefix_res[1], "yes"));
            }
          }
        }

        // Also not a prefix option!
        if (!found)
        {
          if (registration_check == ePRC_Strict)
          {
            std::cerr << "Found unknown option " << *arg_it << "." << std::endl;
            printHelp();
            return false;
          }
          else
          {
            m_param_non_opt.push_back(*arg_it);
          }
        }
      }
    }
    else if (m_extra_cmd_param_activated && *arg_it == m_extra_cmd_param_delimiter)
    {
      extra_cmd_params_reached = true;
    }
    else if (positional_parameters_counter < m_required_positional_parameters.size())
    {
      // Found a required positional parameter
      const GetoptPositionalParameter& param = m_required_positional_parameters[positional_parameters_counter];
      m_param_opt[param.name()] = *arg_it;
      positional_parameters_counter++;
    }
    else if (positional_parameters_counter < m_required_positional_parameters.size() + m_optional_positional_parameters.size())
    {
      // Found an optional positional parameter
      const GetoptPositionalParameter& param = m_optional_positional_parameters[positional_parameters_counter - m_required_positional_parameters.size()];
      m_param_opt[param.name()] = *arg_it;
      positional_parameters_counter++;
    }
    else if (positional_parameters_counter >= m_required_positional_parameters.size() + m_optional_positional_parameters.size())
    {
      /*! \note this would be nice but breaks backwards compatibility
       *  where people use ePRC_Strict but want to use unregistered
       *  positional parameters.
       */
//      if (registration_check == ePRC_Strict)
//      {
//        std::cerr << "Found unknown positional parameter \"" << *arg_it << "\" and registration_check is ePRC_Strict. Aborting." << std::endl;
//        printHelp();
//        return false;
//      }
//      else
      {
        m_param_non_opt.push_back(*arg_it);
      }
    }
  }

  // Check if all required positional parameters are given
  if (positional_parameters_counter < m_required_positional_parameters.size())
  {
    std::cerr << "Not all required parameters are given. Aborting." << std::endl;
    printHelp();
    exit(0);
  }

  // Check if the help text has to be printed.
  if (m_param_opt.find("help") != m_param_opt.end())
  {
    printHelp();
    exit(0);
  }

  // Remove all option parameters from the "real" commandline.
  if (cleanup == eCLC_Cleanup)
  {
    int check = 1;
    while (check < argc)
    {
      icl_core::Vector<icl_core::String>::const_iterator find_it =
        std::find(m_param_non_opt.begin(), m_param_non_opt.end(), icl_core::String(argv[check]));
      if (find_it == m_param_non_opt.end())
      {
        for (int move = check + 1; move < argc; ++move)
        {
          argv[move - 1] = argv[move];
        }
        --argc;
      }
      else
      {
        ++check;
      }
    }
  }

  return true;
}

int& Getopt::argc()
{
  return m_argc;
}

char **Getopt::argv() const
{
  return m_argv;
}

icl_core::String Getopt::extraCmdParam(size_t index) const
{
  return m_extra_cmd_param[index];
}

size_t Getopt::extraCmdParamCount() const
{
  return m_extra_cmd_param.size();
}

icl_core::String Getopt::paramOpt(const icl_core::String& name) const
{
  icl_core::Map<icl_core::String, icl_core::String>::const_iterator find_it = m_param_opt.find(name);
  if (find_it == m_param_opt.end())
  {
    return "";
  }
  else
  {
    return find_it->second;
  }
}

bool Getopt::paramOptPresent(const icl_core::String& name) const
{
  return m_param_opt.find(name) != m_param_opt.end();
}

Getopt::KeyValueList Getopt::paramPrefixOpt(const icl_core::String& prefix) const
{
  icl_core::Map<icl_core::String, KeyValueList>::const_iterator find_it = m_prefix_param_opt.find(prefix);
  if (find_it == m_prefix_param_opt.end())
  {
    return KeyValueList();
  }
  else
  {
    return find_it->second;
  }
}

bool Getopt::paramPrefixOptPresent(const icl_core::String& prefix) const
{
  return m_prefix_param_opt.find(prefix) != m_prefix_param_opt.end();
}

icl_core::String Getopt::paramNonOpt(size_t index) const
{
  if (index < m_param_non_opt.size())
  {
    return m_param_non_opt.at(index);
  }
  else
  {
    return "";
  }
}

size_t Getopt::paramNonOptCount() const
{
  return m_param_non_opt.size();
}

icl_core::String Getopt::programName() const
{
  return m_program_name;
}

icl_core::String Getopt::programVersion() const
{
  return m_program_version;
}

void Getopt::setProgramVersion(icl_core::String const & version)
{
  m_program_version = version;
}

icl_core::String Getopt::programDescription() const
{
  return m_program_description;
}

void Getopt::setProgramDescription(icl_core::String const & description)
{
  m_program_description = description;
}

void Getopt::printHelp() const
{
  // prepare list of all positional parameters
  GetoptPositionalParameterList positional_parameters = m_required_positional_parameters;
  std::copy(m_optional_positional_parameters.begin(), m_optional_positional_parameters.end(), std::back_inserter(positional_parameters));

  std::cerr << programName();
  if (programVersion() != "")
  {
    std::cerr << " (version " << programVersion() << ")";
  }
  std::cerr << std::endl << std::endl;

  std::cerr << "Usage: ";
  std::cerr << programName();

  std::cerr << " [OPTIONS...]";

  BOOST_FOREACH(const GetoptPositionalParameter param, positional_parameters)
  {
    if (param.isOptional())
    {
      std::cerr  << " [<" << param.name() << ">]";
    }
    else
    {
      std::cerr  << " <" << param.name() << ">";
    }
  }

  std::cerr << std::endl << std::endl << programDescription() << std::endl << std::endl;

  if (positional_parameters.size() > 0 )
  {
    std::cerr << "Positional Parameters:" << std::endl;

    BOOST_FOREACH(const GetoptPositionalParameter param, positional_parameters)
    {
      std::cerr << "  <" << param.name() << ">" << ":" << std::endl << "\t"
                << boost::regex_replace(param.help(), boost::regex("\\n"), "\n\t")
                << std::endl;
    }
    std::cerr << std::endl;
  }

  for (int i=0; i<2; ++i)
  {
    std::cerr << (i == 0 ? "Generic options:" : "Options:") << std::endl;
    for (ParameterMap::const_iterator it = m_parameters.begin(); it != m_parameters.end(); ++it)
    {
      bool const is_generic =
          it->second.option() == "configfile"  ||
          it->second.option() == "dump-config" ||
          it->second.option() == "help"        ||
          it->second.option() == "log-level"   ||
          it->second.option() == "quick-debug";
      if (!i==is_generic)
      {
        std::cerr << "  ";
        // Short option.
        if (it->second.shortOption() != "")
        {
          std::cerr << "-" << it->second.shortOption();
          if (it->second.hasValue())
          {
            std::cerr << " <value>";
          }
          std::cerr << ", ";
        }

        // Long option.
        std::cerr << "--" << it->second.option();
        if (it->second.hasValue())
        {
          std::cerr << "=<value>";
        }

        // Help text
        std::cerr << ":" << std::endl << "\t"
                  << boost::regex_replace(it->second.help(), boost::regex("\\n"), "\n\t")
                  << std::endl;
      }
    }
    std::cerr << std::endl;
  }
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Get the singleton instance.
 *  \deprecated Obsolete coding style.
 */
Getopt& Getopt::Instance()
{
  return instance();
}

/*! Active extra command parameters.
 *  \deprecated Obsolete coding style.
 */
void Getopt::ActivateExtraCmdParams(const icl_core::String& delimiter)
{
  activateExtraCmdParams(delimiter);
}

/*! Adds a parameter to the list of commandline options.
 *  \deprecated Obsolete coding style.
 */
void Getopt::AddParameter(const GetoptParameter& parameter)
{
  addParameter(parameter);
}

/*! Adds a list of parameters to the list of commandline options.
 *  \deprecated Obsolete coding style.
 */
void Getopt::AddParameter(const GetoptParameterList& parameters)
{
  addParameter(parameters);
}

/*! Initializes Getopt with a commandline.
 *  \deprecated Obsolete coding style.
 */
bool Getopt::Initialize(int& argc, char *argv[], bool remove_read_arguments)
{
  return initialize(argc, argv, remove_read_arguments);
}

/*! Initializes Getopt with a commandline.
 *  \deprecated Obsolete coding style.
 */
bool Getopt::Initialize(int& argc, char *argv[],
                        CommandLineCleaning cleanup,
                        ParameterRegistrationCheck registration_check)
{
  return initialize(argc, argv, cleanup, registration_check);
}

/*! Returns \c true if Getopt has already been initialized.
 *  \deprecated Obsolete coding style.
 */
bool Getopt::IsInitialized() const
{
  return isInitialized();
}

/*! Get the extra command parameter at \a index.
 *  \deprecated Obsolete coding style.
 */
icl_core::String Getopt::ExtraCmdParam(size_t index) const
{
  return extraCmdParam(index);
}

/*! Get the number of extra command parameters.
 *  \deprecated Obsolete coding style.
 */
size_t Getopt::ExtraCmdParamCount() const
{
  return extraCmdParamCount();
}

/*! Get the value of the commandline option \a name.
 *  \deprecated Obsolete coding style.
 */
icl_core::String Getopt::ParamOpt(const icl_core::String& name) const
{
  return paramOpt(name);
}

/*! Checks if the option \a name is present.
 *  \deprecated Obsolete coding style.
 */
bool Getopt::ParamOptPresent(const icl_core::String& name) const
{
  return paramOptPresent(name);
}

/*! Get the list of defined suffixes for the specified \a prefix.
 *  \deprecated Obsolete coding style.
 */
Getopt::KeyValueList Getopt::ParamPrefixOpt(const icl_core::String& prefix) const
{
  return paramPrefixOpt(prefix);
}

/*! Check in a prefix option is present.
 *  \deprecated Obsolete coding style.
 */
bool Getopt::ParamPrefixOptPresent(const icl_core::String& prefix) const
{
  return paramPrefixOptPresent(prefix);
}

/*! Get the non-option parameter at the specified \a index.
 *  \deprecated Obsolete coding style.
 */
icl_core::String Getopt::ParamNonOpt(size_t index) const
{
  return paramNonOpt(index);
}

/*! Get the number of non-option parameters.
 *  \deprecated Obsolete coding style.
 */
size_t Getopt::ParamNonOptCount() const
{
  return paramNonOptCount();
}

/*! Get the program name.
 *  \deprecated Obsolete coding style.
 */
icl_core::String Getopt::ProgramName() const
{
  return programName();
}

/*! Prints the help text.
 *  \deprecated Obsolete coding style.
 */
void Getopt::PrintHelp() const
{
  printHelp();
}

#endif
/////////////////////////////////////////////////

Getopt::Getopt()
  : m_extra_cmd_param_activated(false),
    m_initialized(false)
{
  addParameter(GetoptParameter("help", "h", "Print commandline help."));
}

}
}
