// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2007-10-03
 *
 */
//----------------------------------------------------------------------
#include "icl_core_logging/FileLogOutput.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <iostream>

#include "icl_core/os_fs.h"
#include "icl_core_config/Config.h"
#include "icl_core_logging/Logging.h"

#if defined(_SYSTEM_POSIX_) && !defined(__ANDROID__)
# include <wordexp.h>
#endif

namespace icl_core {
namespace logging {

REGISTER_LOG_OUTPUT_STREAM(File, &FileLogOutput::create)

LogOutputStream *FileLogOutput::create(const icl_core::String& name, const icl_core::String& config_prefix,
                                       icl_core::logging::LogLevel log_level)
{
  return new FileLogOutput(name, config_prefix, log_level);
}

FileLogOutput::FileLogOutput(const icl_core::String& name, const icl_core::String& config_prefix,
                             icl_core::logging::LogLevel log_level)
  : LogOutputStream(name, config_prefix, log_level),
    m_rotate(false),
    m_last_rotation(0),
    m_delete_old_files(false),
    m_delete_older_than_days(0)
#if defined(_IC_BUILDER_ZLIB_)
  , m_zipped_log_file(NULL)
#endif
{
  icl_core::config::get<bool>(config_prefix + "/Rotate", m_rotate);
  if (m_rotate)
  {
    m_last_rotation = icl_core::TimeStamp::now().days();
  }

  if (icl_core::config::get<uint32_t>(config_prefix + "/DeleteOlderThan", m_delete_older_than_days))
  {
    m_delete_old_files = true;
  }

#if defined(_IC_BUILDER_ZLIB_)
  m_online_zip = icl_core::config::getDefault<bool>(config_prefix + "/Zip", false);
#endif

  m_flush = icl_core::config::getDefault<bool>(config_prefix + "/Flush", true);

  if (icl_core::config::get<icl_core::String>(config_prefix + "/FileName", m_filename))
  {
    expandFilename();

    // Determine the last write time of the log file, if it already
    // exists.
    boost::filesystem::path log_file_path(m_filename);
    if (boost::filesystem::exists(log_file_path))
    {
      if (boost::filesystem::is_directory(log_file_path))
      {
        std::cerr << "The filename specified for log output stream "
                  << config_prefix << " is a directory." << std::endl;
      }
      else
      {
        m_last_rotation = icl_core::TimeStamp(boost::filesystem::last_write_time(log_file_path)).days();
        rotateLogFile();
      }
    }

    openLogFile();
  }
  else
  {
    std::cerr << "No filename specified for file log output stream " << config_prefix << std::endl;
  }
}

FileLogOutput::FileLogOutput(const icl_core::String& name, const icl_core::String& filename,
                             icl_core::logging::LogLevel log_level, bool flush)
  : LogOutputStream(name, log_level),
    m_filename(filename),
    m_rotate(false),
    m_last_rotation(0),
    m_delete_old_files(false),
    m_delete_older_than_days(0),
    m_flush(flush)
#if defined(_IC_BUILDER_ZLIB_)
  , m_online_zip(false),
    m_zipped_log_file(NULL)
#endif
{
  expandFilename();
  openLogFile();
}

FileLogOutput::~FileLogOutput()
{
  closeLogFile();
}

void FileLogOutput::pushImpl(const icl_core::String& log_line)
{
  rotateLogFile();

  if (!isOpen())
  {
    openLogFile();
  }

  if (isOpen())
  {
#ifdef _IC_BUILDER_ZLIB_
    if (m_online_zip)
    {
      gzwrite(m_zipped_log_file, log_line.c_str(), static_cast<unsigned int>(log_line.length()));
    }
    else
#endif
    {
      m_log_file << log_line;
    }

    if (m_flush)
    {
      flush();
    }
  }
}

bool FileLogOutput::isOpen()
{
#ifdef _IC_BUILDER_ZLIB_
  if (m_online_zip)
  {
    return m_zipped_log_file != NULL;
  }
  else
#endif
  {
    return m_log_file.is_open();
  }
}

void FileLogOutput::flush()
{
#ifdef _IC_BUILDER_ZLIB_
  if (m_online_zip)
  {
    gzflush(m_zipped_log_file, Z_SYNC_FLUSH);
  }
  else
#endif
  {
    m_log_file.flush();
  }
}

void FileLogOutput::closeLogFile()
{
#ifdef _IC_BUILDER_ZLIB_
  if (m_online_zip)
  {
    if (m_zipped_log_file != NULL)
    {
      gzclose(m_zipped_log_file);
      m_zipped_log_file = NULL;
    }
  }
  else
#endif
  {
    if (m_log_file.is_open())
    {
      m_log_file.close();
    }
  }
}

void FileLogOutput::openLogFile()
{
#if defined(_IC_BUILDER_ZLIB_)
  if (m_online_zip)
  {
    m_zipped_log_file = gzopen(m_filename.c_str(), "a+b");
    if (m_zipped_log_file == NULL)
    {
      std::cerr << "Could not open log file " << m_filename << std::endl;
    }
    else
    {
      const char *buffer = "\n\n-------------FILE (RE-)OPENED------------------\n";
      gzwrite(m_zipped_log_file, buffer, static_cast<unsigned int>(strlen(buffer)));
    }
  }
  else
#endif
    if (!m_log_file.is_open())
    {
      m_log_file.open(m_filename.c_str(), std::ios::out | std::ios::app);
      if (m_log_file.is_open())
      {
        m_log_file << "\n\n-------------FILE (RE-)OPENED------------------\n";
        m_log_file.flush();
      }
      else
      {
        std::cerr << "Could not open log file " << m_filename << std::endl;
      }
    }
}

void FileLogOutput::rotateLogFile()
{
  if (m_rotate)
  {
    int64_t current_day = icl_core::TimeStamp::now().days();
    if (m_last_rotation != current_day)
    {
      // First, close the log file if it's open.
      closeLogFile();

      // Move the file. ZIP it, if libz is available.
      char time_str[12];
      icl_core::TimeStamp(24*3600*m_last_rotation).strfTime(time_str, 12, ".%Y-%m-%d");
#ifdef _IC_BUILDER_ZLIB_
      if (!m_online_zip)
      {
        icl_core::os::zipFile(m_filename.c_str(), time_str);
        icl_core::os::unlink(m_filename.c_str());
      }
      else
#endif
      {
        icl_core::os::rename(m_filename.c_str(), (m_filename + time_str).c_str());
      }

      // Delete old log files.
      if (m_delete_old_files)
      {
#if !defined(BOOST_FILESYSTEM_VERSION) || BOOST_FILESYSTEM_VERSION == 2
        boost::filesystem::path log_file_path = boost::filesystem::path(m_filename).branch_path();
        std::string log_file_name = boost::filesystem::path(m_filename).leaf();
#else
        boost::filesystem::path log_file_path = boost::filesystem::path(m_filename).parent_path();
        std::string log_file_name = boost::filesystem::path(m_filename).filename().string();
#endif
        if (boost::filesystem::exists(log_file_path) && boost::filesystem::is_directory(log_file_path))
        {
          icl_core::TimeStamp delete_older_than(24*3600*(current_day - m_delete_older_than_days));
          for (boost::filesystem::directory_iterator it(log_file_path), end; it != end; ++it)
          {
            // If the found file starts with the name of the log file the check its last write time.
            if (!is_directory(*it)
                && icl_core::TimeStamp(boost::filesystem::last_write_time(*it)) < delete_older_than
#if !defined(BOOST_FILESYSTEM_VERSION) || BOOST_FILESYSTEM_VERSION == 2
                && it->leaf().find(log_file_name) == 0
#else
                && it->path().filename().string().find(log_file_name) == 0
#endif
                )
            {
              boost::filesystem::remove(*it);
            }
          }
        }
      }

      // Store the rotation time.
      m_last_rotation = current_day;

      // Re-open the log file.
      openLogFile();
    }
  }
}

void FileLogOutput::expandFilename()
{
  // Expand environment variables.
#if defined(_SYSTEM_POSIX_) && !defined(__ANDROID__)
  wordexp_t p;
  if (wordexp(m_filename.c_str(), &p, 0) == 0)
  {
    if (p.we_wordc > 0)
    {
      m_filename = p.we_wordv[0];
    }
  }
  //wordfree(&p);
#elif defined(_SYSTEM_WIN32_)
  // TODO: Implement this with ExpandEnvironmenStrings()
#endif
}

}
}
