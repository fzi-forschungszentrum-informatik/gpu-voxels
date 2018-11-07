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
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2001-01-11
 *
 */
//----------------------------------------------------------------------
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _SYSTEM_WIN32_
#include <direct.h>
#endif

#include "icl_core/os_string.h"
#include "icl_core_config/AttributeTree.h"

using namespace std;

#define LOCAL_PRINTF printf

namespace icl_core {
namespace config {

#ifdef _SYSTEM_WIN32_
const char DSEP = '\\';
const char DSEP_OTHER = '/';
#else
const char DSEP = '/';
const char DSEP_OTHER = '\\';
#endif

const char *comment_end_str = "}_COMMENT_";
const char *comment_str = comment_end_str + 1;
const char *include_str = "_INCLUDE_";
const char *AttributeTree::m_file_path_str = "_ATTRIBUTE_TREE_FILE_PATH_";
const char *AttributeTree::m_file_name_str = "_ATTRIBUTE_TREE_FILE_NAME_";
int file_path_str_len = 0;
int file_name_str_len = 0;
const int INCLUDE_OFFSET = 10;

static char buffer[2000];

void readNextLineInBuffer(istream &in)
{
  in.getline(buffer, 1998);
  // Window/Unix
  int line_length=strlen(buffer);
  if (line_length > 0 && buffer[line_length-1]=='\r')
  {
    buffer[line_length-1]=0;
  }
}

// ===============================================================
// FilePath
// ===============================================================

icl_core::String FilePath::absolutePath(const icl_core::String& filename) const
{
  if (isRelativePath(filename))
  {
    return normalizePath(currentDir() + DSEP + filename);
  }
  else
  {
    return normalizePath(filename);
  }
}

bool FilePath::isRelativePath(const icl_core::String& filename)
{
  if (filename.empty())
  {
    return true;
  }
#ifdef _SYSTEM_WIN32_
  //directory??
  return (filename.length() < 2 || filename[1] != ':');
#else
  return filename[0] != DSEP;
#endif
}

icl_core::String FilePath::normalizePath(const icl_core::String& _filename)
{
  if (_filename.empty())
  {
    return _filename;
  }
  string filename(_filename);

  //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath(%s)\n", filename.c_str());

  // Replace directory separators.
  string::size_type slash_before_dot;
  string::size_type dot_start = 0;

  dot_start = filename.find(DSEP_OTHER, 0);
  while (dot_start != string::npos)
  {
    filename[dot_start] = DSEP;

    dot_start = filename.find(DSEP_OTHER, dot_start);
  }
  dot_start = 0;

  //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath>> status '%s'\n", filename.c_str());

  // Search for single dots at the beginning.
  while (!filename.find(string(".") + DSEP))
  {
    //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath>> SingleDots at begin '%s'\n",
    //         filename.c_str());
    string temp_filename(filename, 2, string::npos);
    temp_filename.swap(filename);
  }
#ifdef _SYSTEM_WIN32_
  // Search for single dots after the drive letter at the beginning.
  if (filename.find(string(":.") + DSEP) == 1)
  {
    string temp_filename;
    temp_filename.append(filename, 0, 2);
    dot_start += 3;
    temp_filename.append(filename, dot_start, filename.length() - dot_start);
    temp_filename.swap(filename);
    dot_start = 0;
  }
#endif
  //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath>> status '%s'\n", filename.c_str());

  // Search for single dots inside.
  dot_start = filename.find(string(1, DSEP) + "." + DSEP, 0);
  while (dot_start != string::npos)
  {
    //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath>> SingleDots inside '%s'\n",
    //         filename.c_str());
    string temp_filename(filename, 0, dot_start);
    temp_filename.append(filename, dot_start + 2, filename.length() - dot_start - 2);
    temp_filename.swap(filename);

    dot_start = filename.find(string(1, DSEP) + "." + DSEP, dot_start);
  }
  //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath>> status '%s'\n", filename.c_str());

  // Search for double dots.
  dot_start = filename.find(string(1, DSEP) + ".." + DSEP, 0);
  while (dot_start != string::npos)
  {
    //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath>> DoubleDots '%s' found at %i\n",
    //         filename.c_str(), dot_start);

    // Check if we can shorten the path.
    slash_before_dot = filename.rfind(DSEP, dot_start - 1);
    if (slash_before_dot != string::npos)
    {
      // OK to shorten?
      if (filename[slash_before_dot+1] != DSEP && filename[slash_before_dot+1] != '.' &&
          (slash_before_dot >= 1) ? filename[slash_before_dot-1] != ':' : 1)
      {
        dot_start += 3;
        string temp_filename(filename, 0, slash_before_dot);
        temp_filename.append(filename, dot_start, filename.length() - dot_start);
        temp_filename.swap(filename);
      }
      else
      {
        break;
      }

      dot_start = slash_before_dot;
    }
    else if (dot_start > 0)
    {
      string temp_filename;

#ifdef _SYSTEM_WIN32_
      // Add drive letter?
      if (filename[1] = ':')
      {
        temp_filename.append(filename, 0, 2);
      }
#endif
      // Really??
      dot_start += 2;
      temp_filename.append(filename, dot_start, filename.length() - dot_start);
      temp_filename.swap(filename);
      dot_start = 0;
    }
    else
    {
      break;
    }

    dot_start = filename.find(string(1, DSEP) + ".." + DSEP, dot_start);
  }
  //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath>> status '%s'\n", filename.c_str());
  // Search again for single dots at the beginning.
  while (!filename.find(string(".") + DSEP))
  {
    //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath>> SingleDots at begin '%s'\n",
    //         filename.c_str());
    string temp_filename(filename, 2, string::npos);
    temp_filename.swap(filename);
  }
#ifdef _SYSTEM_WIN32_
  dot_start = 0;
  // Search again for single dots after the drive letter at the
  // beginning.
  if ((filename.find(string(":.") + DSEP) == 1))
  {
    string temp_filename;
    temp_filename.append(filename, 0, 2);
    dot_start += 3;
    temp_filename.append(filename, dot_start, filename.length() - dot_start);
    temp_filename.swap(filename);
    dot_start = 0;
  }
#endif
  //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::normalizePath>> Return '%s'\n", filename.c_str());
  return filename;
}

icl_core::String FilePath::exchangeSeparators(const icl_core::String& _filename)
{
  if (_filename.empty())
  {
    return _filename;
  }
  string filename(_filename);

  //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::exchangeSeparators(%s)\n", filename.c_str());

  for (unsigned i = 0;i < filename.length();++i)
  {
    if (filename[i] == DSEP_OTHER)
    {
      filename[i] = DSEP;
    }
  }

  return filename;
}

icl_core::String FilePath::getEnvironment(const icl_core::String& var_name)
{
  const char* get_env = getenv(var_name.c_str());
  if (get_env == NULL)
  {
    return var_name;
  }
  else
  {
    return string(get_env);
  }
}

icl_core::String FilePath::replaceEnvironment(const icl_core::String& _filename)
{
  if (_filename.empty())
  {
    return _filename;
  }
  string filename(_filename);

  //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::replaceEnvironment(%s)\n", filename.c_str());

  string::size_type dot_start = filename.find("${");
  while (dot_start != string::npos)
  {
    dot_start += 2;
    string::size_type dot_end = filename.find("}", dot_start);
    if (dot_end == string::npos)
    {
      printf("tFilePath::replaceEnvironment(%s)>> Failure on matching closing bracket "
             "'}' in substring '%s'\n", (const char*) _filename.c_str(),
             (const char*) string(filename, dot_start, string::npos).c_str());
      return _filename;
    }
    string var_name(filename, dot_start, dot_end - dot_start);
    string temp_filename(filename, 0, dot_start - 2);
    temp_filename += getEnvironment(var_name);
    temp_filename += string(filename, dot_end + 1, string::npos);
    filename.swap(temp_filename);

    dot_start = filename.find("${");
  }

  //DEBUGMSG(DD_SYSTEM, DL_VERBOSE, "FilePath::replaceEnvironment(%s) done (%s)\n",
  //         _filename.c_str(), filename.c_str());
  return filename;
}

#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Returns the absolute path of the file without the file name.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::Path() const
  {
    return path();
  }

  /*! Returns the name of the file without path.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::Name() const
  {
    return name();
  }

  /*! Returns the path and name of the file.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::AbsoluteName() const
  {
    return absoluteName();
  }

  /*! Returns the absolute path of the current directory.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::CurrentDir() const
  {
    return currentDir();
  }

  /*! Returns the extension of the file.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::Extension() const
  {
    return extension();
  }

  /*! Returns the absolute path of the file \a filename.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::AbsolutePath(const icl_core::String& filename) const
  {
    return absolutePath(filename);
  }

  /*! Returns \c true if the \a filename is a relative path.
   *  \deprecated Obsolete coding style.
   */
  bool FilePath::IsRelativePath(const icl_core::String& filename)
  {
    return isRelativePath(filename);
  }

  /*! Returns \c true if the objects filename is a relative path.
   *  \deprecated Obsolete coding style.
   */
  bool FilePath::IsRelativePath() const
  {
    return isRelativePath();
  }

  /*! Returns the normalized path of the given \a filename.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::NormalizePath(const icl_core::String& filename)
  {
    return normalizePath(filename);
  }

  /*! Searches for directory separators, which are not supported by
   *  the underlying operation system and exchanges these with
   *  separators, that are supported.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::ExchangeSeparators(const icl_core::String& filename)
  {
    return exchangeSeparators(filename);
  }

  /*! Returns the value of the given environment variable \a var_name.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::GetEnvironment(const icl_core::String& var_name)
  {
    return getEnvironment(var_name);
  }

  /*! Replaces environment variables in the given filename.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String FilePath::ReplaceEnvironment(const icl_core::String& filename)
  {
    return replaceEnvironment(filename);
  }

#endif
  /////////////////////////////////////////////////


void FilePath::init(const char* filename)
{
  // Get the current directory.
#ifdef _SYSTEM_WIN32_
  {
    char*  buffer = new char[1000];
    if (_getcwd(buffer, 1000) != NULL)
    {
      buffer[999] = 0; // Safety precaution
      m_pwd = buffer;
    }
    else
    {
      m_pwd = "";
    }
    delete[] buffer;
  }
#else
  char* tmp;
  tmp = getenv("PWD");
  if (tmp != NULL)
  {
    m_pwd = tmp;
  }
  else
  {
    m_pwd = "";
  }
#endif
  m_pwd = normalizePath(m_pwd);

  m_file = normalizePath(absolutePath(exchangeSeparators(string(filename))));

  // Search the last directory.
  string::size_type last_directory_separator = m_file.rfind(DSEP);
  if (last_directory_separator < m_file.length())
  {
    m_file_path_name_split = last_directory_separator + 1;
  }
  // Didn't find anything?
  else
  {
    m_file_path_name_split = 0;
  }

  m_file_name_extension_split = m_file.rfind('.');
}


// ===============================================================
// SubTreeList
// ===============================================================
SubTreeList::SubTreeList(AttributeTree *sub_tree, SubTreeList *next)
  : m_next(next),
    m_sub_tree(sub_tree)
{
}

SubTreeList::~SubTreeList()
{
  if (m_sub_tree)
  {
    // parent auf 0 damit unlink nicht wieder diesen Destruktor aufruft (cycle!)
    m_sub_tree->m_parent = 0;
    delete m_sub_tree;
  }
  delete m_next;
}

void SubTreeList::copy(AttributeTree *parent)
{
  assert(parent != NULL
         && "SubTreeList::copy() called with NULL parent! Allocated attribute tree would be lost!");

  SubTreeList *loop = this;
  while (loop)
  {
    new AttributeTree(*loop->m_sub_tree, parent);
    loop = loop->m_next;
  }
}

void SubTreeList::unlinkParent()
{
  SubTreeList *loop = this;
  while (loop)
  {
    if (loop->m_sub_tree)
    {
      loop->m_sub_tree->m_parent = 0;
    }
    loop = loop->m_next;
  }
}

void SubTreeList::unlink(AttributeTree *obsolete_tree)
{
  SubTreeList *loop = this;
  SubTreeList *prev = 0;
  while (loop)
  {
    if (loop->m_sub_tree == obsolete_tree)
    {
      if (prev)
      {
        prev->m_next = loop->m_next;
      }
      loop->m_next = 0;
      loop->m_sub_tree = 0;
      delete loop;
      return;
    }
    prev = loop;
    loop = loop->m_next;
  }
  // nicht gefunden? Dann machen wir gar nichts!
}

AttributeTree* SubTreeList::subTree(const char *description)
{
  SubTreeList *loop = this;
  while (loop)
  {
    if (loop->m_sub_tree && loop->m_sub_tree->getDescription()
        && !strcmp(loop->m_sub_tree->getDescription(), description))
    {
      return loop->m_sub_tree;
    }
    loop = loop->m_next;
  }
  return 0;
}

int SubTreeList::contains()
{
  int ret = 0;
  SubTreeList *loop = this;
  while (loop)
  {
    ret += loop->m_sub_tree->contains();
    loop = loop->m_next;
  }
  return ret;
}

void SubTreeList::printSubTree(ostream& out, int change_style_depth, char *upper_description)
{
  SubTreeList *loop = this;
  while (loop)
  {
    loop->m_sub_tree->printSubTree(out, change_style_depth, upper_description);
    loop = loop->m_next;
  }
}

AttributeTree* SubTreeList::search(const char *description, const char *attribute)
{
  SubTreeList *loop = this;
  while (loop)
  {
    AttributeTree* search = loop->m_sub_tree->search(description, attribute);
    if (search)
    {
      return search;
    }
    loop = loop->m_next;
  }
  return NULL;
}

bool SubTreeList::changed()
{
  SubTreeList *loop = this;
  while (loop)
  {
    if (loop->m_sub_tree->changed())
    {
      return true;
    }
    loop = loop->m_next;
  }
  return false;
}

AttributeTree* SubTreeList::next(AttributeTree *prev)
{
  SubTreeList *loop = this;
  while (loop)
  {
    if (loop->m_sub_tree == prev)
    {
      if (loop->m_next)
      {
        return loop->m_next->m_sub_tree;
      }
    }
    loop = loop->m_next;
  }
  return 0;
}

void SubTreeList::unmarkChanges()
{
  SubTreeList *loop = this;
  while (loop)
  {
    loop->m_sub_tree->unmarkChanges();
    loop = loop->m_next;
  }
}

SubTreeList* SubTreeList::revertOrder(SubTreeList *new_next)
{
  SubTreeList* ret = this;
  if (m_sub_tree)
  {
    m_sub_tree->revertOrder();
  }
  if (m_next)
  {
    ret = m_next->revertOrder(this);
  }
  m_next = new_next;
  return ret;
}

void SubTreeList::newSubTreeList(AttributeTree *new_tree, AttributeTree* after)
{
  SubTreeList *loop;
  // loop through all next ones
  for (loop = this; loop->m_next && loop->m_sub_tree != after; loop = loop->m_next)
  { }
  // and insert the new
  loop->m_next = new SubTreeList(new_tree, loop->m_next);
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Reverts the order of the sub tree list.
   *  \deprecated Obsolete coding style.
   */
  SubTreeList *SubTreeList::RevertOrder(SubTreeList *new_next)
  {
    return revertOrder(new_next);
  }

  /*! Creates new SubTreeList with \a new_tree as subtree after the
   *  next tree with subtree \a after.
   *  \deprecated Obsolete coding style.
   */
  void SubTreeList::NewSubTreeList(AttributeTree *new_tree, AttributeTree *after)
  {
    return newSubTreeList(new_tree, after);
  }

#endif
/////////////////////////////////////////////////

// ===============================================================
// AttributeTree
// ===============================================================

AttributeTree::AttributeTree(const char *description, AttributeTree *parent)
  : m_parent(parent),
    m_subtree_list(NULL)
{
  file_path_str_len = strlen(m_file_path_str);
  file_name_str_len = strlen(m_file_name_str);
  if (description)
  {
    m_this_description = icl_core::os::strdup(description);
  }
  else
  {
    m_this_description = 0;
  }
  m_this_attribute = 0;
  m_changed = false;

  // Beim Parent in die Liste einfügen
  if (m_parent)
  {
    m_parent->m_subtree_list = new SubTreeList(this, m_parent->m_subtree_list);
  }
}

AttributeTree::AttributeTree(const AttributeTree& tree)
  : m_parent(NULL),
    m_subtree_list(NULL)
{
  file_path_str_len = strlen(m_file_path_str);
  file_name_str_len = strlen(m_file_name_str);

  if (tree.m_this_description)
  {
    m_this_description = icl_core::os::strdup(tree.m_this_description);
  }
  else
  {
    m_this_description = 0;
  }
  if (tree.m_this_attribute)
  {
    m_this_attribute = icl_core::os::strdup(tree.m_this_attribute);
  }
  else
  {
    m_this_attribute = 0;
  }
  if (tree.m_subtree_list)
  {
    tree.m_subtree_list->copy(this);
  }

  m_changed = false;
}

AttributeTree::AttributeTree(const AttributeTree &tree, AttributeTree *parent)
  : m_parent(parent),
    m_subtree_list(NULL)
{
  file_path_str_len = strlen(m_file_path_str);
  file_name_str_len = strlen(m_file_name_str);

  if (tree.m_this_description)
  {
    m_this_description = icl_core::os::strdup(tree.m_this_description);
  }
  else
  {
    m_this_description = 0;
  }
  if (tree.m_this_attribute)
  {
    m_this_attribute = icl_core::os::strdup(tree.m_this_attribute);
  }
  else
  {
    m_this_attribute = 0;
  }
  if (tree.m_subtree_list)
  {
    tree.m_subtree_list->copy(this);
  }
  // Beim Parent in die Liste einfügen
  if (m_parent)
  {
    m_parent->m_subtree_list = new SubTreeList(this, m_parent->m_subtree_list);
  }

  m_changed = false;
}

AttributeTree::~AttributeTree()
{
  //  DEBUGMSG(-3, "AttributeTree::~ >> Deleting ...\n");
  if (m_this_description)
  {
    // DEBUGMSG(-3, "\t descr(%p)='%s'\n", this, m_this_description);
    free(m_this_description);
    m_this_description = 0;
  }
  if (m_this_attribute)
  {
    //      DEBUGMSG(-3, "\t attr=%s\n", m_this_attribute);
    free(m_this_attribute);
    m_this_attribute = 0;
  }
  // subtree wird komplett ausgelöscht
  if (m_subtree_list)
  {
    // DEBUGMSG(-3, "Entering sub (%p)...\n", this);
    delete m_subtree_list;
    m_subtree_list = 0;
    // DEBUGMSG(-3, "Leaving sub (%p) ...\n", this);
  }

  unlink();
}

void AttributeTree::unlinkSub()
{
  if (m_subtree_list)
  {
    // die parent-Zeiger der Unterbäume auf 0 setzen
    m_subtree_list->unlinkParent();
    // den Unterbaumzeiger auf 0
    m_subtree_list = 0;
  }
}

void AttributeTree::unlink()
{
  if (m_parent)
  {
    SubTreeList *first_entry = m_parent->m_subtree_list;
    if (first_entry->m_sub_tree == this)
    {
      m_parent->m_subtree_list = first_entry->m_next;
    }

    first_entry->unlink(this);
    m_parent->m_changed = true;
  }
  m_parent = 0;
}


void AttributeTree::setDescription(const char *description)
{
  free(m_this_description);
  if (description)
  {
    m_this_description = icl_core::os::strdup(description);
  }
  else
  {
    m_this_description = 0;
  }
}

void AttributeTree::setAttribute(const char *attribute)
{
  //printf("Change Attribute:%s %s\n",m_this_attribute,attribute);
  if (!m_this_attribute || !attribute || strcmp(attribute, m_this_attribute))
  {
    free(m_this_attribute);
    if (attribute)
    {
      m_this_attribute = icl_core::os::strdup(attribute);
    }
    else
    {
      m_this_attribute = 0;
    }
    m_changed = true;
  }
}


AttributeTree* AttributeTree::subTree(const char *description)
{
  AttributeTree *m_sub_tree = getSubTree(description);
  if (m_sub_tree != NULL)
  {
    // Gibt's schon einen, geben wir diesen zurück
    return m_sub_tree;
  }
  else
  {
    // Ansonsten erzeugen wir einen:
    return new AttributeTree(description, this);
  }
}

AttributeTree* AttributeTree::getSubTree(const char *description)
{
  AttributeTree *m_sub_tree;
  // Erstmal suchen, obs schon einen gibt:
  if (m_subtree_list)
  {
    m_sub_tree = m_subtree_list->subTree(description);
    if (m_sub_tree)
    {
      return m_sub_tree;
    }
  }
  // Ansonsten geben wir NULL zurück
  return NULL;
}


AttributeTree *AttributeTree::setAttribute(const char *param_description, const char *attribute)
{
  if (param_description)
  {
    char *description = icl_core::os::strdup(param_description);
    //printf("a:%s:%s\n",description,attribute);
    char *subdescription;
    split(description, subdescription);
    //printf("b:%s--%p\n",description,subdescription);
    AttributeTree *ret = setAttribute(description, subdescription, attribute);
    free(description);
    //printf("c:%p \n",ret);
    return ret;
  }
  setAttribute(attribute);
  return this;
}

AttributeTree* AttributeTree::setAttribute(const char *description, const char *subdescription,
                                           const char *attribute)
{
  // printf("%p---%s,%s,%s\n",this,description,subdescription,attribute);
  if (!description || !*description)
  {
    // Keine Description -> Wir sind am Endknoten -> Eintrag machen
    //  printf("set attribute: %s :%s\n",m_this_description,attribute);
    setAttribute(attribute);

    return this;
  }

  //printf("1--%p\n",m_this_description);
  // Ansonsten müssen wir weiter nach unten suchen:
  AttributeTree *subtree = 0;
  if (m_subtree_list)
  {
    subtree = m_subtree_list->subTree(description);
  }

  //printf("2--\n");
  if (subtree)
  {
    return subtree->setAttribute(subdescription, attribute);
  }

  // Kein passender Eintrag gefunden -> neuen Sub-Baum erzeugen:
  AttributeTree  *new_subtree = new AttributeTree(description, this);
  //printf("3--:%p\n",new_subtree);

  return new_subtree->setAttribute(subdescription, attribute);
}


char *AttributeTree::getSpecialAttribute(const char *description, AttributeTree **subtree)
{
  // search recursive to the root for that attribute
  AttributeTree *at_path_parent = this;
  AttributeTree *at_path = at_path_parent->m_subtree_list->subTree(description);
  while (at_path_parent && at_path == NULL)
  {
    at_path = at_path_parent->m_subtree_list->subTree(description);
    at_path_parent = at_path_parent->parentTree();
  }

  // found
  if (at_path && at_path->m_this_attribute)
  {
    //DEBUGMSG(DD_AT, DL_DEBUG, "AttributeTree::getSpecialAttribute>> found special attribute %s with %s\n",
    //         m_file_path_str, at_path->m_this_attribute);
    if (subtree)
    {
      (*subtree) = at_path;
    }
    return at_path->m_this_attribute;
  }
  return NULL;
}

char *AttributeTree::getAttribute(const char *param_description, const char *default_attribute,
                                  AttributeTree **subtree)
{
  char*ret = 0;
  if (param_description)
  {
    char *description = icl_core::os::strdup(param_description);
    if (description)
    {
      AttributeTree* at = this;
      // check for 'm_file_path_str' and 'm_file_name_str'
      int len = strlen(description);
      if (len >= file_path_str_len
          && !strncmp(description + (len - file_path_str_len), m_file_path_str, file_path_str_len))
      {
        ret = getSpecialAttribute(m_file_path_str, subtree);
      }
      else if (len >= file_name_str_len
               && !strncmp(description + (len - file_name_str_len), m_file_name_str, file_name_str_len))
      {
        ret = getSpecialAttribute(m_file_name_str, subtree);
      }

      // not found yet ... trying the standard search
      if (!ret)
      {
        char *description_part = description;
        // go into the attribute tree structure
        for (; at && description_part;)
        {
          // save the begin of the description
          char *next_description = description_part;
          // searching for further dots
          description_part = strchr(description_part, '.');
          if (description_part)
          {
            *description_part = 0;
            description_part++;
          }
          at = at->m_subtree_list->subTree(next_description);
        }
        // now we are at the inner attribute tree
        // is there an attribute
        if (at && at->m_this_attribute)
        {
          //DEBUGMSG(DD_AT, DL_DEBUG, "AttributeTree::getAttribute>> found %s\n", at->m_this_attribute);
          if (subtree)
          {
            (*subtree) = at;
          }
          ret = at->m_this_attribute;
        }
      }
      free(description);
    }
  }
  // didn't find anything
  if (!ret)
  {
    if (subtree)
    {
      (*subtree) = 0;
    }
    //DEBUGMSG(DD_AT, DL_DEBUG, "AttributeTree::getAttribute>> nothing found. Return default %s\n",
    //         default_attribute ? default_attribute : "(null)");
    ret = const_cast<char*>(default_attribute);
  }

  return ret;
}

char* AttributeTree::getOrSetDefault(const char *description, const char *default_attribute)
{
  char *attribute = getAttribute(description, 0);
  if (!attribute)
  {
    setAttribute(description, default_attribute);
    attribute = const_cast<char*>(default_attribute);
  }
  return attribute;
}

char *AttributeTree::newSubNodeDescription(const char *base_description)
{
  int base_len = strlen(base_description);
  char *description = (char*)malloc(base_len + 6);
  assert(description != NULL); // Just abort if we are out of memory.
  strcpy(description, base_description);
  int i = 1;
  int j = 0;

  // find the maxima length of number in base_description
  if (base_len>0)
  {
    while (base_len>=j-1 &&
           sscanf(description+base_len-j-1, "%i", &i)==1)
    {
      j++;
    }
    if (j!=0)
    {
      i++;
    }
  }

  sprintf(description + base_len - j, "%i", i);

  while (m_subtree_list->subTree(description) && i < 100000)
  {
    i++;
    sprintf(description + base_len - j, "%i", i);
  }
  return description;
}


AttributeTree *AttributeTree::addNewSubTree()
{
  char *name = newSubNodeDescription();
  AttributeTree *ret = setAttribute(name, 0);
  free(name);
  return ret;
}


AttributeTree *AttributeTree::addSubTree(AttributeTree *tree, AttributeTree *after)
{
  if (tree)
  {
    if (m_subtree_list->subTree(tree->m_this_description))
    {
      char *new_description = newSubNodeDescription(tree->m_this_description);
      free(tree->m_this_description);
      tree->m_this_description = new_description;
    }

    if (after == NULL)
    {
      m_subtree_list = new SubTreeList(tree, m_subtree_list);
    }
    else
    {
      m_subtree_list->newSubTreeList(tree, after);
    }

    tree->m_parent = this;
    return tree;
  }
  else
  {
    return NULL;
  }
}

void AttributeTree::printSubTree(ostream& out, int change_style_depth, const char *upper_description)
{
  // virtual attributes are not stored !
  if (m_this_description && (!strcmp(m_this_description, m_file_path_str)
                             || !strcmp(m_this_description, m_file_name_str)))
  {
    return;
  }

  char *the_upper_description = strdup(upper_description ? upper_description : "");
  char *t_description = strdup(m_this_description ? m_this_description : "");
  assert(the_upper_description != NULL);
  assert(t_description != NULL);

  // is this the comment attribute tree ?
  if (isMultilineComment())
  {
    out << the_upper_description << comment_str << '{' << endl;
    out << m_this_attribute << endl;
    out << the_upper_description << '}' << comment_str << endl;

    free(the_upper_description);
    free(t_description);

    return;
  }

  int contents = contains();
  if (contents >= change_style_depth || hasMultilineComment())
  {
    out << the_upper_description << t_description << '{' << endl;
    if (m_this_attribute && strcmp(m_this_attribute, ""))
    {
      out << the_upper_description << ':' << m_this_attribute << endl;
    }

    if (m_subtree_list)
    {
      char *tab = (char*)malloc(strlen(the_upper_description) + 2);
      assert(tab != NULL); // Just abort if we are out of memory.
      strcat(strcpy(tab, the_upper_description), " ");
      m_subtree_list->printSubTree(out, change_style_depth, tab);
      free(tab);
    }

    out << the_upper_description << '}' << t_description << endl;
  }
  else
  {
    size_t tud_len = strlen(the_upper_description);
    size_t len = strlen(t_description) + tud_len + 1;
    char *description = static_cast<char*>(malloc(len + 1));
    assert(description != NULL); // Just abort if we are out of memory.
    memset(description, 0, len + 1);

    if ((tud_len > 0) && (the_upper_description[tud_len-1] == ' '))
    {
      strcat(strcpy(description, the_upper_description), t_description);
    }
    else
    {
      strcat(strcat(strcpy(description, the_upper_description), "."), t_description);
    }

    if (m_this_attribute)
    {
      out << description << ':' << m_this_attribute << endl;
    }

    if (m_subtree_list)
    {
      m_subtree_list->printSubTree(out, change_style_depth, description);
    }
    free(description);
  }

  free(the_upper_description);
  free(t_description);
}

int AttributeTree::save(const char *filename, int change_style_depth, bool unmark_changes)
{
  /*
  if (!m_this_description)
    return eEMPTY_TREE;
  */
  ofstream out(filename);
  if (!out)
  {
    return eFILE_SAVE_ERROR;
  }
  printSubTree(out, change_style_depth, "");

  if (unmark_changes)
  {
    unmarkChanges();
  }

  return eOK;
}

int AttributeTree::load(const char *filename, bool unmark_changes, bool process_include,
                        bool load_comments, bool preserve_order)
{
  if (filename == NULL || strcmp(filename, "") == 0)
  {
    printf("tAttributeTree >> Trying to load an empty configuration file.\n");
    return eFILE_LOAD_ERROR;
  }

  icl_core::config::FilePath at_file(filename);
  //LOCAL_PRINTF("AttributeTree >> Loading %s\n", at_file.AbsoluteName().c_str());
  if (this == root() && !getAttribute(m_file_path_str))
  {
    //LOCAL_PRINTF("AttributeTree >> Setting Virtual Attributes Path(%s) Name(%s)\n",
    //             at_file.Path().c_str(), at_file.Name().c_str());
    setAttribute(m_file_path_str, at_file.path().c_str());
    setAttribute(m_file_name_str, at_file.name().c_str());
  }

  int error;
  ifstream in(at_file.absoluteName().c_str());
  if (!in)
  {
    printf("tAttributeTree >> Could not open file '%s'\n", (const char*)at_file.absoluteName().c_str());
    return eFILE_LOAD_ERROR;
  }

  error = get(in, process_include, load_comments, &at_file);
  if (error >= 0)
  {
    printf("Error in line %i while reading AttributeTree %s\n", error,
           (const char*) at_file.absoluteName().c_str());
    return eFILE_LOAD_ERROR;
  }


  if (unmark_changes)
  {
    unmarkChanges();
  }
  if (preserve_order)
  {
    revertOrder();
  }
  //DEBUGMSG(DD_AT, DL_DEBUG, "AttributeTree >> Loading successful\n");

  return eOK;
}

int AttributeTree::get(istream &in, bool process_include, bool load_comments,
                       const class FilePath *file_path)
{
  // save stack memory on reccursive calls!
  // without static in the insmod call for the RTL-module we crash !
  buffer[1999] = 0;
  char *attribute, *line;
  AttributeTree *at = this;
  int lineno = 1;

  readNextLineInBuffer(in);

  do
  {
    //LOCAL_PRINTF("get next line %i\n",lineno);
    lineno++;
    line = buffer;
    while (isspace(*line))
    {
      line++;
    }
    //LOCAL_PRINTF("%s\n",line);
    if (line[0] != '#')
    {
      attribute = strchr(line, ':');
      if (attribute)
      {
        *attribute = 0;
        if (!line[0])
        {
          //LOCAL_PRINTF("AttributeTree::get >> found ':%s'\n", attribute+1);
          at->setAttribute(attribute + 1);
        }
        else
        {
          if (!strcmp(line, include_str))
          {
            if (process_include)
            {
              string include_filename(line + INCLUDE_OFFSET);
              include_filename = FilePath::exchangeSeparators(FilePath::replaceEnvironment(include_filename));
              if (FilePath::isRelativePath(include_filename))
              {
                string absolute_include_filename(file_path ? file_path->path() : getFilePath());
                absolute_include_filename += include_filename;
                include_filename = FilePath::normalizePath(absolute_include_filename);
              }
              if (at->load(include_filename.c_str(), false, process_include, load_comments) != eOK)
              {
                printf("error loading include file %s\n", (const char*)include_filename.c_str());
              }
            }
            else
            {
              // falls nicht includen: als "normalen" Eintrag speichern
              (new AttributeTree(include_str, at))->setAttribute(line + INCLUDE_OFFSET);
            }
          }
          else if (!strstr(line, comment_str) || load_comments)
          {
            //LOCAL_PRINTF("AttributeTree::get >> found '%s:%s'\n", line, attribute+1);
            at->setAttribute(line, attribute + 1);
          }
        }
      }
      else
      {
        attribute = strchr(line, '{');
        if (attribute)
        {
          *attribute = 0;
          //LOCAL_PRINTF("AttributeTree::get >> found '%s{'\n",  line);
          // multiline comments
          if (!strcmp(line, comment_str))
          {
            AttributeTree *at_c = 0;
            bool comment_end = false;
            if (load_comments)
            {
              at_c = new AttributeTree(comment_str, at);
            }
            do
            {
              lineno++;
              readNextLineInBuffer(in);
              line = buffer;
              char *line_end = buffer + strlen(buffer);
              line_end--;
              while (isspace(*line))
              {
                line++;
              }
              while (line_end >= buffer && isspace(*line_end))
              {
                line_end--;
              }
              *(line_end + 1) = 0;
              comment_end = (strstr(line, comment_end_str) != NULL);

              if (load_comments && !comment_end)
              {
                at_c->appendAttribute(line, "\n");
              }
            }
            while (!comment_end);
          }
          else
          {
            at = at->setAttribute(line, 0);
          }
        }
        else
        {
          attribute = strchr(line, '}');
          if (attribute)
          {
            if (at == this)
            {
              //LOCAL_PRINTF("AttributeTree::get >> found last '}'\n");
              return -1;
            }
            else
            {
              //LOCAL_PRINTF("AttributeTree::get >> found '}'\n");
              if (!at->parentTree())
              {
                return lineno;
              }
              at = at->parentTree();
            }
          }
          else
          {
            if (!in.eof() && line[0])
            {
              //LOCAL_PRINTF("AttributeTree::get >> found '%s' and could not interpret\n", line);
              return lineno;
            }
          }
        }
      }
    }
    readNextLineInBuffer(in);
  }
  while (!in.eof());
  return -1;
}

void AttributeTree::split(char *&description, char *&subdescription)
{
  subdescription = strchr(description, '.');
  if (subdescription)
  {
    *subdescription = 0;
    subdescription++;
  }
}

AttributeTree* AttributeTree::search(const char *description, const char *attribute)
{
  if (description)
  {
    if ((m_this_description && (!strcmp(description, m_this_description))) &&
        (attribute == 0 || (m_this_attribute && (!strcmp(attribute, m_this_attribute)))))
    {
      return this;
    }
    if (m_subtree_list)
    {
      return m_subtree_list->search(description, attribute);
    }
  }
  return NULL;
}

bool AttributeTree::isAttribute(const char *description, const char *attribute)
{
  char *content = getAttribute(description, 0);

  if (attribute)
  {
    if (content)
    {
      return !strcmp(content, attribute);
    }
    return false;
  }
  else
  {
    return content != NULL;
  }
}

bool AttributeTree::changed()
{
  if (m_changed)
  {
    return true;
  }
  if (m_subtree_list)
  {
    return m_subtree_list->changed();
  }
  return false;
}

void AttributeTree::unmarkChanges()
{
  m_changed = false;
  if (m_subtree_list)
  {
    m_subtree_list->unmarkChanges();
  }
}

int AttributeTree::contains()
{
  int ret = 0;
  if (m_this_attribute)
  {
    ret++;
  }
  if (m_subtree_list)
  {
    ret += m_subtree_list->contains();
  }
  return ret;
}

void AttributeTree::appendString(char *& dest, const char *src, const char *additional_separator)
{
  if (!src)
  {
    return;
  }
  if (!additional_separator)
  {
    additional_separator = "";
  }
  if (dest)
  {
    int old_len = strlen(dest);
    int additional_len = strlen(additional_separator);
    int whole_len = old_len + additional_len + strlen(src);
    char *new_attr = static_cast<char*>(malloc(whole_len + 1));
    assert(new_attr != NULL); // Just abort if out of memory!
    strcpy(new_attr, dest);
    strcpy(new_attr + old_len, additional_separator);
    strcpy(new_attr + old_len + additional_len, src);
    free(dest);
    dest = new_attr;
  }
  else
  {
    dest = icl_core::os::strdup(src);
  }
  m_changed = true;
}

AttributeTree* AttributeTree::commentAttributeTree()
{
  AttributeTree *loop = firstSubTree();
  while (loop)
  {
    if (loop->isComment() && loop->attribute())
    {
      return loop;
    }
    loop = nextSubTree(loop);
  }
  return NULL;
}

bool AttributeTree::isComment()
{
  return m_this_description && !strcmp(m_this_description, comment_str);
}

bool AttributeTree::isMultilineComment()
{
  return isComment() && (strchr(attribute(), '\n') != NULL);
}

const char* AttributeTree::comment()
{
  AttributeTree* sub_comment = commentAttributeTree();
  if (sub_comment)
  {
    return sub_comment->attribute();
  }
  else
  {
    return "";
  }
}

bool AttributeTree::hasMultilineComment()
{
  return strchr(comment(), '\n') != NULL;
}

void AttributeTree::setComment(const char *comment)
{
  setAttribute(comment_str, comment);
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! This function is used to unlink this tree from its parent.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::Unlink()
  {
    unlink();
  }

  /*! This function is used to unlink this tree's subtree.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::UnlinkSub()
  {
    unlinkSub();
  }

  /*! Sets an attribute.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::SetAttribute(const char *description,
                                             const char *attribute)
  {
    return setAttribute(description, attribute);
  }

  /*! Same function as above but with the main description already
   *  split off from the description string.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::SetAttribute(const char *description,
                                             const char *subdescription,
                                             const char *attribute)
  {
    return setAttribute(description, subdescription, attribute);
  }

  /*! Returns the attribute of this node.
   *  \deprecated Obsolete coding style.
   */
  char *AttributeTree::GetAttribute()
  {
    return getAttribute();
  }

  /*! Returns the attribute of a given description.
   *  \deprecated Obsolete coding style.
   */
  char *AttributeTree::GetAttribute(const char *description,
                                    const char *default_attribute,
                                    AttributeTree **subtree)
  {
    return getAttribute(description, default_attribute, subtree);
  }

  /*! Gets an attribute.
   *  \deprecated Obsolete coding style.
   */
  char *AttributeTree::GetOrSetDefault(const char *description,
                                       const char *default_attribute)
  {
    return getOrSetDefault(description, default_attribute);
  }

  /*! Saves the tree into a file.
   *  \deprecated Obsolete coding style.
   */
  int AttributeTree::Save(const char *filename, int change_style_depth,
                          bool unmark_changes)
  {
    return save(filename, change_style_depth, unmark_changes);
  }

  /*! Reads information from a file and stores it into a tree.
   *  \deprecated Obsolete coding style.
   */
  int AttributeTree::Load(const char *filename, bool unmark_changes,
                          bool process_include, bool load_comments,
                          bool preserve_order)
  {
    return load(filename, unmark_changes, process_include, load_comments, preserve_order);
  }

  /*! Reads information from a stream and stores it into a tree.
   *  \deprecated Obsolete coding style.
   */
  int AttributeTree::Get(std::istream &in, bool process_include,
                         bool load_comments, const FilePath *file_path)
  {
    return get(in, process_include, load_comments, file_path);
  }

  /*! Creates and returns a new node description.
   *  \deprecated Obsolete coding style.
   */
  char *AttributeTree::NewSubNodeDescription(const char *base_description)
  {
    return newSubNodeDescription(base_description);
  }

  /*! Adds a new sub-node using newSubNodeDescription().
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::AddNewSubTree()
  {
    return addNewSubTree();
  }

  /*! Prints all stored attributes of a subtree in an std::ostream.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::PrintSubTree(std::ostream &out,
                                   int change_style_depth,
                                   const char *upper_description)
  {
    printSubTree(out, change_style_depth, upper_description);
  }

  /*! Returns the description of this node.
   *  \deprecated Obsolete coding style.
   */
  const char *AttributeTree::Description() const
  {
    return getDescription();
  }

  /*! Returns the Attribute of this node.
   *  \deprecated Obsolete coding style.
   */
  const char *AttributeTree::Attribute() const
  {
    return attribute();
  }

  /*! Changes the description of this node.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::SetDescription(const char *description)
  {
    setDescription(description);
  }

  /*! Sets the Attribute of this AttributeTree.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::SetAttribute(const char *attribute)
  {
    setAttribute(attribute);
  }

  /*! Sets the Comment of this AttributeTree.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::SetComment(const char *comment)
  {
    setComment(comment);
  }

  /*! Returns the subtree with the given \a description.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::SubTree(const char *description)
  {
    return subTree(description);
  }

  /*! Returns the subtree with the given \a description if it exists.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::GetSubTree(const char *description)
  {
    return getSubTree(description);
  }

  /*! Get a pointer to the first subtree.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::FirstSubTree()
  {
    return firstSubTree();
  }

  /*! Get a pointer to the next subtree.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::NextSubTree(AttributeTree *subtree)
  {
    return nextSubTree(subtree);
  }

  /*! Get a pointer to the parent tree.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::ParentTree()
  {
    return parentTree();
  }

  /*! Inserts a tree as a subtree of this node.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::AddSubTree(AttributeTree *subtree)
  {
    return addSubTree(subtree);
  }

  /*! Inserts a tree as a next node of this node.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::AddNextTree(AttributeTree *nexttree)
  {
    return addNextTree(nexttree);
  }

  /*! Returns the AttributeTree found if there is any description or
   *  subdescription that matches the given description AND its
   *  attribute is equal to the given one.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::Search(const char *description, const char *attribute)
  {
    return search(description, attribute);
  }

  /*! Returns \c true if the description's attribute is equal to the
   *  given attribute.
   *  \deprecated Obsolete coding style.
   */
  bool AttributeTree::IsAttribute(const char *description, const char *attribute)
  {
    return isAttribute(description, attribute);
  }

  /*! Returns \c true if any changes occurred since the last call of
   *  unmarkChanges().
   *  \deprecated Obsolete coding style.
   */
  bool AttributeTree::Changed()
  {
    return changed();
  }

  /*! Sets all changed flags to \c false.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::UnmarkChanges()
  {
    return unmarkChanges();
  }

  /*! Returns the number of valid entries in the subtree.
   *  \deprecated Obsolete coding style.
   */
  int AttributeTree::Contains()
  {
    return contains();
  }

  /*! Returns the parent tree of this attribute tree.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::Parent()
  {
    return parent();
  }

  /*! Return the top-level attribute tree.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::Root()
  {
    return root();
  }

  /*! Returns the Comment of this Attribute Tree.
   *  \deprecated Obsolete coding style.
   */
  const char *AttributeTree::Comment()
  {
    return comment();
  }

  /*! Checks if the Attribute Tree has a comment.
   *  \deprecated Obsolete coding style.
   */
  bool AttributeTree::HasComment()
  {
    return hasComment();
  }

  /*! Checks if this Attribute Tree is a comment.
   *  \deprecated Obsolete coding style.
   */
  bool AttributeTree::IsComment()
  {
    return isComment();
  }

  /*! Checks if the Attribute Tree has a multiline comment.
   *  \deprecated Obsolete coding style.
   */
  bool AttributeTree::HasMultilineComment()
  {
    return hasMultilineComment();
  }

  /*! Checks if this Attribute Tree is a multiline comment.
   *  \deprecated Obsolete coding style.
   */
  bool AttributeTree::IsMultilineComment()
  {
    return isMultilineComment();
  }

  /*! Get the comment subtree.  If none exists NULL is returned.
   *  \deprecated Obsolete coding style.
   */
  AttributeTree *AttributeTree::CommentAttributeTree()
  {
    return commentAttributeTree();
  }

  /*! Append a string to the attribute.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::AppendAttribute(const char *attribute, const char *separator)
  {
    appendAttribute(attribute, separator);
  }

  /*! Append a string to the description.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::AppendDescription(const char *description, const char *separator)
  {
    appendDescription(description, separator);
  }

  /*! This function reverts the order of the AttributeTree.
   *  \deprecated Obsolete coding style.
   */
  void AttributeTree::RevertOrder()
  {
    revertOrder();
  }

  /*! This function directly returns the virtual attribute
   *  'm_file_path_str'.
   *  \deprecated Obsolete coding style.
   */
  const char *AttributeTree::GetFilePath()
  {
    return getFilePath();
  }

  /*! This function directly returns the filename this attribute
   *  belongs to.
   *  \deprecated Obsolete coding style.
   */
  const char *AttributeTree::FileName()
  {
    return fileName();
  }

#endif
/////////////////////////////////////////////////

}
}
