// this is a -*- C++ -*- file

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
 * \date   2001-01-11
 *
 * \brief Reads a configuration file in the old MCA attribute tree format.
 *
 * \b icl_core::config::AttributeTree
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_ATTRIBUTE_TREE_H_INCLUDED
#define ICL_CORE_CONFIG_ATTRIBUTE_TREE_H_INCLUDED

#include <fstream>
#include <string>

#include "icl_core/BaseTypes.h"
#include "icl_core_config/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace config {

class AttributeTree;

class FilePath
{
public:
  FilePath(const char *filename = "")
  {
    init(filename);
  }
  FilePath(const icl_core::String& filename)
  {
    init(filename.c_str());
  }

  ~FilePath()
  { }

  /*! Returns the absolute path of the file without the file name.
   */
  icl_core::String path() const
  {
    return icl_core::String(m_file, 0, m_file_path_name_split);
  }

  /*! Returns the name of the file without path.
   */
  icl_core::String name() const
  {
    return icl_core::String(m_file, m_file_path_name_split, icl_core::String::npos);
  }

  /*! Returns the path and name of the file.
   */
  icl_core::String absoluteName() const
  {
    return m_file;
  }

  /*! Returns the absolute path of the current directory.  The last
   *  character of the returned string is a '/'.
   */
  icl_core::String currentDir() const
  {
    return m_pwd;
  }

  /*! Returns the extension of the file.
   */
  icl_core::String extension() const
  {
    return icl_core::String(m_file, m_file_name_extension_split, icl_core::String::npos);
  }

  /*! Returns the absolute path of the file \a filename.  Given
   *  relative filenames start from the current directory.
   */
  icl_core::String absolutePath(const icl_core::String& filename) const;

  /*! Returns \c true if the \a filename is a relative path (It does
   *  not begin with a '/').
   */
  static bool isRelativePath(const icl_core::String& filename);

  /*! Returns \c true if the objects filename is a relative path.
   */
  bool isRelativePath() const
  {
    return FilePath::isRelativePath(m_file);
  }

  /*! Returns the normalized path of the given \a filename.
   *  Normalized means: all leading "./" are removed, all "/./" are
   *  reduced to "/" and all "something/../" are reduced to "/".
   */
  static icl_core::String normalizePath(const icl_core::String& filename);

  /*! Searches for directory separators, which are not supported by
   *  the underlying operation system and exchanges these with
   *  separators, that are supported.
   *
   *  \li Windows: '/' -> '\'
   *  \li Linux: '\' -> '/'
   */
  static icl_core::String exchangeSeparators(const icl_core::String& filename);

  /*! Returns the value of the given environment variable \a var_name.
   */
  static icl_core::String getEnvironment(const icl_core::String& var_name);

  /*! Replaces environment variables in the given filename.  An
   *  environment variable must have the following form:
   *  '${VARIABLE_NAME}'
   */
  static icl_core::String replaceEnvironment(const icl_core::String& filename);

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Returns the absolute path of the file without the file name.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String Path() const ICL_CORE_GCC_DEPRECATE_STYLE;
  /*! Returns the name of the file without path.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String Name() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the path and name of the file.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String AbsoluteName() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the absolute path of the current directory.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String CurrentDir() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the extension of the file.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String Extension() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the absolute path of the file \a filename.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String AbsolutePath(const icl_core::String& filename) const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns \c true if the \a filename is a relative path.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE static bool IsRelativePath(const icl_core::String& filename)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns \c true if the objects filename is a relative path.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsRelativePath() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the normalized path of the given \a filename.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE static icl_core::String NormalizePath(const icl_core::String& filename)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Searches for directory separators, which are not supported by
   *  the underlying operation system and exchanges these with
   *  separators, that are supported.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE static icl_core::String ExchangeSeparators(const icl_core::String& filename)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the value of the given environment variable \a var_name.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE static icl_core::String GetEnvironment(const icl_core::String& var_name)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Replaces environment variables in the given filename.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE static icl_core::String ReplaceEnvironment(const icl_core::String& filename)
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  void init(const char *filename);

  icl_core::String m_pwd;
  icl_core::String m_file;
  icl_core::String::size_type m_file_path_name_split;
  icl_core::String::size_type m_file_name_extension_split;
};


/*! Only for internal use in AttributeTree.  Realises a list of
 *  AttributeTrees.
 */
class ICL_CORE_CONFIG_IMPORT_EXPORT SubTreeList
{
  friend class AttributeTree;

public:
  SubTreeList(AttributeTree *sub_tree = 0, SubTreeList *next = 0);
  ~SubTreeList();
  void copy(AttributeTree *parent);
  void unlinkParent();
  void unlink(AttributeTree *obsolete_tree);
  AttributeTree *subTree(const char *description);
  int contains();
  void printSubTree(std::ostream& out, int change_style_depth, char *upper_description);
  void printXmlSubTree(std::ostream& out, const std::string& upper_description);
  bool changed();
  AttributeTree *search(const char *description, const char *attribute);
  AttributeTree *next(AttributeTree *prev);
  void unmarkChanges();

  /*! Reverts the order of the sub tree list.
   */
  SubTreeList *revertOrder(SubTreeList *new_next = NULL);

  /*! Creates new SubTreeList with \a new_tree as subtree after the
   *  next tree with subtree \a after.
   */
  void newSubTreeList(AttributeTree *new_tree, AttributeTree *after);

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Reverts the order of the sub tree list.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE SubTreeList *RevertOrder(SubTreeList *new_next = NULL)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Creates new SubTreeList with \a new_tree as subtree after the
   *  next tree with subtree \a after.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void NewSubTreeList(AttributeTree *new_tree, AttributeTree *after)
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  // Forbid copying.
  SubTreeList(const SubTreeList&);
  SubTreeList& operator = (const SubTreeList&);

  SubTreeList *m_next;
  AttributeTree *m_sub_tree;
};

/*! This class can be used to store attribute information in a tree
 *  structure.  The information can be stored and requested by
 *  browsing the tree or by directly giving a description.
 *
 *  Descriptions have the following format:
 *  \li node descriptions are separated by points,
 *      e.g. "Pushbutton.1.geometry.x".
 *  \li When saved to a file the attributes are separated by a colon
 *      (':'), e.g.  "Pushbutton.1.geometry.x:121".
 *
 *  Examples for setting or reading attributes:
 *  \code
 *  at->setAttribute("Pushbutton.1.geometry.x","135");
 *  \endcode
 *  When getting an attribute that is not stored in the tree, a
 *  default value can be specified to be returned in that case.
 *  \code
 *  at->getAttribute("Pushbutton.1.geometry.x","120");
 *  \endcode
 *  To store the default attribute if the given description can not be
 *  found in the tree, another function can be used:
 *  \code
 *  at->getOrSetDefault("Pushbutton.1.geometry.x","120");
 *  \endcode
 *  \note All attributes are stored as strings.
 */
class ICL_CORE_CONFIG_IMPORT_EXPORT AttributeTree
{
  friend class SubTreeList;

public:
  /*! Save and Load error codes.
   */
  enum
  {
    eOK,              //!< All OK.
    eEMPTY_TREE,      //!< Tree is empty.
    eFILE_SAVE_ERROR, //!< Save Error.
    eFILE_LOAD_ERROR  //!< Load Error.
  };
  /*! Creates a new empty Tree.
   */
  AttributeTree(const char *description = 0, AttributeTree *parent = 0);

  /*! Creates new Tree with same attributes and subtree as the \a
   *  other tree, i.e. this function copies a subtree!
   */
  AttributeTree(const AttributeTree& other);

  /*! Creates new Tree with same attribute and subtree as the \a other
   *  tree, i.e. this function copies a subtree!
   *
   *  Attention: The Tree is inserted into the list of specified
   *  parent (if given).  There is no check whether another subtree
   *  with same description exists or not!
   */
  AttributeTree(const AttributeTree& other, AttributeTree *parent);

  /*! Deletes tree including its subtrees.  The parent is informed
   *  about this deletion (the sub_tree pointer is set to 0).
   */
  ~AttributeTree();

  /*! This function is used to unlink this tree from its parent.
   *  Deleting the parent afterwards does not delete this subtree.
   */
  void unlink();

  /*! This function is used to unlink this tree's subtree.  Deleting
   *  this afterwards does not delete the subtree.
   */
  void unlinkSub();

  /*! Returns a subtree that fits the given description.  Additional
   *  subtrees are created if necessary.  This inline function just
   *  calls setAttribute(\a description, 0).
   */
  // inline AttributeTree *getSubTree(const char *description)
  // { return getOrCreateSubTree()->setAttribute(description, 0); }

  /*! Sets an attribute.  Additional subtrees are created if
   *  necessary.  The return value is a direct pointer to the subtree
   *  entry that contains the attribute.  If \a attribute is 0, it is
   *  not set, but the return value points to the corresponding entry
   *  in the tree.
   */
  AttributeTree *setAttribute(const char *description, const char *attribute);

  /*! Same function as above but with the main description already
   *  split off from the description string.
   */
  AttributeTree *setAttribute(const char *description, const char *subdescription, const char *attribute);

  /*! Set an attribute in the subtree of this tree.  This is used to
   *  set an attribute in the tree without knowing the root
   *  descriptor.
   */
  // inline AttributeTree *setSubAttribute(const char *description, const char *attribute)
  // { return setAttribute(m_this_description, description, attribute); }

  /*! Returns the attribute of this node.
   */
  char *getAttribute()
  {
    return m_this_attribute;
  }

  /*! Returns the attribute of a given description.  If the
   *  description is not stored in the tree, the default_attribute is
   *  returned.
   */
  char *getAttribute(const char *description, const char *default_attribute = 0,
                     AttributeTree **subtree = 0);

  /*! Gets an attribute. If the attribute is not set in the tree, the
   *  given default-value is saved into the tree.
   */
  char *getOrSetDefault(const char *description, const char *default_attribute);

  /*! Saves the tree into a file.  If unmark_changes is \c true, all
   *  changed flags will be set to \c false.
   */
  int save(const char *filename, int change_style_depth = 3, bool unmark_changes = true);

  /*! Reads information from a file and stores it into a tree.  If
   *  unmark_changes is \c true, all changed flags will be set to \c
   *  false.
   *
   *  \param filename Name of the file from which to load the
   *         information.
   *  \param unmark_changes If \c true, all changed flags will be set
   *         to \c false.
   *  \param process_include If \c true, process include directives in
   *         the file.
   *  \param load_comments If \c true, load comments as tree nodes
   *         rather than ignoring them.  This way the comments remain
   *         intact when saving the tree back to a file.
   *  \param preserve_order if this is set \a true the same tree order
   *         than on save is preserved.
   *  \note this reverts the internal list after loading
   */
  int load(const char *filename, bool unmark_changes = true, bool process_include = true,
           bool load_comments = false, bool preserve_order = false);

  /*! Reads information from a stream and stores it into a tree.
   */
  int get(std::istream &in, bool process_include = true, bool load_comments = false,
          const FilePath *file_path = NULL);

  /*! Creates and returns a new node description.  This description
   *  contains a so far unused number.  You have to delete it if you
   *  don't use it anymore.  Note: setDescription() makes an own copy
   *  of a given string on the heap!  You normally don't use this, use
   *  addNewSubTree to create new subtrees.  Creates and returns a new
   *  sub-node description. This description contains a so far unused
   *  number.  You have to delete it if you dont use it anymore.
   *
   *  \param base_description the base description for the creation of
   *         the new one
   */
  char *newSubNodeDescription(const char *base_description = "");

  /*! Adds a new sub-node using newSubNodeDescription().
   */
  AttributeTree *addNewSubTree();

  /*! Prints all stored attributes of a subtree in an std::ostream.
   *  All descriptions will be appended to upper_description before
   *  being printed.
   */
  void printSubTree(std::ostream &out, int change_style_depth = 3, const char *upper_description = 0);

  /*! Returns the description of this node.
   */
  const char *getDescription() const
  {
    return m_this_description;
  }

  /*! Returns the Attribute of this node.
   */
  const char *attribute() const
  {
    return m_this_attribute;
  }

  /*! Changes the description of this node.
   */
  void setDescription(const char *description);

  /*! Sets the Attribute of this AttributeTree.
   */
  void setAttribute(const char *attribute);

  /*! Sets the Comment of this AttributeTree.
   */
  void setComment(const char *comment);

  /*! Returns the subtree with the given \a description.  Creates a
   *  new one if necessary.
   */
  AttributeTree *subTree(const char *description);

  /*! Returns the subtree with the given \a description if it exists.
   *  Returns NULL if not.
   */
  AttributeTree *getSubTree(const char *description);

  /*! Get a pointer to the first subtree.  This function can be used
   *  to browse the whole tree.
   */
  AttributeTree *firstSubTree()
  {
    if (m_subtree_list)
    {
      return m_subtree_list->m_sub_tree;
    }
    else
    {
      return NULL;
    }
  }

  /*! Get a pointer to the next subtree.  This function can be used to
   *  browse the whole tree.
   */
  AttributeTree *nextSubTree(AttributeTree *subtree)
  {
    if (m_subtree_list)
    {
      return m_subtree_list->next(subtree);
    }
    else
    {
      return NULL;
    }
  }

  /*! Get a pointer to the parent tree.  Returns NULL if this tree has
   *  no parent.
   */
  AttributeTree *parentTree()
  {
    return m_parent;
  }

  /*! Inserts a tree as a subtree of this node.  If the tree's
   *  description is already used by another subtree, the description
   *  is set to a new unused description by using
   *  newSubNodeDescription().
   */
  AttributeTree *addSubTree(AttributeTree *subtree)
  {
    return addSubTree(subtree, NULL);
  }

  /*! Inserts a tree as a next node of this node.  If the tree's
   *  description is already used by another node on that level, the
   *  description is set to a new unused description by using
   *  newSubNodeDescription().
   *  \note This is only possible if this object has a parent.  On
   *  failure NULL is returned.
   */
  AttributeTree *addNextTree(AttributeTree *nexttree)
  {
    if (m_parent)
    {
      return m_parent->addSubTree(nexttree, this);
    }
    else
    {
      return NULL;
    }
  }

  /*! Returns the AttributeTree found if there is any description or
   *  subdescription that matches the given description AND its
   *  attribute is equal to the given one.  NULL is returned if
   *  nothing is found.  If the given attribute is NULL, only the
   *  match of the descriptions is considered.
   */
  AttributeTree *search(const char *description, const char *attribute);

  /*! Returns \c true if the description's attribute is equal to the
   *  given attribute.
   */
  bool isAttribute(const char *description, const char *attribute = 0);

  /*! Returns \c true if any changes occurred since the last call of
   *  unmarkChanges().
   */
  bool changed();

  /*! Sets all changed flags to \c false.
   */
  void unmarkChanges();

  /*! Returns the number of valid entries in the subtree.
   */
  int contains();

  /*! Returns the parent tree of this attribute tree.
   */
  AttributeTree *parent()
  {
    return m_parent;
  }

  /*! Return the top-level attribute tree.
   */
  AttributeTree *root()
  {
    AttributeTree *at = this;
    for (; at->m_parent; at = at->m_parent) {}
    return at;
  }

  /*! Returns the Comment of this Attribute Tree.  If no comment
   *  subtree exists "" is returned.
   */
  const char *comment();

  /*! Checks if the Attribute Tree has a comment.
   */
  bool hasComment()
  {
    return commentAttributeTree() != NULL;
  }

  /*! Checks if this Attribute Tree is a comment.
   */
  bool isComment();

  /*! Checks if the Attribute Tree has a multiline comment.
   */
  bool hasMultilineComment();

  /*! Checks if this Attribute Tree is a multiline comment.
   */
  bool isMultilineComment();

  /*! Get the comment subtree.  If none exists NULL is returned.
   */
  AttributeTree *commentAttributeTree();

  /*! Append a string to the attribute.
   *  \param attribute the string to be appended
   *  \param separator an additional seperator string which is added
   *         in between (e.g. "\n")
   */
  void appendAttribute(const char *attribute, const char *separator = "")
  {
    appendString(m_this_attribute, attribute, separator);
  }

  /*! Append a string to the description.
   *  \param description the string to be appended
   *  \param separator an additional seperator string which is added
   *         in between (e.g. "\n")
   */
  void appendDescription(const char *description, const char *separator = "")
  {
    appendString(m_this_description, description, separator);
  }

  /*! This function reverts the order of the AttributeTree.
   */
  void revertOrder()
  {
    if (m_subtree_list)
    {
      m_subtree_list = m_subtree_list->revertOrder();
    }
  }

  /*! On loading of an attribute tree a virtual attribute with name
   *  'm_file_path_str' is created storing the absolute path to the
   *  loaded attribute tree file.  By Calling
   *  'GetAttribute(AttributeTree::m_file_path_str, "")' this virtual
   *  attribute is returned.  If you are loading more than one
   *  attribute tree file (e.g. by includes), the path stored in here
   *  is the one of the last loaded file.
   */
  static const char *m_file_path_str;

  /*! This function directly returns the virtual attribute
   *  'm_file_path_str'.
   */
  const char *getFilePath()
  {
    const char *ret = root()->getSpecialAttribute(m_file_path_str);
    if (!ret)
    {
      ret = "";
    }
    return ret;
  }

  /*! On loading of an attribute tree a virtual attribute with name
   *  'm_file_name_str' is created storing the filename of the loaded
   *  attribute tree file (without path).  By Calling
   *  'GetAttribute(AttributeTree::m_file_name_str, "")' this virtual
   *  attribute is returned.  If you are loading more than one
   *  attribute tree file (e.g. by includes), the file stored in here
   *  is the one of the last loaded file.
   */
  static const char *m_file_name_str;

  /*! This function directly returns the filename this attribute
   *  belongs to.  This corresponds to the virtual attribute
   *  'm_file_name_str'.
   */
  const char *fileName()
  {
    const char *ret = root()->getSpecialAttribute(m_file_name_str);
    if (!ret)
    {
      ret = "";
    }
    return ret;
  }

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! This function is used to unlink this tree from its parent.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Unlink() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! This function is used to unlink this tree's subtree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void UnlinkSub() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Sets an attribute.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *SetAttribute(const char *description,
                                                          const char *attribute)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Same function as above but with the main description already
   *  split off from the description string.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *SetAttribute(const char *description,
                                                          const char *subdescription,
                                                          const char *attribute)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the attribute of this node.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE char *GetAttribute() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the attribute of a given description.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE char *GetAttribute(const char *description,
                                                 const char *default_attribute = 0,
                                                 AttributeTree **subtree = 0)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Gets an attribute.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE char *GetOrSetDefault(const char *description,
                                                    const char *default_attribute)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Saves the tree into a file.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int Save(const char *filename, int change_style_depth = 3,
                                       bool unmark_changes = true)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Reads information from a file and stores it into a tree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int Load(const char *filename, bool unmark_changes = true,
                                       bool process_include = true,
                                       bool load_comments = false,
                                       bool preserve_order = false)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Reads information from a stream and stores it into a tree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int Get(std::istream &in, bool process_include = true,
                                      bool load_comments = false,
                                      const FilePath *file_path = NULL)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Creates and returns a new node description.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE char *NewSubNodeDescription(const char *base_description = "")
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Adds a new sub-node using newSubNodeDescription().
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *AddNewSubTree() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Prints all stored attributes of a subtree in an std::ostream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void PrintSubTree(std::ostream &out,
                                                int change_style_depth = 3,
                                                const char *upper_description = 0)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the description of this node.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE const char *Description() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the Attribute of this node.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE const char *Attribute() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Changes the description of this node.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void SetDescription(const char *description)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Sets the Attribute of this AttributeTree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void SetAttribute(const char *attribute) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Sets the Comment of this AttributeTree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void SetComment(const char *comment) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the subtree with the given \a description.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *SubTree(const char *description)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the subtree with the given \a description if it exists.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *GetSubTree(const char *description)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get a pointer to the first subtree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *FirstSubTree() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get a pointer to the next subtree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *NextSubTree(AttributeTree *subtree)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get a pointer to the parent tree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *ParentTree() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Inserts a tree as a subtree of this node.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *AddSubTree(AttributeTree *subtree)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Inserts a tree as a next node of this node.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *AddNextTree(AttributeTree *nexttree)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the AttributeTree found if there is any description or
   *  subdescription that matches the given description AND its
   *  attribute is equal to the given one.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *Search(const char *description, const char *attribute)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns \c true if the description's attribute is equal to the
   *  given attribute.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsAttribute(const char *description, const char *attribute = 0)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns \c true if any changes occurred since the last call of
   *  unmarkChanges().
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Changed() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Sets all changed flags to \c false.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void UnmarkChanges() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the number of valid entries in the subtree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int Contains() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the parent tree of this attribute tree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *Parent() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Return the top-level attribute tree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *Root() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the Comment of this Attribute Tree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE const char *Comment() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Checks if the Attribute Tree has a comment.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool HasComment() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Checks if this Attribute Tree is a comment.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsComment() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Checks if the Attribute Tree has a multiline comment.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool HasMultilineComment() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Checks if this Attribute Tree is a multiline comment.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsMultilineComment() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the comment subtree.  If none exists NULL is returned.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE AttributeTree *CommentAttributeTree() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Append a string to the attribute.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AppendAttribute(const char *attribute, const char *separator = "")
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Append a string to the description.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AppendDescription(const char *description, const char *separator = "")
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! This function reverts the order of the AttributeTree.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RevertOrder() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! This function directly returns the virtual attribute
   *  'm_file_path_str'.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE const char *GetFilePath() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! This function directly returns the filename this attribute
   *  belongs to.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE const char *FileName() ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////


private:
  void split(char *&description, char *&sub_description);
  void appendString(char * &dest, const char *src, const char *separator);

  // Forbid copying.
  AttributeTree& operator = (const AttributeTree&);

  /*! This function searches for special attributes up to the root.
   */
  char *getSpecialAttribute(const char *description, AttributeTree **subtree = NULL);

  /*! This function implements addSubTree and addNextTree.
   */
  AttributeTree *addSubTree(AttributeTree *subtree, AttributeTree *after_node);

  AttributeTree *m_parent;
  SubTreeList *m_subtree_list;

  char *m_this_description;
  char *m_this_attribute;

  bool m_changed;
};

}
}

#endif
