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
 * \date    2012-01-25
 *
 */
//----------------------------------------------------------------------
#include <icl_core_config/Config.h>

#include <iterator>
#include <vector>
#include <boost/assign/list_of.hpp>
#include <boost/test/unit_test.hpp>

namespace icc = icl_core::config;
namespace icl = icl_core::logging;

enum EnumValue
{
  ONE,
  TWO,
  THREE
};

std::vector<std::string> enum_value_description =
  boost::assign::list_of<std::string>("ONE")("TWO")("THREE");

char const * enum_value_char_description[] = {"ONE", "TWO", "THREE", NULL};

struct ConfigListEntry
{
  std::string string_value;
  EnumValue enum_value;
  struct Foo
  {
    uint32_t uint32_value;
    EnumValue enum_value;
    struct Bar {
      double double_value;
      EnumValue enum_value;
    } bar;
  } foo;
};

BOOST_AUTO_TEST_SUITE(ts_BatchGet)

#ifdef ICL_CORE_CONFIG_HAS_ENHANCED_CONFIG_MACROS

BOOST_AUTO_TEST_CASE(ConfigValue)
{
  // Simulate the following configuration file:
  //
  // <?xml version="1.0" encoding="UTF-8"?>
  //
  // <ConfigValue>
  //   <StringEntry>string-value</StringEntry>
  //   <UInt32Entry>12345</UInt32Entry>
  //   <DoubleEntry>1.2345</DoubleEntry>
  // </ConfigValue>
  icc::setValue("/ConfigValue/StringEntry", "string-value");
  icc::setValue("/ConfigValue/UInt32Entry", "12345");
  icc::setValue("/ConfigValue/DoubleEntry", "1.2345");

  std::string string_value = "";
  uint32_t uint32_value = 0;
  double double_value = 0.;
  bool read_success =
    icc::get(CONFIG_VALUES(CONFIG_VALUE("/ConfigValue/StringEntry", string_value)
                           CONFIG_VALUE("/ConfigValue/UInt32Entry", uint32_value)
                           CONFIG_VALUE("/ConfigValue/DoubleEntry", double_value)),
             icl::Nirwana::instance());
  BOOST_CHECK(read_success);

  BOOST_CHECK_EQUAL(string_value, "string-value");
  BOOST_CHECK_EQUAL(uint32_value, uint32_t(12345));
  BOOST_CHECK_EQUAL(double_value, double(1.2345));
}

BOOST_AUTO_TEST_CASE(ConfigValueDefault)
{
  // Simulate the following configuration file:
  //
  // <?xml version="1.0" encoding="UTF-8"?>
  //
  // <ConfigValueDefault>
  //   <StringEntry>string-value</StringEntry>
  // </ConfigValueDefault>
  icc::setValue("/ConfigValueDefault/StringEntry", "string-value");

  std::string string_value = "";
  uint32_t uint32_value = 0;
  bool read_success =
    icc::get(CONFIG_VALUES(
               CONFIG_VALUE_DEFAULT("/ConfigValueDefault/StringEntry", string_value, "other-string-value")
               CONFIG_VALUE_DEFAULT("/ConfigValue/UInt32Entry", uint32_value, 12345)),
             icl::Nirwana::instance());
  BOOST_CHECK(read_success);

  BOOST_CHECK_EQUAL(string_value, "string-value");
  BOOST_CHECK_EQUAL(uint32_value, uint32_t(12345));
}

BOOST_AUTO_TEST_CASE(ConfigEnum)
{
  // Simulate the following configuration file:
  //
  // <?xml version="1.0" encoding="UTF-8"?>
  //
  // <ConfigEnum>
  //   <Entry1>ONE</Entry1>
  //   <Entry2>TWO</Entry2>
  // </ConfigEnum>
  icc::setValue("/ConfigEnum/Entry1", "ONE");
  icc::setValue("/ConfigEnum/Entry2", "TWO");

  EnumValue value1 = THREE;
  EnumValue value2 = THREE;
  bool read_success =
    icc::get(CONFIG_VALUES(CONFIG_ENUM("/ConfigEnum/Entry1", value1, enum_value_char_description)
                           CONFIG_ENUM("/ConfigEnum/Entry2", value2, enum_value_char_description)),
             icl::Nirwana::instance());
  BOOST_CHECK(read_success);

  BOOST_CHECK_EQUAL(value1, ONE);
  BOOST_CHECK_EQUAL(value2, TWO);
}

BOOST_AUTO_TEST_CASE(ConfigEnumDefault)
{
  // Simulate the following configuration file:
  //
  // <?xml version="1.0" encoding="UTF-8"?>
  //
  // <ConfigEnumDefault>
  //   <Entry1>ONE</Entry1>
  // </ConfigEnumDefault>
  icc::setValue("/ConfigEnumDefault/Entry1", "ONE");

  EnumValue value1 = THREE;
  EnumValue value2 = THREE;
  bool read_success =
    icc::get(CONFIG_VALUES(
               CONFIG_ENUM_DEFAULT("/ConfigEnumDefault/Entry1", value1, TWO, enum_value_char_description)
               CONFIG_ENUM_DEFAULT("/ConfigEnumDefault/Entry2", value2, TWO, enum_value_char_description)),
             icl::Nirwana::instance());
  BOOST_CHECK(read_success);

  BOOST_CHECK_EQUAL(value1, ONE);
  BOOST_CHECK_EQUAL(value2, TWO);
}

BOOST_AUTO_TEST_CASE(ConfigPrefix)
{
  // Simulate the following configuration file:
  //
  // <?xml version="1.0" encoding="UTF-8"?>
  //
  // <ConfigPrefix>
  //   <StringEntry>string-value</StringEntry>
  //   <UInt32Value>12345</UInt32Value>
  //   <DoubleEntry>1.2345</DoubleEntry>
  //   <EnumEntry>TWO</EnumEntry>
  // </ConfigPrefix>
  icc::setValue("/ConfigPrefix/StringEntry", "string-value");
  icc::setValue("/ConfigPrefix/UInt32Entry", "12345");
  icc::setValue("/ConfigPrefix/DoubleEntry", "1.2345");
  icc::setValue("/ConfigPrefix/EnumEntry",   "TWO");

  std::string string_value = "";
  uint32_t uint32_value = 0;
  double double_value = 0.;
  EnumValue enum_value = ONE;
  bool read_success =
    icc::get("/ConfigPrefix",
             CONFIG_VALUES(CONFIG_VALUE("StringEntry", string_value)
                           CONFIG_VALUE("UInt32Entry", uint32_value)
                           CONFIG_VALUE("DoubleEntry", double_value)
                           CONFIG_ENUM ("EnumEntry",   enum_value, enum_value_char_description)),
             icl::Nirwana::instance());
  BOOST_CHECK(read_success);

  BOOST_CHECK_EQUAL(string_value, "string-value");
  BOOST_CHECK_EQUAL(uint32_value, uint32_t(12345));
  BOOST_CHECK_EQUAL(double_value, double(1.2345));
  BOOST_CHECK_EQUAL(enum_value,   TWO);
}

BOOST_AUTO_TEST_CASE(ConfigList)
{
  // Simulate the following configuration file:
  //
  // <?xml version="1.0" encoding="UTF-8"?>
  //
  // <ConfigList>
  //   <List>
  //     <Entry1>
  //       <StringEntry>string-value-1</StringEntry>
  //       <EnumEntry>ONE</EnumEntry>
  //       <Foo>
  //         <UInt32Value>1</UInt32Value>
  //         <EnumEntry>ONE</EnumEntry>
  //         <Bar>
  //           <DoubleEntry>0.1</DoubleEntry>
  //           <EnumEntry>ONE</EnumEntry>
  //         </Bar>
  //       </Foo>
  //     </Entry1>
  //     <Entry2>
  //       <StringEntry>string-value-2</StringEntry>
  //       <EnumEntry>TWO</EnumEntry>
  //       <Foo>
  //         <UInt32Value>2</UInt32Value>
  //         <EnumEntry>TWO</EnumEntry>
  //         <Bar>
  //           <DoubleEntry>0.2</DoubleEntry>
  //           <EnumEntry>TWO</EnumEntry>
  //         </Bar>
  //       </Foo>
  //     </Entry2>
  //     <Entry3>
  //       <StringEntry>string-value-3</StringEntry>
  //       <EnumEntry>THREE</EnumEntry>
  //       <Foo>
  //         <UInt32Value>3</UInt32Value>
  //         <EnumEntry>TWO</EnumEntry>
  //         <Bar>
  //           <DoubleEntry>0.3</DoubleEntry>
  //           <EnumEntry>TWO</EnumEntry>
  //         </Bar>
  //       </Foo>
  //     </Entry3>
  //   </List>
  // </ConfigValuePrefix>
  icc::setValue("/ConfigList/List/Entry1/StringEntry",         "string-value-1");
  icc::setValue("/ConfigList/List/Entry1/EnumEntry",           "ONE");
  icc::setValue("/ConfigList/List/Entry1/Foo/UInt32Entry",     "1");
  icc::setValue("/ConfigList/List/Entry1/Foo/EnumEntry",       "ONE");
  icc::setValue("/ConfigList/List/Entry1/Foo/Bar/DoubleEntry", "0.1");
  icc::setValue("/ConfigList/List/Entry1/Foo/Bar/EnumEntry",   "ONE");
  icc::setValue("/ConfigList/List/Entry2/StringEntry",         "string-value-2");
  icc::setValue("/ConfigList/List/Entry2/EnumEntry",           "TWO");
  icc::setValue("/ConfigList/List/Entry2/Foo/UInt32Entry",     "2");
  icc::setValue("/ConfigList/List/Entry2/Foo/EnumEntry",       "TWO");
  icc::setValue("/ConfigList/List/Entry2/Foo/Bar/DoubleEntry", "0.2");
  icc::setValue("/ConfigList/List/Entry2/Foo/Bar/EnumEntry",   "TWO");
  icc::setValue("/ConfigList/List/Entry3/StringEntry",         "string-value-3");
  icc::setValue("/ConfigList/List/Entry3/EnumEntry",           "THREE");
  icc::setValue("/ConfigList/List/Entry3/Foo/UInt32Entry",     "3");
  icc::setValue("/ConfigList/List/Entry3/Foo/EnumEntry",       "THREE");
  icc::setValue("/ConfigList/List/Entry3/Foo/Bar/DoubleEntry", "0.3");
  icc::setValue("/ConfigList/List/Entry3/Foo/Bar/EnumEntry",   "THREE");

  // Remard: This also works with std::list.
  // Actually, every container that provides a push_back() method can be used!
  std::vector<ConfigListEntry> config_list;
  bool read_successful =
    icc::get(CONFIG_VALUES(
               CONFIG_LIST(
                 ConfigListEntry, "/ConfigList/List",
                 MEMBER_MAPPING(
                   ConfigListEntry,
                   MEMBER_VALUE_1("StringEntry", ConfigListEntry, string_value)
                   MEMBER_VALUE_2("Foo/UInt32Entry", ConfigListEntry, foo, uint32_value)
                   MEMBER_VALUE_3("Foo/Bar/DoubleEntry", ConfigListEntry, foo, bar, double_value)
                   MEMBER_ENUM_1 ("EnumEntry", ConfigListEntry, enum_value, enum_value_description)
                   MEMBER_ENUM_2 ("Foo/EnumEntry", ConfigListEntry, foo, enum_value, enum_value_description)
                   MEMBER_ENUM_3 ("Foo/Bar/EnumEntry",   ConfigListEntry, foo, bar, enum_value,
                                  // Enum members also work with plain C-string arrays.
                                  enum_value_char_description)),
                 std::back_inserter(config_list))),
             icl::Nirwana::instance());
  BOOST_CHECK(read_successful);

  BOOST_CHECK_EQUAL(config_list.size(), 3u);

  BOOST_CHECK_EQUAL(config_list[0].string_value, "string-value-1");
  BOOST_CHECK_EQUAL(config_list[0].foo.uint32_value, 1u);
  BOOST_CHECK_EQUAL(config_list[0].foo.bar.double_value, 0.1);
  BOOST_CHECK_EQUAL(config_list[0].enum_value,   ONE);
  BOOST_CHECK_EQUAL(config_list[0].foo.enum_value,   ONE);
  BOOST_CHECK_EQUAL(config_list[0].foo.bar.enum_value,   ONE);

  BOOST_CHECK_EQUAL(config_list[1].string_value, "string-value-2");
  BOOST_CHECK_EQUAL(config_list[1].foo.uint32_value, 2u);
  BOOST_CHECK_EQUAL(config_list[1].foo.bar.double_value, 0.2);
  BOOST_CHECK_EQUAL(config_list[1].enum_value,   TWO);
  BOOST_CHECK_EQUAL(config_list[1].foo.enum_value,   TWO);
  BOOST_CHECK_EQUAL(config_list[1].foo.bar.enum_value,   TWO);

  BOOST_CHECK_EQUAL(config_list[2].string_value, "string-value-3");
  BOOST_CHECK_EQUAL(config_list[2].foo.uint32_value, 3u);
  BOOST_CHECK_EQUAL(config_list[2].foo.bar.double_value, 0.3);
  BOOST_CHECK_EQUAL(config_list[2].enum_value,   THREE);
  BOOST_CHECK_EQUAL(config_list[2].foo.enum_value,   THREE);
  BOOST_CHECK_EQUAL(config_list[2].foo.bar.enum_value,   THREE);
}

#else
# if defined(_MSC_VER)
#  pragma message("The icl_core_config batch convenience macros are only available in Visual Studio 2010 and newer.")
# endif
#endif

BOOST_AUTO_TEST_SUITE_END()
