//===--------------- UserDefinedRules.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_USER_DEFINED_RULES_H
#define DPCT_USER_DEFINED_RULES_H
#include "Utility.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/YAMLTraits.h"
#include <string>
#include <vector>

enum RuleKind {
  API,
  DataType,
  Macro,
  Header,
  TypeRule,
  Class,
  Enum,
  DisableAPIMigration,
  PatternRewriter,
  CMakeRule,
  HelperFunction,
  PythonRule
};

enum RulePriority { Takeover, Default, Fallback };
enum RuleMatchMode { Partial, Full, StrictFull };

struct TypeNameRule {
  std::string NewName;
  clang::dpct::HelperFeatureEnum RequestFeature;
  RulePriority Priority;
  RuleMatchMode MatchMode;
  std::string BuildScriptSyntax;
  std::vector<std::string> Includes;
  TypeNameRule(std::string Name)
      : NewName(Name), RequestFeature(clang::dpct::HelperFeatureEnum::none),
        Priority(RulePriority::Fallback), MatchMode(RuleMatchMode::Partial) {}
  TypeNameRule(std::string Name, clang::dpct::HelperFeatureEnum Feature,
               RulePriority Priority = RulePriority::Fallback,
               RuleMatchMode MatchMode = RuleMatchMode::Partial)
      : NewName(Name), RequestFeature(Feature), Priority(Priority),
        MatchMode(MatchMode) {}
};

struct ClassFieldRule : public TypeNameRule {
  std::string SetterName;
  std::string GetterName;
  ClassFieldRule(std::string Name) : TypeNameRule(Name) {}
  ClassFieldRule(std::string Name, clang::dpct::HelperFeatureEnum Feature,
                 RulePriority Priority = RulePriority::Fallback,
                 RuleMatchMode MatchMode = RuleMatchMode::Partial)
      : TypeNameRule(Name, Feature) {}
  ClassFieldRule(std::string SetterName, std::string GetterName,
                 clang::dpct::HelperFeatureEnum Feature,
                 RulePriority Priority = RulePriority::Fallback,
                 RuleMatchMode MatchMode = RuleMatchMode::Partial)
      : TypeNameRule(SetterName, Feature), SetterName(SetterName),
        GetterName(GetterName) {}
};

struct EnumNameRule : public TypeNameRule {
  EnumNameRule(std::string Name) : TypeNameRule(Name) {}
  EnumNameRule(std::string Name, clang::dpct::HelperFeatureEnum Feature,
                 RulePriority Priority = RulePriority::Fallback,
                 RuleMatchMode MatchMode = RuleMatchMode::Partial)
      : TypeNameRule(Name, Feature) {}
};

// Record all information of imported rules
class MetaRuleObject {
public:
  struct ClassField {
    std::string In;
    std::string Out;
    std::string OutGetter;
    std::string OutSetter;
    ClassField() {}
  };
  struct ClassMethod {
    std::string In;
    std::string Out;
    ClassMethod() {}
  };
  struct Attributes {
    bool ReplaceCalleeNameOnly = false;
    bool HasExplicitTemplateArgs = false;
    int NumOfTemplateArgs = -1;
  };
  struct APIRestrictCondition {
    int ArgCount = -1;
  };

  struct PatternRewriter {
    std::string In = "";
    std::string Out = "";
    RuleMatchMode MatchMode = RuleMatchMode::Partial;
    std::string Warning = "";
    std::string BuildScriptSyntax = "";
    std::string RuleId = "";
    RulePriority Priority = RulePriority::Default;
    std::map<std::string, PatternRewriter> Subrules;
    PatternRewriter(){};

    PatternRewriter &operator=(const PatternRewriter &PR);
    PatternRewriter(const PatternRewriter &PR);
    PatternRewriter(const std::string &I, const std::string &O,
                    const std::map<std::string, PatternRewriter> &S,
                    RuleMatchMode MatchMode, std::string Warning,
                    std::string RuleId, std::string BuildScriptSyntax,
                    RulePriority Priority);
  };

  static std::vector<clang::tooling::UnifiedPath> RuleFiles;
  clang::tooling::UnifiedPath RuleFile;
  std::string RuleId;
  RulePriority Priority;
  RuleMatchMode MatchMode;
  std::string Warning;
  std::string BuildScriptSyntax;
  RuleKind Kind;
  std::string In;
  std::string Out;
  std::string EnumName;
  std::string Prefix;
  std::string Postfix;
  Attributes RuleAttributes;
  std::vector<std::string> Includes;
  std::vector<std::shared_ptr<ClassField>> Fields;
  std::vector<std::shared_ptr<ClassMethod>> Methods;
  std::map<std::string, PatternRewriter> Subrules;
  APIRestrictCondition RuleAPIRestrictCondition;
  MetaRuleObject()
      : Priority(RulePriority::Default), MatchMode(RuleMatchMode::Partial),
        Kind(RuleKind::API) {}
  MetaRuleObject(std::string id, RulePriority priority, RuleKind kind,
                 RuleMatchMode MatchMode)
      : RuleId(id), Priority(priority), MatchMode(MatchMode), Warning{Warning},
        Kind(kind) {}
  static void setRuleFiles(clang::tooling::UnifiedPath File) {
    RuleFiles.push_back(File);
  }
};

template <>
struct llvm::yaml::CustomMappingTraits<
    std::map<std::string, MetaRuleObject::PatternRewriter>> {
  static void
  inputOne(IO &IO, StringRef Key,
           std::map<std::string, MetaRuleObject::PatternRewriter> &Value) {
    IO.mapRequired(Key.str().c_str(), Value[Key.str().c_str()]);
  }

  static void
  output(IO &IO, std::map<std::string, MetaRuleObject::PatternRewriter> &V) {
    for (auto &P : V) {
      IO.mapRequired(P.first.c_str(), P.second);
    }
  }
};

template <>
struct llvm::yaml::SequenceElementTraits<MetaRuleObject::PatternRewriter> {
  static const bool flow = false;
};

template <class T>
struct llvm::yaml::SequenceTraits<std::vector<std::shared_ptr<T>>> {
  static size_t size(llvm::yaml::IO &Io, std::vector<std::shared_ptr<T>> &Seq) {
    return Seq.size();
  }
  static std::shared_ptr<T> &element(IO &, std::vector<std::shared_ptr<T>> &Seq,
                                     size_t Index) {
    if (Index >= Seq.size())
      Seq.resize(Index + 1);
    return Seq[Index];
  }
};

template <> struct llvm::yaml::ScalarEnumerationTraits<RulePriority> {
  static void enumeration(llvm::yaml::IO &Io, RulePriority &Value) {
    Io.enumCase(Value, "Takeover", RulePriority::Takeover);
    Io.enumCase(Value, "Default", RulePriority::Default);
    Io.enumCase(Value, "Fallback", RulePriority::Fallback);
  }
};

template <> struct llvm::yaml::ScalarEnumerationTraits<RuleMatchMode> {
  static void enumeration(llvm::yaml::IO &Io, RuleMatchMode &Value) {
    Io.enumCase(Value, "Partial", RuleMatchMode::Partial);
    Io.enumCase(Value, "Full", RuleMatchMode::Full);
    Io.enumCase(Value, "StrictFull", RuleMatchMode::StrictFull);
  }
};

template <> struct llvm::yaml::ScalarEnumerationTraits<RuleKind> {
  static void enumeration(llvm::yaml::IO &Io, RuleKind &Value) {
    Io.enumCase(Value, "API", RuleKind::API);
    Io.enumCase(Value, "DataType", RuleKind::DataType);
    Io.enumCase(Value, "Macro", RuleKind::Macro);
    Io.enumCase(Value, "Header", RuleKind::Header);
    Io.enumCase(Value, "Type", RuleKind::TypeRule);
    Io.enumCase(Value, "Class", RuleKind::Class);
    Io.enumCase(Value, "Enum", RuleKind::Enum);
    Io.enumCase(Value, "DisableAPIMigration", RuleKind::DisableAPIMigration);
    Io.enumCase(Value, "PatternRewriter", RuleKind::PatternRewriter);
    Io.enumCase(Value, "CMakeRule", RuleKind::CMakeRule);
    Io.enumCase(Value, "HelperFunction", RuleKind::HelperFunction);
    Io.enumCase(Value, "PythonRule", RuleKind::PythonRule);
  }
};

template <> struct llvm::yaml::MappingTraits<std::shared_ptr<MetaRuleObject>> {
  static void mapping(llvm::yaml::IO &Io,
                      std::shared_ptr<MetaRuleObject> &Doc) {
    Doc = std::make_shared<MetaRuleObject>();
    Io.mapRequired("Rule", Doc->RuleId);
    Io.mapRequired("Kind", Doc->Kind);
    Io.mapRequired("Priority", Doc->Priority);
    Io.mapOptional("CmakeSyntax", Doc->BuildScriptSyntax);
    Io.mapOptional("PythonSyntax", Doc->BuildScriptSyntax);
    Io.mapRequired("In", Doc->In);
    Io.mapRequired("Out", Doc->Out);
    Io.mapOptional("Includes", Doc->Includes);
    Io.mapOptional("Fields", Doc->Fields);
    Io.mapOptional("Methods", Doc->Methods);
    Io.mapOptional("EnumName", Doc->EnumName);
    Io.mapOptional("Prefix", Doc->Prefix);
    Io.mapOptional("Postfix", Doc->Postfix);
    Io.mapOptional("Attributes", Doc->RuleAttributes);
    Io.mapOptional("Subrules", Doc->Subrules);
    Io.mapOptional("MatchMode", Doc->MatchMode);
    Io.mapOptional("Warning", Doc->Warning);
    Io.mapOptional("APIRestrictCondition", Doc->RuleAPIRestrictCondition);
  }
};

template <>
struct llvm::yaml::MappingTraits<MetaRuleObject::APIRestrictCondition> {
  static void mapping(llvm::yaml::IO &Io,
                      MetaRuleObject::APIRestrictCondition &Doc) {
    Io.mapOptional("ArgCount", Doc.ArgCount);
  }
};

template <>
struct llvm::yaml::MappingTraits<std::shared_ptr<MetaRuleObject::ClassField>> {
  static void mapping(llvm::yaml::IO &Io,
                      std::shared_ptr<MetaRuleObject::ClassField> &Doc) {
    Doc = std::make_shared<MetaRuleObject::ClassField>();
    Io.mapRequired("In", Doc->In);
    Io.mapOptional("Out", Doc->Out);
    Io.mapOptional("OutGetter", Doc->OutGetter);
    Io.mapOptional("OutSetter", Doc->OutSetter);
  }
};

template <>
struct llvm::yaml::MappingTraits<std::shared_ptr<MetaRuleObject::ClassMethod>> {
  static void mapping(llvm::yaml::IO &Io,
                      std::shared_ptr<MetaRuleObject::ClassMethod> &Doc) {
    Doc = std::make_shared<MetaRuleObject::ClassMethod>();
    Io.mapRequired("In", Doc->In);
    Io.mapRequired("Out", Doc->Out);
  }
};

template <>
struct llvm::yaml::MappingTraits<MetaRuleObject::PatternRewriter> {
  static void mapping(llvm::yaml::IO &Io,
                      MetaRuleObject::PatternRewriter &Doc) {
    Io.mapRequired("In", Doc.In);
    Io.mapRequired("Out", Doc.Out);
    Io.mapOptional("Subrules", Doc.Subrules);
    Io.mapOptional("MatchMode", Doc.MatchMode);
    Io.mapOptional("Warning", Doc.Warning);
    Io.mapOptional("RuleId", Doc.RuleId);
  }
};

template<>
struct llvm::yaml::MappingTraits<MetaRuleObject::Attributes> {
  static void mapping(llvm::yaml::IO &Io, MetaRuleObject::Attributes &Doc) {
    Io.mapOptional("ReplaceCalleeNameOnly", Doc.ReplaceCalleeNameOnly);
    Io.mapOptional("HasExplicitTemplateArgs", Doc.HasExplicitTemplateArgs);
    Io.mapOptional("NumOfTemplateArgs", Doc.NumOfTemplateArgs);
  }
};

class RuleBase {
public:
  std::string Id;
  RulePriority Priority;
  RuleMatchMode MatchMode;
  RuleKind Kind;
  std::string In;
  std::string Out;
  clang::dpct::HelperFeatureEnum HelperFeature;
  std::vector<std::string> Includes;

  RuleBase(
      std::string Id, RulePriority Priority, RuleKind Kind, std::string In,
      std::string Out, clang::dpct::HelperFeatureEnum HelperFeature,
      const std::vector<std::string> &Includes = std::vector<std::string>())
      : Id(Id), Priority(Priority), MatchMode(RuleMatchMode::Partial),
        Kind(Kind), In(In), Out(Out), HelperFeature(HelperFeature),
        Includes(Includes) {}
};

class MacroMigrationRule : public RuleBase {
public:
  MacroMigrationRule(
      std::string Id, RulePriority Priority, std::string InStr,
      std::string OutStr,
      clang::dpct::HelperFeatureEnum Helper =
          clang::dpct::HelperFeatureEnum::none,
      const std::vector<std::string> &Includes = std::vector<std::string>())
      : RuleBase(Id, Priority, RuleKind::Macro, InStr, OutStr, Helper,
                 Includes) {}
};

// The parsing result of the "Out" attribute of a API rule
// Kind::Top labels the root node.
// For example, if the input "Out" string is:
// foo($1, $deref($2))
// The SubBuilders of the "Top" OutputBuilder will be:
// 1. OutputBuilder: Kind="String", Str="foo("
// 2. OutputBuilder: Kind = "Arg", ArgIndex=1
// 3. OutputBuilder: Kind = "Deref", ArgIndex=2
// 4. OutputBuilder: Kind = "String", Str=")"
class OutputBuilder {
public:
  enum Kind {
    String,
    Top,
    Arg,
    Queue,
    Context,
    Device,
    Deref,
    TypeName,
    AddrOf,
    DerefedTypeName,
    TemplateArg,
    MethodBase
  };
  std::string RuleName;
  clang::tooling::UnifiedPath RuleFile;
  Kind Kind;
  size_t ArgIndex;
  std::string Str;
  std::vector<std::shared_ptr<OutputBuilder>> SubBuilders;
  void parse(std::string &);
  virtual ~OutputBuilder();
protected:
  // /OutStr is the string specified in rule's "Out" session
  virtual std::shared_ptr<OutputBuilder> consumeKeyword(std::string &OutStr,
                                                size_t &Idx);
  int consumeArgIndex(std::string &OutStr, size_t &Idx, std::string &&Keyword);
  void ignoreWhitespaces(std::string &OutStr, size_t &Idx);
  void consumeRParen(std::string &OutStr, size_t &Idx, std::string &&Keyword);
  void consumeLParen(std::string &OutStr, size_t &Idx, std::string &&Keyword);
};

class TypeOutputBuilder : public OutputBuilder {
private:
  // /OutStr is the string specified in rule's "Out" session
  std::shared_ptr<OutputBuilder> consumeKeyword(std::string &OutStr,
                                                size_t &Idx) override;
};

void importRules(std::vector<clang::tooling::UnifiedPath> &RuleFiles);

#endif // DPCT_USER_DEFINED_RULES_H