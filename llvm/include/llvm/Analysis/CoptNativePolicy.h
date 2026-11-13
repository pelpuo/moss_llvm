#ifndef LLVM_ANALYSIS_COPTNATIVEPOLICY_H
#define LLVM_ANALYSIS_COPTNATIVEPOLICY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

#include <string>
#include <vector>

namespace llvm {

class CallBase;
class Function;
class Loop;

struct CoptNativeTraceRecord {
  std::string Kind;
  std::string RegionID;
  std::string Function;
  std::string Subject;
  std::string Outcome;
  double ProfileCount = 0.0;
  double MetricA = 0.0;
  double MetricB = 0.0;
};

struct CoptNativeRegionSelection {
  std::string Kind;
  std::string RegionID;
};

/// Temporarily presents a function as normal O2/speed policy to one native
/// decision point. The original size attributes are restored automatically.
class LLVM_ABI CoptScopedO2Policy {
public:
  CoptScopedO2Policy(Function &F, bool Enable);
  ~CoptScopedO2Policy();

  CoptScopedO2Policy(const CoptScopedO2Policy &) = delete;
  CoptScopedO2Policy &operator=(const CoptScopedO2Policy &) = delete;

private:
  Function &F;
  bool Enabled = false;
  bool HadOptSize = false;
  bool HadMinSize = false;
};

/// Native-policy bridge shared by the external prototype and the internal
/// Full-LTO planner.
///
/// External mode reads regional selections from COPT_NATIVE_REGION_FILE and
/// optionally appends traces to COPT_NATIVE_TRACE_FILE.
///
/// Internal mode keeps both traces and the final regional plan in memory. This
/// lets one LTO process run Oz-like/Os-like/O2 shadow pipelines, solve a plan,
/// and then apply selected O2 regional relaxations to the real module without
/// round-tripping through files or another compiler invocation.
class LLVM_ABI CoptNativePolicy {
public:
  static CoptNativePolicy &get();

  std::string inlineRegionID(const CallBase &CB) const;
  std::string unrollRegionID(const Loop &L) const;

  bool useO2ForInline(const CallBase &CB);
  bool useO2ForUnroll(const Loop &L);

  void traceInline(const CallBase &CB, StringRef RegionID, bool Profitable,
                   double ProfileCount, int Cost, int Threshold);
  void noteUnrollProfile(const Loop &L, double ProfileCount);
  void traceUnroll(const Loop &L, StringRef RegionID, unsigned Count);

  /// Begin one in-memory shadow trace. Existing in-memory trace records and
  /// per-loop profile-count state are cleared. External file tracing, when
  /// explicitly configured, remains independent.
  void beginInternalTrace();

  /// Finish and return the current in-memory trace.
  std::vector<CoptNativeTraceRecord> endInternalTrace();

  /// Install the in-memory final regional plan used by the real Full-LTO run.
  /// While this plan is active it takes precedence over any external region
  /// policy file.
  void setInternalRegionPlan(ArrayRef<CoptNativeRegionSelection> Regions);
  void clearInternalRegionPlan();

private:
  CoptNativePolicy();
  CoptNativePolicy(const CoptNativePolicy &) = delete;
  CoptNativePolicy &operator=(const CoptNativePolicy &) = delete;

  void loadRegionPlan();
  bool selected(StringRef Kind, StringRef RegionID) const;
  void appendTrace(StringRef Kind, StringRef RegionID, StringRef Function,
                   StringRef Subject, StringRef Outcome, double ProfileCount,
                   double MetricA, double MetricB);

  bool Loaded = false;
  bool LogEnabled = false;
  std::string RegionFile;
  std::string TraceFile;

  class Impl;
  Impl *PImpl = nullptr;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_COPTNATIVEPOLICY_H
