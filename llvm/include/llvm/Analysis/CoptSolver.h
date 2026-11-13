#ifndef LLVM_ANALYSIS_COPTSOLVER_H
#define LLVM_ANALYSIS_COPTSOLVER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

#include <string>
#include <vector>

namespace llvm {

struct CoptPlanValue {
  std::string Function;
  std::string PlanID;

  // Relative to the global conservative COPT baseline.
  double SizeDelta = 0.0;
  double PerformanceBenefit = 0.0;

  // Lower plans are exact by definition: delta=0, benefit=0.
  bool LowerEndpoint = false;
  bool UpperEndpoint = false;

  // True only after the plan has been materialized against the global lower
  // baseline, except for the lower plan itself.
  bool Exact = false;
  // False when exact materialization showed that this discovery-time plan can
  // no longer be replayed under its candidate configuration.
  bool Replayable = true;
};

struct CoptSolverSelection {
  std::string Function;
  std::string PlanID;

  double SizeDelta = 0.0;
  double PerformanceBenefit = 0.0;

  bool Exact = false;
};

struct CoptSolverResult {
  bool Feasible = false;

  double TotalSizeDelta = 0.0;
  double TotalPerformanceBenefit = 0.0;

  std::vector<CoptSolverSelection> Selections;
};

/// Solve a multiple-choice knapsack:
///
///   choose exactly one plan per function
///   maximize total PerformanceBenefit
///   subject to total SizeDelta <= MaxSizeDelta
///
/// Negative SizeDelta values are supported.
///
/// When ExactOnly is true, all non-lower non-exact plans are ignored.
LLVM_ABI CoptSolverResult
solveCoptMultipleChoice(
    ArrayRef<CoptPlanValue> Values,
    double MaxSizeDelta,
    bool ExactOnly);

LLVM_ABI void
writeCoptSolverReport(
    StringRef Path,
    double LowerSizeCost,
    double MaxSizeCost,
    unsigned ExactEvaluations,
    ArrayRef<CoptPlanValue> Values,
    const CoptSolverResult &Result);

} // namespace llvm

#endif // LLVM_ANALYSIS_COPTSOLVER_H