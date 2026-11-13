#ifndef LLVM_ANALYSIS_COPTFUNCTIONPLANS_H
#define LLVM_ANALYSIS_COPTFUNCTIONPLANS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ConstrainedOptimization.h"
#include "llvm/Support/Compiler.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {

struct CoptPlanChoice {
  std::string DecisionID;
  std::string State;
};

struct CoptFunctionPlan {
  std::string ID;
  std::string Function;

  std::vector<CoptPlanChoice> Choices;

  bool IsLowerEndpoint = false;
  bool IsUpperEndpoint = false;
};

/// Generate a bounded collection of dependency-valid function plans.
///
/// The generator starts from the conservative lower configuration, creates
/// legal promotion bundles by taking the transitive closure of provenance
/// dependencies, and then lazily expands a bounded beam of configurations.
///
/// A non-lower state is legal only when every decision listed in Dependencies
/// is at that dependency's upper endpoint.
///
/// The lower and valid upper endpoint plans are retained explicitly. The
/// generator never enumerates the full Cartesian product of regional states.
LLVM_ABI std::vector<CoptFunctionPlan>
buildCoptFunctionPlans(
    ArrayRef<CoptCandidate> Candidates,
    unsigned MaxPlansPerFunction);

/// Build the global conservative COPT plan: every discovered decision uses its
/// lower-endpoint state.
LLVM_ABI std::unordered_map<std::string, std::string>
buildCoptGlobalLowerPlan(
    ArrayRef<CoptCandidate> Candidates);

/// Overlay a function plan onto a global plan.
LLVM_ABI std::unordered_map<std::string, std::string>
overlayCoptFunctionPlan(
    const std::unordered_map<std::string, std::string> &Base,
    const CoptFunctionPlan &Plan);

LLVM_ABI const CoptFunctionPlan *
findCoptFunctionPlan(
    ArrayRef<CoptFunctionPlan> Plans,
    StringRef Function,
    StringRef PlanID);

LLVM_ABI void
writeCoptFunctionPlans(
    StringRef Path,
    ArrayRef<CoptFunctionPlan> Plans);

} // namespace llvm

#endif // LLVM_ANALYSIS_COPTFUNCTIONPLANS_H
