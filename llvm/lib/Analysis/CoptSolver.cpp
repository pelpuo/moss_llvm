#include "llvm/Analysis/CoptSolver.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <system_error>
#include <unordered_map>
#include <vector>

using namespace llvm;

namespace {

struct DPNode {
  int64_t Size = 0;
  double Benefit = 0.0;

  size_t ValueIndex =
      std::numeric_limits<size_t>::max();

  std::shared_ptr<DPNode> Prev;
};

static int64_t conservativeCostUnits(
    double Value) {

  // TTI code-size sums are normally integral, but keep the solver safe if a
  // future estimator produces fractional values. ceil() prevents us from
  // claiming more size credit than the estimate provides, including for
  // negative deltas.
  return static_cast<int64_t>(
      std::ceil(Value - 1.0e-9));
}

static int64_t conservativeBudgetUnits(
    double Value) {

  return static_cast<int64_t>(
      std::floor(Value + 1.0e-9));
}

} // namespace

CoptSolverResult
llvm::solveCoptMultipleChoice(
    ArrayRef<CoptPlanValue> Values,
    double MaxSizeDelta,
    bool ExactOnly) {

  if (MaxSizeDelta < 0.0)
    report_fatal_error(
        "COPT: predicted size budget is below the global lower baseline");

  std::unordered_map<
      std::string,
      std::vector<size_t>>
      ByFunction;

  for (size_t I = 0;
       I < Values.size();
       ++I) {

      const CoptPlanValue &V =
          Values[I];

      if (!V.Replayable)
        continue;

      if (ExactOnly &&
        !V.LowerEndpoint &&
        !V.Exact)
      continue;

    ByFunction[V.Function].push_back(I);
  }

  std::vector<std::string> Functions;
  Functions.reserve(ByFunction.size());

  for (const auto &KV : ByFunction)
    Functions.push_back(KV.first);

  std::sort(
      Functions.begin(),
      Functions.end());

  // Every function must retain at least one lower endpoint.
  for (const std::string &Function :
       Functions) {

    bool HasLower = false;

    for (size_t Index :
         ByFunction[Function])
      HasLower |=
          Values[Index].LowerEndpoint;

    if (!HasLower)
      report_fatal_error(
          Twine("COPT: function '") +
          Function +
          "' has no selectable lower endpoint");
  }

  const int64_t Budget =
      conservativeBudgetUnits(
          MaxSizeDelta);

  // Suffix minimum costs make pruning correct even when a later function has
  // a negative-size option.
  std::vector<int64_t>
      MinRemaining(
          Functions.size() + 1,
          0);

  for (size_t Rev = Functions.size();
       Rev > 0;
       --Rev) {

    size_t I = Rev - 1;

    int64_t MinCost =
        std::numeric_limits<int64_t>::max();

    for (size_t ValueIndex :
         ByFunction[Functions[I]]) {

      MinCost =
          std::min(
              MinCost,
              conservativeCostUnits(
                  Values[ValueIndex]
                      .SizeDelta));
    }

    MinRemaining[I] =
        MinRemaining[I + 1] +
        MinCost;
  }

  std::vector<std::shared_ptr<DPNode>>
      Frontier;

  Frontier.push_back(
      std::make_shared<DPNode>());

  for (size_t FunctionIndex = 0;
       FunctionIndex < Functions.size();
       ++FunctionIndex) {

    const std::string &Function =
        Functions[FunctionIndex];

    std::vector<std::shared_ptr<DPNode>>
        Next;

    for (const std::shared_ptr<DPNode> &Base :
         Frontier) {

      for (size_t ValueIndex :
           ByFunction[Function]) {

        const CoptPlanValue &V =
            Values[ValueIndex];

        int64_t NewSize =
            Base->Size +
            conservativeCostUnits(
                V.SizeDelta);

        // Even after taking the smallest possible choices in all remaining
        // functions, this partial solution could not meet the budget.
        if (NewSize +
                MinRemaining[
                    FunctionIndex + 1] >
            Budget)
          continue;

        auto N =
            std::make_shared<DPNode>();

        N->Size = NewSize;
        N->Benefit =
            Base->Benefit +
            V.PerformanceBenefit;

        N->ValueIndex =
            ValueIndex;

        N->Prev = Base;

        Next.push_back(
            std::move(N));
      }
    }

    if (Next.empty()) {
      CoptSolverResult R;
      return R;
    }

    // Exact Pareto pruning:
    //
    // A DP state is dominated when another state uses <= size and has >=
    // benefit. Sorting by size lets us remove such states in one sweep.
    std::sort(
        Next.begin(),
        Next.end(),
        [](const std::shared_ptr<DPNode> &A,
           const std::shared_ptr<DPNode> &B) {

          if (A->Size != B->Size)
            return A->Size < B->Size;

          return A->Benefit >
                 B->Benefit;
        });

    std::vector<std::shared_ptr<DPNode>>
        Pruned;

    double BestBenefit =
        -std::numeric_limits<double>::infinity();

    for (const auto &N : Next) {
      if (N->Benefit <=
          BestBenefit + 1.0e-12)
        continue;

      BestBenefit = N->Benefit;
      Pruned.push_back(N);
    }

    Frontier =
        std::move(Pruned);
  }

  std::shared_ptr<DPNode> Best;

  for (const auto &N : Frontier) {
    if (N->Size > Budget)
      continue;

    if (!Best ||
        N->Benefit >
            Best->Benefit + 1.0e-12 ||
        (std::abs(
             N->Benefit -
             Best->Benefit) <= 1.0e-12 &&
         N->Size < Best->Size))
      Best = N;
  }

  if (!Best) {
    CoptSolverResult R;
    return R;
  }

  CoptSolverResult Result;
  Result.Feasible = true;

  Result.TotalSizeDelta =
      static_cast<double>(
          Best->Size);

  Result.TotalPerformanceBenefit =
      Best->Benefit;

  std::vector<CoptSolverSelection>
      ReverseSelections;

  for (std::shared_ptr<DPNode> N = Best;
       N && N->Prev;
       N = N->Prev) {

    const CoptPlanValue &V =
        Values[N->ValueIndex];

    CoptSolverSelection S;

    S.Function = V.Function;
    S.PlanID = V.PlanID;
    S.SizeDelta = V.SizeDelta;
    S.PerformanceBenefit =
        V.PerformanceBenefit;
    S.Exact = V.Exact;

    ReverseSelections.push_back(
        std::move(S));
  }

  std::reverse(
      ReverseSelections.begin(),
      ReverseSelections.end());

  Result.Selections =
      std::move(ReverseSelections);

  return Result;
}

void llvm::writeCoptSolverReport(
    StringRef Path,
    double LowerSizeCost,
    double MaxSizeCost,
    unsigned ExactEvaluations,
    ArrayRef<CoptPlanValue> Values,
    const CoptSolverResult &Result) {

  if (Path.empty())
    return;

  json::Array ValueArray;

  for (const CoptPlanValue &V : Values) {
    json::Object O;

    O["function"] = V.Function;
    O["plan_id"] = V.PlanID;

    O["size_delta"] =
        V.SizeDelta;

    O["performance_benefit"] =
        V.PerformanceBenefit;

    O["lower_endpoint"] =
        V.LowerEndpoint;

    O["upper_endpoint"] =
        V.UpperEndpoint;

    O["exact"] =
        V.Exact;
    
    O["replayable"] =
        V.Replayable;

    ValueArray.push_back(
        std::move(O));
  }

  json::Array SelectionArray;

  for (const CoptSolverSelection &S :
       Result.Selections) {

    json::Object O;

    O["function"] =
        S.Function;

    O["plan_id"] =
        S.PlanID;

    O["size_delta"] =
        S.SizeDelta;

    O["performance_benefit"] =
        S.PerformanceBenefit;

    O["exact"] =
        S.Exact;

    SelectionArray.push_back(
        std::move(O));
  }

  json::Object Root;

  Root["version"] = 1;

  Root["lower_size_cost"] =
      LowerSizeCost;

  Root["max_size_cost"] =
      MaxSizeCost;

  Root["size_budget_delta"] =
      MaxSizeCost -
      LowerSizeCost;

  Root["exact_evaluations"] =
      static_cast<int64_t>(
          ExactEvaluations);

  Root["feasible"] =
      Result.Feasible;

  Root["selected_size_delta"] =
      Result.TotalSizeDelta;

  Root["selected_performance_benefit"] =
      Result.TotalPerformanceBenefit;

  Root["values"] =
      std::move(ValueArray);

  Root["selections"] =
      std::move(SelectionArray);

  std::error_code EC;

  raw_fd_ostream OS(
      Path,
      EC,
      sys::fs::OF_Text);

  if (EC)
    report_fatal_error(
        Twine("COPT: cannot write solver report: ") +
        EC.message());

  OS << formatv(
      "{0:2}\n",
      json::Value(std::move(Root)));
}