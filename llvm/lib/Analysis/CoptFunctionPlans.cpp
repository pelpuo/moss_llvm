#include "llvm/Analysis/CoptFunctionPlans.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

using CoptStateVector =
    std::vector<const CoptAlternative *>;

using CoptDecisionIndex =
    std::unordered_map<std::string, size_t>;

struct CoptSearchState {
  CoptStateVector States;
  std::string Key;
  double Aggressiveness = 0.0;
};

static std::string md5(StringRef Input) {
  MD5 Hasher;
  MD5::MD5Result Hash;
  SmallString<32> Hex;

  Hasher.update(Input);
  Hasher.final(Hash);
  MD5::stringifyResult(Hash, Hex);

  return std::string(Hex);
}

static const CoptAlternative *
findLowerState(const CoptCandidate &C) {
  for (const CoptAlternative &A : C.Alternatives)
    if (A.Replayable && A.LowerEndpoint)
      return &A;

  return nullptr;
}

static const CoptAlternative *
findUpperState(const CoptCandidate &C) {
  for (const CoptAlternative &A : C.Alternatives)
    if (A.Replayable && A.UpperEndpoint)
      return &A;

  return nullptr;
}

static double stateOrderKey(
    const CoptAlternative &A) {

  if (A.LowerEndpoint)
    return -1.0e30;

  if (auto It = A.Metrics.find("unroll_count");
      It != A.Metrics.end())
    return It->second;

  if (auto It = A.Metrics.find("vf");
      It != A.Metrics.end())
    return It->second;

  if (A.SizeCost)
    return *A.SizeCost;

  if (A.PerfBenefit)
    return *A.PerfBenefit;

  if (A.UpperEndpoint)
    return 1.0e30;

  return 0.0;
}

static std::vector<const CoptAlternative *>
replayableStates(const CoptCandidate &C) {

  std::vector<const CoptAlternative *> States;

  for (const CoptAlternative &A : C.Alternatives)
    if (A.Replayable)
      States.push_back(&A);

  std::stable_sort(
      States.begin(),
      States.end(),
      [](const CoptAlternative *A,
         const CoptAlternative *B) {

        // The lower endpoint is always the beginning of the search path.
        if (A->LowerEndpoint != B->LowerEndpoint)
          return A->LowerEndpoint;

        // Among non-lower states, keep the upper/reference endpoint at the
        // end of the path. Intermediate UF/VF states remain between the
        // endpoints.
        if (A->UpperEndpoint != B->UpperEndpoint)
          return !A->UpperEndpoint;

        const double AK = stateOrderKey(*A);
        const double BK = stateOrderKey(*B);

        if (AK != BK)
          return AK < BK;

        return A->Name < B->Name;
      });

  return States;
}

static std::string planKey(
    StringRef Function,
    ArrayRef<const CoptCandidate *> Decisions,
    ArrayRef<const CoptAlternative *> States) {

  std::string Key;
  raw_string_ostream OS(Key);

  OS << Function << "|";

  for (size_t I = 0;
       I < Decisions.size();
       ++I) {

    OS << Decisions[I]->ID
       << "="
       << States[I]->Name
       << ";";
  }

  OS.flush();
  return Key;
}

static CoptFunctionPlan makePlan(
    StringRef Function,
    ArrayRef<const CoptCandidate *> Decisions,
    ArrayRef<const CoptAlternative *> States) {

  CoptFunctionPlan P;
  P.Function = Function.str();

  bool AllLower = true;
  bool AllUpper = true;

  for (size_t I = 0;
       I < Decisions.size();
       ++I) {

    CoptPlanChoice Choice;
    Choice.DecisionID = Decisions[I]->ID;
    Choice.State = States[I]->Name;

    P.Choices.push_back(
        std::move(Choice));

    AllLower &=
        States[I]->LowerEndpoint;

    AllUpper &=
        States[I]->UpperEndpoint;
  }

  P.IsLowerEndpoint = AllLower;
  P.IsUpperEndpoint = AllUpper;

  P.ID =
      md5(planKey(
          Function,
          Decisions,
          States));

  return P;
}

static CoptDecisionIndex
buildDecisionIndex(
    ArrayRef<const CoptCandidate *> Decisions) {

  CoptDecisionIndex Result;
  Result.reserve(Decisions.size());

  for (size_t I = 0;
       I < Decisions.size();
       ++I)
    Result.emplace(
        Decisions[I]->ID,
        I);

  return Result;
}

static bool isDependencyValid(
    ArrayRef<const CoptCandidate *> Decisions,
    ArrayRef<const CoptAlternative *> States,
    const CoptDecisionIndex &DecisionIndex) {

  if (Decisions.size() != States.size())
    return false;

  for (size_t I = 0;
       I < Decisions.size();
       ++I) {

    const CoptAlternative *State =
        States[I];

    if (!State)
      return false;

    // Lower states do not require the region to survive.
    if (State->LowerEndpoint)
      continue;

    for (const std::string &Dependency :
         Decisions[I]->Dependencies) {

      auto It =
          DecisionIndex.find(Dependency);

      // A dependency outside this function cannot currently be represented
      // safely by a function-local plan.
      if (It == DecisionIndex.end())
        return false;

      const CoptAlternative *DependencyState =
          States[It->second];

      if (!DependencyState ||
          !DependencyState->UpperEndpoint)
        return false;
    }
  }

  return true;
}

/// Starting from States, make every dependency required by every selected
/// non-lower state legal.
///
/// A non-lower state requires each dependency to be at that dependency's upper
/// endpoint. Setting a dependency to upper may itself introduce dependencies,
/// so this computes the transitive closure.
///
/// Returns false if the dependency is outside the function or does not have a
/// replayable upper endpoint.
static bool applyDependencyClosure(
    ArrayRef<const CoptCandidate *> Decisions,
    const CoptDecisionIndex &DecisionIndex,
    CoptStateVector &States) {

  if (Decisions.size() != States.size())
    return false;

  std::vector<uint8_t>
      Visiting(Decisions.size(), 0);

  std::vector<uint8_t>
      Closed(Decisions.size(), 0);

  std::function<bool(size_t)>
      CloseDecision =
          [&](size_t I) -> bool {

    if (I >= Decisions.size() ||
        !States[I])
      return false;

    // A lower state does not require provenance dependencies.
    if (States[I]->LowerEndpoint) {
      Closed[I] = 1;
      return true;
    }

    if (Closed[I])
      return true;

    // A dependency cycle is legal only if the cycle can be driven to upper.
    // The caller sets a dependency to upper before recursing, so encountering
    // an already-visiting node here is safe. The final validity check below
    // verifies that all edges in the cycle are satisfied.
    if (Visiting[I])
      return true;

    Visiting[I] = 1;

    for (const std::string &Dependency :
         Decisions[I]->Dependencies) {

      auto It =
          DecisionIndex.find(Dependency);

      if (It == DecisionIndex.end())
        return false;

      const size_t DependencyIndex =
          It->second;

      const CoptAlternative *Upper =
          findUpperState(
              *Decisions[DependencyIndex]);

      if (!Upper)
        return false;

      // Dependency closure is monotonic: dependencies are only promoted to
      // their upper endpoint; no state is ever demoted.
      States[DependencyIndex] =
          Upper;

      if (!CloseDecision(
              DependencyIndex))
        return false;
    }

    Visiting[I] = 0;
    Closed[I] = 1;
    return true;
  };

  // Close every currently non-lower state. This also handles states that were
  // already present before the most recent promotion.
  for (size_t I = 0;
       I < Decisions.size();
       ++I) {

    if (!States[I])
      return false;

    if (States[I]->LowerEndpoint)
      continue;

    if (!CloseDecision(I))
      return false;
  }

  return isDependencyValid(
      Decisions,
      States,
      DecisionIndex);
}

static void appendPlanIfNew(
    std::vector<CoptFunctionPlan> &Out,
    std::unordered_set<std::string> &Seen,
    CoptFunctionPlan P) {

  if (!Seen.insert(P.ID).second)
    return;

  Out.push_back(std::move(P));
}

static size_t statePosition(
    ArrayRef<const CoptAlternative *> Options,
    const CoptAlternative *State) {

  for (size_t I = 0;
       I < Options.size();
       ++I)
    if (Options[I] == State)
      return I;

  return Options.size();
}

static double configurationAggressiveness(
    ArrayRef<std::vector<const CoptAlternative *>>
        Options,
    ArrayRef<const CoptAlternative *> States) {

  if (Options.empty() ||
      Options.size() != States.size())
    return 0.0;

  double Sum = 0.0;
  size_t Count = 0;

  for (size_t I = 0;
       I < Options.size();
       ++I) {

    const CoptAlternative *State =
        States[I];

    if (!State)
      continue;

    double Local = 0.0;

    if (State->LowerEndpoint) {
      Local = 0.0;
    } else if (State->UpperEndpoint) {
      Local = 1.0;
    } else {
      const size_t Pos =
          statePosition(
              Options[I],
              State);

      if (Pos < Options[I].size() &&
          Options[I].size() > 1)
        Local =
            static_cast<double>(Pos) /
            static_cast<double>(
                Options[I].size() - 1);
    }

    Sum +=
        std::clamp(
            Local,
            0.0,
            1.0);

    ++Count;
  }

  if (Count == 0)
    return 0.0;

  return Sum /
         static_cast<double>(Count);
}

static CoptSearchState makeSearchState(
    StringRef Function,
    ArrayRef<const CoptCandidate *> Decisions,
    ArrayRef<std::vector<const CoptAlternative *>>
        Options,
    CoptStateVector States) {

  CoptSearchState S;

  S.Key =
      planKey(
          Function,
          Decisions,
          States);

  S.Aggressiveness =
      configurationAggressiveness(
          Options,
          States);

  S.States =
      std::move(States);

  return S;
}

/// Keep a bounded, deterministic, aggressiveness-diverse subset of candidates.
///
/// We deliberately do not rank here by CoptAlternative::PerfBenefit or
/// SizeCost because those values are transformation-specific and are not
/// globally comparable. Milestone 5 performs the actual approximate and exact
/// value modeling. This stage only needs to provide a compact, diverse set of
/// legal configurations.
static std::vector<CoptSearchState>
selectDiverseStates(
    std::vector<CoptSearchState> Candidates,
    unsigned Limit) {

  if (Limit == 0 ||
      Candidates.empty())
    return {};

  std::sort(
      Candidates.begin(),
      Candidates.end(),
      [](const CoptSearchState &A,
         const CoptSearchState &B) {

        if (A.Aggressiveness !=
            B.Aggressiveness)
          return A.Aggressiveness <
                 B.Aggressiveness;

        return A.Key < B.Key;
      });

  if (Candidates.size() <= Limit)
    return Candidates;

  std::vector<CoptSearchState> Selected;
  Selected.reserve(Limit);

  if (Limit == 1) {
    Selected.push_back(
        std::move(
            Candidates[
                Candidates.size() / 2]));
    return Selected;
  }

  // Evenly sample the sorted aggressiveness range. This prevents a large
  // function from using the entire beam on nearly-identical low or high plans.
  for (unsigned I = 0;
       I < Limit;
       ++I) {

    const size_t Index =
        static_cast<size_t>(
            (static_cast<uint64_t>(I) *
             static_cast<uint64_t>(
                 Candidates.size() - 1)) /
            static_cast<uint64_t>(
                Limit - 1));

    Selected.push_back(
        std::move(
            Candidates[Index]));
  }

  return Selected;
}

static void addCandidateIfNew(
    StringRef Function,
    ArrayRef<const CoptCandidate *> Decisions,
    ArrayRef<std::vector<const CoptAlternative *>>
        Options,
    const CoptDecisionIndex &DecisionIndex,
    CoptStateVector States,
    const std::unordered_set<std::string>
        &AlreadySelected,
    std::unordered_set<std::string>
        &RoundSeen,
    std::vector<CoptSearchState>
        &Candidates) {

  if (!applyDependencyClosure(
          Decisions,
          DecisionIndex,
          States))
    return;

  std::string Key =
      planKey(
          Function,
          Decisions,
          States);

  if (AlreadySelected.count(Key) ||
      !RoundSeen.insert(Key).second)
    return;

  CoptSearchState S;

  S.Key =
      std::move(Key);

  S.Aggressiveness =
      configurationAggressiveness(
          Options,
          States);

  S.States =
      std::move(States);

  Candidates.push_back(
      std::move(S));
}

} // namespace

std::vector<CoptFunctionPlan>
llvm::buildCoptFunctionPlans(
    ArrayRef<CoptCandidate> Candidates,
    unsigned MaxPlansPerFunction) {

  if (MaxPlansPerFunction < 2)
    MaxPlansPerFunction = 2;

  std::unordered_map<
      std::string,
      std::vector<const CoptCandidate *>>
      ByFunction;

  for (const CoptCandidate &C :
       Candidates)
    ByFunction[C.Function].push_back(&C);

  std::vector<CoptFunctionPlan> Result;

  for (auto &[FunctionName, Decisions] :
       ByFunction) {

    std::sort(
        Decisions.begin(),
        Decisions.end(),
        [](const CoptCandidate *A,
           const CoptCandidate *B) {

          if (A->Sequence != B->Sequence)
            return A->Sequence <
                   B->Sequence;

          return A->ID < B->ID;
        });

    bool MissingEndpoint = false;

    std::vector<
        std::vector<const CoptAlternative *>>
        Options;

    Options.reserve(
        Decisions.size());

    for (const CoptCandidate *C :
         Decisions) {

      auto States =
          replayableStates(*C);

      if (!findLowerState(*C) ||
          !findUpperState(*C) ||
          States.empty()) {
        MissingEndpoint = true;
        break;
      }

      Options.push_back(
          std::move(States));
    }

    if (MissingEndpoint)
      continue;

    const CoptDecisionIndex
        DecisionIndex =
            buildDecisionIndex(
                Decisions);

    CoptStateVector LowerStates;
    LowerStates.reserve(
        Decisions.size());

    CoptStateVector UpperStates;
    UpperStates.reserve(
        Decisions.size());

    for (const CoptCandidate *C :
         Decisions) {

      LowerStates.push_back(
          findLowerState(*C));

      UpperStates.push_back(
          findUpperState(*C));
    }

    std::vector<CoptFunctionPlan>
        FunctionPlans;

    FunctionPlans.reserve(
        MaxPlansPerFunction);

    std::unordered_set<std::string>
        SeenPlanIDs;

    // Keys of configurations that have actually been admitted to the bounded
    // search. Candidates pruned by the beam are intentionally not inserted
    // here; they may be rediscovered from a different admitted parent later.
    std::unordered_set<std::string>
        SelectedKeys;

    // ---------------------------------------------------------------------
    // Anchor 1: conservative lower endpoint.
    // ---------------------------------------------------------------------
    if (isDependencyValid(
            Decisions,
            LowerStates,
            DecisionIndex)) {

      appendPlanIfNew(
          FunctionPlans,
          SeenPlanIDs,
          makePlan(
              FunctionName,
              Decisions,
              LowerStates));

      SelectedKeys.insert(
          planKey(
              FunctionName,
              Decisions,
              LowerStates));
    }

    // ---------------------------------------------------------------------
    // Anchor 2: reference upper endpoint.
    //
    // Keep it explicitly even though the bounded search may never naturally
    // walk all the way from lower to upper.
    // ---------------------------------------------------------------------
    if (FunctionPlans.size() <
            MaxPlansPerFunction &&
        isDependencyValid(
            Decisions,
            UpperStates,
            DecisionIndex)) {

      appendPlanIfNew(
          FunctionPlans,
          SeenPlanIDs,
          makePlan(
              FunctionName,
              Decisions,
              UpperStates));

      SelectedKeys.insert(
          planKey(
              FunctionName,
              Decisions,
              UpperStates));
    }

    if (FunctionPlans.size() <
        MaxPlansPerFunction) {

    // The beam width controls breadth while MaxPlansPerFunction controls the
    // total amount of work exposed to the Milestone-5 solver. With the default
    // 128-plan limit this gives a beam of 16 and enough rounds to construct
    // meaningful multi-decision configurations.
    const unsigned BeamWidth =
        std::min(
            MaxPlansPerFunction,
            std::max(
                4u,
                std::min(
                    16u,
                    MaxPlansPerFunction / 4u)));

    // ---------------------------------------------------------------------
    // Seed frontier:
    //
    // Promote exactly one root decision away from lower, then immediately
    // compute its transitive dependency closure. These are legal "promotion
    // bundles", not raw Cartesian-product assignments.
    // ---------------------------------------------------------------------
    std::vector<CoptSearchState>
        SeedCandidates;

    std::unordered_set<std::string>
        SeedSeen;

    for (size_t Decision = 0;
         Decision < Decisions.size();
         ++Decision) {

      for (const CoptAlternative *State :
           Options[Decision]) {

        if (State->LowerEndpoint)
          continue;

        CoptStateVector Child =
            LowerStates;

        Child[Decision] =
            State;

        addCandidateIfNew(
            FunctionName,
            Decisions,
            Options,
            DecisionIndex,
            std::move(Child),
            SelectedKeys,
            SeedSeen,
            SeedCandidates);
      }
    }

    {
      const unsigned Remaining =
          MaxPlansPerFunction -
          static_cast<unsigned>(
              FunctionPlans.size());

      std::vector<CoptSearchState>
          Frontier =
              selectDiverseStates(
                  std::move(
                      SeedCandidates),
                  std::min(
                      BeamWidth,
                      Remaining));

      for (const CoptSearchState &S :
           Frontier) {

        appendPlanIfNew(
            FunctionPlans,
            SeenPlanIDs,
            makePlan(
                FunctionName,
                Decisions,
                S.States));

        SelectedKeys.insert(
            S.Key);

        if (FunctionPlans.size() >=
            MaxPlansPerFunction)
          break;
      }

      // -------------------------------------------------------------------
      // Lazy bounded expansion.
      //
      // Each admitted plan is expanded by promoting one decision by one state
      // along its ordered local path. Dependency closure may promote additional
      // prerequisite decisions to upper. Only a bounded, diverse subset of
      // children becomes the next frontier.
      //
      // This changes the worst practical behavior from exhaustive
      //
      //   product_i |states_i|
      //
      // enumeration to work proportional to the plan cap and number of
      // decisions.
      // -------------------------------------------------------------------
      while (!Frontier.empty() &&
             FunctionPlans.size() <
                 MaxPlansPerFunction) {

        std::vector<CoptSearchState>
            NextCandidates;

        std::unordered_set<std::string>
            RoundSeen;

        for (const CoptSearchState &Parent :
             Frontier) {

          for (size_t Decision = 0;
               Decision < Decisions.size();
               ++Decision) {

            const size_t Pos =
                statePosition(
                    Options[Decision],
                    Parent.States[Decision]);

            if (Pos >=
                    Options[Decision].size() ||
                Pos + 1 >=
                    Options[Decision].size())
              continue;

            CoptStateVector Child =
                Parent.States;

            Child[Decision] =
                Options[Decision][Pos + 1];

            addCandidateIfNew(
                FunctionName,
                Decisions,
                Options,
                DecisionIndex,
                std::move(Child),
                SelectedKeys,
                RoundSeen,
                NextCandidates);
          }
        }

        if (NextCandidates.empty())
          break;

        const unsigned RemainingNow =
            MaxPlansPerFunction -
            static_cast<unsigned>(
                FunctionPlans.size());

        std::vector<CoptSearchState>
            NextFrontier =
                selectDiverseStates(
                    std::move(
                        NextCandidates),
                    std::min(
                        BeamWidth,
                        RemainingNow));

        if (NextFrontier.empty())
          break;

        for (const CoptSearchState &S :
             NextFrontier) {

          appendPlanIfNew(
              FunctionPlans,
              SeenPlanIDs,
              makePlan(
                  FunctionName,
                  Decisions,
                  S.States));

          SelectedKeys.insert(
              S.Key);

          if (FunctionPlans.size() >=
              MaxPlansPerFunction)
            break;
        }

        Frontier =
            std::move(
                NextFrontier);
      }
    }

    }

    std::sort(
        FunctionPlans.begin(),
        FunctionPlans.end(),
        [](const CoptFunctionPlan &A,
           const CoptFunctionPlan &B) {

          if (A.IsLowerEndpoint !=
              B.IsLowerEndpoint)
            return A.IsLowerEndpoint;

          if (A.IsUpperEndpoint !=
              B.IsUpperEndpoint)
            return !A.IsUpperEndpoint;

          return A.ID < B.ID;
        });

    Result.insert(
        Result.end(),
        std::make_move_iterator(
            FunctionPlans.begin()),
        std::make_move_iterator(
            FunctionPlans.end()));
  }

  std::sort(
      Result.begin(),
      Result.end(),
      [](const CoptFunctionPlan &A,
         const CoptFunctionPlan &B) {

        if (A.Function != B.Function)
          return A.Function < B.Function;

        if (A.IsLowerEndpoint !=
            B.IsLowerEndpoint)
          return A.IsLowerEndpoint;

        if (A.IsUpperEndpoint !=
            B.IsUpperEndpoint)
          return !A.IsUpperEndpoint;

        return A.ID < B.ID;
      });

  return Result;
}

std::unordered_map<std::string, std::string>
llvm::buildCoptGlobalLowerPlan(
    ArrayRef<CoptCandidate> Candidates) {

  std::unordered_map<
      std::string,
      std::string>
      Result;

  for (const CoptCandidate &C :
       Candidates) {

    const CoptAlternative *Lower =
        findLowerState(C);

    if (!Lower)
      report_fatal_error(
          Twine("COPT: candidate ") +
          C.ID +
          " has no replayable lower endpoint");

    Result[C.ID] =
        Lower->Name;
  }

  return Result;
}

std::unordered_map<std::string, std::string>
llvm::overlayCoptFunctionPlan(
    const std::unordered_map<
        std::string,
        std::string> &Base,
    const CoptFunctionPlan &Plan) {

  auto Result = Base;

  for (const CoptPlanChoice &Choice :
       Plan.Choices)
    Result[Choice.DecisionID] =
        Choice.State;

  return Result;
}

const CoptFunctionPlan *
llvm::findCoptFunctionPlan(
    ArrayRef<CoptFunctionPlan> Plans,
    StringRef Function,
    StringRef PlanID) {

  for (const CoptFunctionPlan &P :
       Plans)
    if (P.Function == Function &&
        P.ID == PlanID)
      return &P;

  return nullptr;
}

void llvm::writeCoptFunctionPlans(
    StringRef Path,
    ArrayRef<CoptFunctionPlan> Plans) {

  if (Path.empty())
    return;

  json::Array FunctionArray;

  std::string CurrentFunction;
  json::Array CurrentPlans;

  auto Flush =
      [&]() {

    if (CurrentFunction.empty())
      return;

    json::Object F;
    F["function"] = CurrentFunction;
    F["plans"] =
        std::move(CurrentPlans);

    FunctionArray.push_back(
        std::move(F));

    CurrentPlans =
        json::Array();
  };

  for (const CoptFunctionPlan &P :
       Plans) {

    if (CurrentFunction !=
        P.Function) {
      Flush();
      CurrentFunction =
          P.Function;
    }

    json::Object O;

    O["id"] =
        P.ID;

    O["lower_endpoint"] =
        P.IsLowerEndpoint;

    O["upper_endpoint"] =
        P.IsUpperEndpoint;

    json::Array Choices;

    for (const CoptPlanChoice &Choice :
         P.Choices) {

      json::Object C;

      C["decision_id"] =
          Choice.DecisionID;

      C["state"] =
          Choice.State;

      Choices.push_back(
          std::move(C));
    }

    O["choices"] =
        std::move(Choices);

    CurrentPlans.push_back(
        std::move(O));
  }

  Flush();

  json::Object Root;

  Root["version"] = 1;

  Root["functions"] =
      std::move(FunctionArray);

  std::error_code EC;

  raw_fd_ostream OS(
      Path,
      EC,
      sys::fs::OF_Text);

  if (EC)
    report_fatal_error(
        Twine(
            "COPT: cannot write function-plan file: ") +
        EC.message());

  OS << formatv(
      "{0:2}\n",
      json::Value(
          std::move(Root)));
}
