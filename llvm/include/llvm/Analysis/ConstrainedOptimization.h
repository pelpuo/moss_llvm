#ifndef LLVM_ANALYSIS_CONSTRAINEDOPTIMIZATION_H
#define LLVM_ANALYSIS_CONSTRAINEDOPTIMIZATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

class Function;
class Instruction;
class Loop;
class Module;

enum class CoptMode {
  Off,
  Discover,
  Apply,
  Integrated,
};

enum class CoptPhase {
  None,
  Discovery,

  // Temporary replay on a disposable module used only to evaluate a
  // candidate function plan.
  Evaluation,

  // Final replay on the untouched real LTO module.
  Apply,
};

enum class CoptKind {
  Inline,
  FunctionSpecialization,
  NonTrivialUnswitch,
  LoopUnroll,
  LoopVectorize,
};

enum class CoptRegionKind {
  Unknown,
  Function,
  CallSite,
  Loop,
};

struct CoptOrigin {
  std::string DecisionID;
  std::string RegionID;
};

struct CoptAlternative {
  std::string Name;

  // Common metrics. They are still optional because Milestone 3 is primarily
  // about discovering/replaying states rather than globally comparing them.
  std::optional<double> PerfBenefit;
  std::optional<double> SizeCost;

  // Native/pass-specific measurements.
  std::map<std::string, double> Metrics;

  // True for the state LLVM selected in the upper/reference discovery path.
  bool LLVMDefault = false;

  // Endpoint markers.
  //
  // A lower endpoint is the conservative state of this region.
  // An upper endpoint is the state produced by the reference O2/O3 pipeline.
  //
  // For an upper pipeline that already chooses the conservative state,
  // LowerEndpoint and UpperEndpoint may both be true on the same state.
  bool LowerEndpoint = false;
  bool UpperEndpoint = false;

  // False means the state is informative only and should not be selected by
  // the future solver. Every state generated in Milestone 3 is replayable
  // unless explicitly stated otherwise.
  bool Replayable = true;
};

struct CoptCandidate {
  std::string ID;
  CoptKind Kind;

  std::string Function;
  std::string Location;
  std::string Stage;

  // Region-level identity.
  CoptRegionKind RegionKind = CoptRegionKind::Unknown;
  std::string RegionID;

  // Lexical nesting, e.g. inner loop -> outer loop.
  std::string StructuralParentRegionID;

  // If this region was created by another optimization decision, these
  // identify that creator.
  std::string CreatedByDecisionID;
  std::string CreatedByRegionID;

  // Decisions whose generated IR is present in this region. For a loop this
  // may include an inline into an already-existing loop even when that inline
  // did not literally create the loop.
  std::vector<std::string> Dependencies;

  // Sequence is per owning function, not global.
  uint64_t Sequence = 0;

  std::vector<CoptAlternative> Alternatives;
};

class LLVM_ABI CoptContext {
public:
  static CoptContext &get();

  void begin(Module &M);
  void finish(Module &M);

  bool active() const { return Mode != CoptMode::Off; }

  bool isIntegrated() const { return Mode == CoptMode::Integrated; }

  bool isDiscover() const {
    return Mode == CoptMode::Discover ||
           (Mode == CoptMode::Integrated &&
            Phase == CoptPhase::Discovery);
  }

  // Existing transformation hooks use isApply() to ask whether decisions should
  // be replayed. Evaluation is also a replay phase, so return true there.
  bool isApply() const {
    return Mode == CoptMode::Apply ||
          (Mode == CoptMode::Integrated &&
            (Phase == CoptPhase::Evaluation ||
            Phase == CoptPhase::Apply));
  }

  bool isEvaluation() const {
    return Mode == CoptMode::Integrated &&
          Phase == CoptPhase::Evaluation;
  }

  void invalidateEvaluation(StringRef Reason);

  bool evaluationInvalid() const {
    return EvaluationReplayInvalid;
  }

  StringRef evaluationInvalidReason() const {
    return EvaluationReplayFailure;
  }

  // Use this for bookkeeping that must only describe the user-visible final
  // apply, not temporary evaluation clones.
  bool isFinalApply() const {
    return Mode == CoptMode::Apply ||
          (Mode == CoptMode::Integrated &&
            Phase == CoptPhase::Apply);
  }

  CoptMode mode() const { return Mode; }
  CoptPhase phase() const { return Phase; }

  void beginIntegratedDiscovery(Module &M);
  void endIntegratedDiscovery(Module &M);

  // Temporarily replace the current plan while a disposable evaluation clone is
  // optimized. endIntegratedEvaluation() restores the previous plan.
  void beginIntegratedEvaluation(
      Module &M,
      const std::unordered_map<std::string, std::string> &Choices);

  void endIntegratedEvaluation(Module &M);

  void beginIntegratedApply(Module &M);

  void report(CoptCandidate C);

  ArrayRef<CoptCandidate> candidates() const {
    return Candidates;
  }

  bool hasCandidate(StringRef ID) const;

  std::optional<std::string> selection(StringRef ID) const;

  void installChoice(StringRef ID, StringRef Alternative);
  void clearPlan();

  void noteSeen(StringRef ID);
  void noteApplied(StringRef ID, StringRef Alternative);

  // -----------------------------------------------------------------------
  // Decision IDs
  // -----------------------------------------------------------------------

  std::string makeKeyID(CoptKind Kind, StringRef Key) const;

  std::string makeInstructionID(CoptKind Kind, const Instruction &I,
                                StringRef Extra = "") const;

  std::string makeLoopID(CoptKind Kind, const Loop &L,
                         StringRef Extra = "") const;

  // -----------------------------------------------------------------------
  // Region IDs
  // -----------------------------------------------------------------------

  std::string makeFunctionRegionID(const Function &F) const;

  // Compute the stable region ID without modifying IR.
  std::string computeInstructionRegionID(
      CoptRegionKind Kind, const Instruction &I,
      StringRef Extra = "") const;

  std::string computeLoopRegionID(const Loop &L) const;

  // Discovery-only wrappers which additionally attach the computed ID as
  // metadata to the disposable discovery IR.
  std::string getOrCreateInstructionRegionID(
      CoptRegionKind Kind, Instruction &I,
      StringRef Extra = "") const;

  std::string getOrCreateLoopRegionID(Loop &L) const;

  // -----------------------------------------------------------------------
  // Optimization-origin metadata
  // -----------------------------------------------------------------------

  // Tag newly created discovery IR with the decision and region that created
  // it. The metadata format is:
  //
  //   !copt.origin = !{!"<decision-id>", !"<region-id>"}
  //
  void tagInstructionOrigin(Instruction &I,
                            StringRef DecisionID,
                            StringRef RegionID) const;

  std::optional<CoptOrigin>
  getInstructionOrigin(const Instruction &I) const;

  // Populate provenance for an instruction/callsite candidate.
  void populateInstructionProvenance(CoptCandidate &C,
                                     const Instruction &I) const;

  // Populate provenance for a loop candidate.
  //
  // - structural parent comes from Loop::getParentLoop()
  // - creator comes from provenance present in the loop header
  // - dependencies are collected from all instructions in the loop
  void populateLoopProvenance(CoptCandidate &C, Loop &L) const;

private:
  CoptContext() = default;

  void reset();

  void loadPlan();

  void writeCandidates() const;
  void writeApplyReport() const;

  CoptMode Mode = CoptMode::Off;
  CoptPhase Phase = CoptPhase::None;

  std::string CandidatePath;
  std::string PlanPath;
  std::string ApplyReportPath;

  std::vector<CoptCandidate> Candidates;

  std::unordered_map<std::string, std::string> Plan;
  // Used only while evaluating a disposable module clone.
  std::unordered_map<std::string, std::string> SavedPlan;

  // Set only while replaying a disposable exact-evaluation clone.
  // A candidate can be valid in discovery but become non-replayable after
  // different upstream choices alter the program.
  bool EvaluationReplayInvalid = false;
  std::string EvaluationReplayFailure;

  std::unordered_set<std::string> Seen;
  std::unordered_map<std::string, std::string> Applied;

  std::unordered_set<std::string> Reported;

  // Decision ordering is local to each function.
  std::unordered_map<std::string, uint64_t> NextSequenceByFunction;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_CONSTRAINEDOPTIMIZATION_H