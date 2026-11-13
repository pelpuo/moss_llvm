#include "llvm/Analysis/ConstrainedOptimization.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <system_error>

using namespace llvm;

namespace {

constexpr StringLiteral CoptCallSiteRegionMD =
    "copt.region.callsite";

constexpr StringLiteral CoptLoopRegionMD =
    "copt.region.loop";

constexpr StringLiteral CoptOriginMD =
    "copt.origin";

static StringRef kindName(CoptKind Kind) {
  switch (Kind) {
  case CoptKind::Inline:
    return "inline";
  case CoptKind::FunctionSpecialization:
    return "function-specialization";
  case CoptKind::NonTrivialUnswitch:
    return "nontrivial-unswitch";
  case CoptKind::LoopUnroll:
    return "loop-unroll";
  case CoptKind::LoopVectorize:
    return "loop-vectorize";
  }

  llvm_unreachable("unknown COPT kind");
}

static StringRef regionKindName(CoptRegionKind Kind) {
  switch (Kind) {
  case CoptRegionKind::Unknown:
    return "unknown";
  case CoptRegionKind::Function:
    return "function";
  case CoptRegionKind::CallSite:
    return "callsite";
  case CoptRegionKind::Loop:
    return "loop";
  }

  llvm_unreachable("unknown COPT region kind");
}

static std::string md5(StringRef Input) {
  MD5 Hasher;
  MD5::MD5Result Hash;
  SmallString<32> Hex;

  Hasher.update(Input);
  Hasher.final(Hash);
  MD5::stringifyResult(Hash, Hex);

  return std::string(Hex);
}

static std::string debugLocation(const Instruction &I) {
  if (DebugLoc DL = I.getDebugLoc()) {
    return formatv("{0}:{1}:{2}",
                   DL.getLine(),
                   DL.getCol(),
                   DL->getBaseDiscriminator())
        .str();
  }

  unsigned Index = 0;

  for (const Instruction &Cur : *I.getParent()) {
    if (&Cur == &I)
      break;

    ++Index;
  }

  return formatv("bb={0}:inst={1}",
                 I.getParent()->getName(),
                 Index)
      .str();
}

static std::string loopLocation(const Loop &L) {
  const BasicBlock *Header = L.getHeader();

  if (!Header)
    return "<no-header>";

  for (const Instruction &I : *Header)
    if (I.getDebugLoc())
      return debugLocation(I);

  return Header->getName().str();
}

static std::optional<std::string>
readSingleStringMetadata(const Instruction &I,
                         StringRef Name) {
  MDNode *N = I.getMetadata(Name);

  if (!N || N->getNumOperands() < 1)
    return std::nullopt;

  auto *S =
      dyn_cast_or_null<MDString>(N->getOperand(0).get());

  if (!S)
    return std::nullopt;

  return S->getString().str();
}

static void writeSingleStringMetadata(Instruction &I,
                                      StringRef Name,
                                      StringRef Value) {
  LLVMContext &Ctx = I.getContext();

  Metadata *Ops[] = {
      MDString::get(Ctx, Value),
  };

  I.setMetadata(Name, MDNode::get(Ctx, Ops));
}

static json::Object
alternativeToJSON(const CoptAlternative &A) {
  json::Object O;

  O["name"] = A.Name;
  O["llvm_default"] = A.LLVMDefault;

  O["lower_endpoint"] = A.LowerEndpoint;
  O["upper_endpoint"] = A.UpperEndpoint;
  O["replayable"] = A.Replayable;

  if (A.PerfBenefit)
    O["perf_benefit"] = *A.PerfBenefit;

  if (A.SizeCost)
    O["size_cost"] = *A.SizeCost;

  json::Object Metrics;

  for (const auto &[Name, Value] : A.Metrics)
    Metrics[Name] = Value;

  O["metrics"] = std::move(Metrics);

  return O;
}

static void appendUnique(std::vector<std::string> &V,
                         StringRef Value) {
  if (Value.empty())
    return;

  if (llvm::find(V, Value.str()) == V.end())
    V.push_back(Value.str());
}

} // namespace

CoptContext &CoptContext::get() {
  static CoptContext Instance;
  return Instance;
}

void CoptContext::reset() {
  Mode = CoptMode::Off;
  Phase = CoptPhase::None;

  CandidatePath.clear();
  PlanPath.clear();
  ApplyReportPath.clear();

  Candidates.clear();
  Plan.clear();
  SavedPlan.clear();

  Seen.clear();
  Applied.clear();

  Reported.clear();
  NextSequenceByFunction.clear();

  EvaluationReplayInvalid = false;
EvaluationReplayFailure.clear();
}

void CoptContext::begin(Module &) {
  reset();

  const char *RawMode = std::getenv("COPT_MODE");

  if (!RawMode)
    return;

  StringRef M(RawMode);

  if (M.equals_insensitive("discover"))
    Mode = CoptMode::Discover;
  else if (M.equals_insensitive("apply"))
    Mode = CoptMode::Apply;
  else if (M.equals_insensitive("integrated"))
    Mode = CoptMode::Integrated;
  else if (M.equals_insensitive("off"))
    Mode = CoptMode::Off;
  else
    report_fatal_error(
        Twine("COPT: invalid COPT_MODE: ") + M);

  if (Mode == CoptMode::Off)
    return;

  if (const char *P = std::getenv("COPT_CANDIDATES"))
    CandidatePath = P;

  if (const char *P = std::getenv("COPT_PLAN"))
    PlanPath = P;

  if (const char *P = std::getenv("COPT_APPLY_REPORT"))
    ApplyReportPath = P;

  if (Mode == CoptMode::Discover) {
    Phase = CoptPhase::Discovery;

    if (CandidatePath.empty())
      report_fatal_error(
          "COPT: COPT_CANDIDATES is required in discover mode");
  }

  if (Mode == CoptMode::Apply) {
    Phase = CoptPhase::Apply;

    if (PlanPath.empty())
      report_fatal_error(
          "COPT: COPT_PLAN is required in apply mode");

    loadPlan();
  }

  if (Mode == CoptMode::Integrated) {
    if (!PlanPath.empty())
      loadPlan();
  }
}

void CoptContext::finish(Module &) {
  if (Mode == CoptMode::Discover) {
    writeCandidates();
    Phase = CoptPhase::None;
    return;
  }

  if (Mode == CoptMode::Apply) {
    writeApplyReport();
    Phase = CoptPhase::None;
    return;
  }

  if (Mode == CoptMode::Integrated) {
    writeApplyReport();
    Phase = CoptPhase::None;
  }
}

void CoptContext::beginIntegratedDiscovery(Module &) {
  if (Mode != CoptMode::Integrated)
    report_fatal_error(
        "COPT: beginIntegratedDiscovery called outside integrated mode");

  Phase = CoptPhase::Discovery;

  Candidates.clear();
  Reported.clear();
  NextSequenceByFunction.clear();

  Seen.clear();
  Applied.clear();
}

void CoptContext::endIntegratedDiscovery(Module &) {
  if (Mode != CoptMode::Integrated ||
      Phase != CoptPhase::Discovery)
    report_fatal_error(
        "COPT: endIntegratedDiscovery called in the wrong phase");

  writeCandidates();

  Phase = CoptPhase::None;
}

void CoptContext::beginIntegratedEvaluation(
    Module &,
    const std::unordered_map<std::string, std::string> &Choices) {

  if (Mode != CoptMode::Integrated)
    report_fatal_error(
        "COPT: beginIntegratedEvaluation called outside integrated mode");

  if (Phase != CoptPhase::None)
    report_fatal_error(
        "COPT: beginIntegratedEvaluation called while another COPT phase "
        "is active");

  SavedPlan = Plan;
  Plan = Choices;

  EvaluationReplayInvalid = false;
  EvaluationReplayFailure.clear();

  Phase = CoptPhase::Evaluation;

  Seen.clear();
  Applied.clear();
}

void CoptContext::endIntegratedEvaluation(Module &) {
  if (Mode != CoptMode::Integrated ||
      Phase != CoptPhase::Evaluation)
    report_fatal_error(
        "COPT: endIntegratedEvaluation called in the wrong phase");

  Plan = std::move(SavedPlan);
  SavedPlan.clear();

  Seen.clear();
  Applied.clear();

  EvaluationReplayInvalid = false;
  EvaluationReplayFailure.clear();

  Phase = CoptPhase::None;
}

void CoptContext::invalidateEvaluation(StringRef Reason) {
  if (!isEvaluation())
    report_fatal_error(
        "COPT: invalidateEvaluation called outside evaluation mode");

  // Preserve the first failure. It is normally the most useful explanation
  // of why this candidate ceased to be replayable.
  if (!EvaluationReplayInvalid) {
    EvaluationReplayInvalid = true;
    EvaluationReplayFailure = Reason.str();
  }
}

void CoptContext::beginIntegratedApply(Module &) {
  if (Mode != CoptMode::Integrated)
    report_fatal_error(
        "COPT: beginIntegratedApply called outside integrated mode");

  Phase = CoptPhase::Apply;

  Seen.clear();
  Applied.clear();
}

void CoptContext::report(CoptCandidate C) {
  if (!active() || C.ID.empty())
    return;

  if (!isDiscover())
    return;

  if (!Reported.insert(C.ID).second)
    return;

  if (C.RegionID.empty())
    C.RegionID = C.ID;

  C.Sequence =
      NextSequenceByFunction[C.Function]++;

  Candidates.push_back(std::move(C));
}

bool CoptContext::hasCandidate(StringRef ID) const {
  return Reported.count(ID.str()) != 0;
}

void CoptContext::noteSeen(StringRef ID) {
  if (isFinalApply() && !ID.empty())
    Seen.insert(ID.str());
}

void CoptContext::noteApplied(StringRef ID,
                              StringRef Alternative) {
  if (!isFinalApply() || ID.empty())
    return;

  Seen.insert(ID.str());
  Applied[ID.str()] = Alternative.str();
}

std::optional<std::string>
CoptContext::selection(StringRef ID) const {
  if (!isApply())
    return std::nullopt;

  auto It = Plan.find(ID.str());

  if (It == Plan.end())
    return std::nullopt;

  return It->second;
}

void CoptContext::installChoice(StringRef ID,
                                StringRef Alternative) {
  Plan[ID.str()] = Alternative.str();
}

void CoptContext::clearPlan() {
  Plan.clear();
}

// ---------------------------------------------------------------------------
// Decision IDs
// ---------------------------------------------------------------------------

std::string
CoptContext::makeKeyID(CoptKind Kind,
                       StringRef Key) const {
  std::string FullKey =
      formatv("{0}|{1}",
              kindName(Kind),
              Key)
          .str();

  return md5(FullKey);
}

std::string
CoptContext::makeInstructionID(CoptKind Kind,
                               const Instruction &I,
                               StringRef Extra) const {
  const Function *F = I.getFunction();

  std::string Key =
      formatv("{0}|function={1}|loc={2}|opcode={3}|extra={4}",
              kindName(Kind),
              F ? F->getName() : "<null>",
              debugLocation(I),
              I.getOpcodeName(),
              Extra)
          .str();

  return md5(Key);
}

std::string
CoptContext::makeLoopID(CoptKind Kind,
                        const Loop &L,
                        StringRef Extra) const {
  const BasicBlock *Header = L.getHeader();
  const Function *F =
      Header ? Header->getParent() : nullptr;

  std::string Key =
      formatv("{0}|function={1}|loop-loc={2}|depth={3}|extra={4}",
              kindName(Kind),
              F ? F->getName() : "<null>",
              loopLocation(L),
              L.getLoopDepth(),
              Extra)
          .str();

  return md5(Key);
}

// ---------------------------------------------------------------------------
// Region IDs
// ---------------------------------------------------------------------------

std::string
CoptContext::makeFunctionRegionID(
    const Function &F) const {
  return md5(
      formatv("region|function|{0}", F.getName())
          .str());
}

std::string
CoptContext::computeInstructionRegionID(
    CoptRegionKind Kind,
    const Instruction &I,
    StringRef Extra) const {

  switch (Kind) {
  case CoptRegionKind::CallSite:
    break;

  case CoptRegionKind::Unknown:
  case CoptRegionKind::Function:
  case CoptRegionKind::Loop:
    report_fatal_error(
        "COPT: instruction region helper currently supports "
        "callsite regions only");
  }

  const Function *F = I.getFunction();

  return md5(
      formatv("region|{0}|function={1}|loc={2}|extra={3}",
              regionKindName(Kind),
              F ? F->getName() : "<null>",
              debugLocation(I),
              Extra)
          .str());
}

std::string
CoptContext::getOrCreateInstructionRegionID(
    CoptRegionKind Kind,
    Instruction &I,
    StringRef Extra) const {

  StringRef MDName;

  switch (Kind) {
  case CoptRegionKind::CallSite:
    MDName = CoptCallSiteRegionMD;
    break;

  case CoptRegionKind::Unknown:
  case CoptRegionKind::Function:
  case CoptRegionKind::Loop:
    report_fatal_error(
        "COPT: instruction region helper currently supports "
        "callsite regions only");
  }

  if (auto Existing =
          readSingleStringMetadata(I, MDName))
    return *Existing;

  if (!isDiscover())
    report_fatal_error(
        "COPT: attempted to create instruction-region metadata "
        "outside discovery mode");

  std::string ID =
      computeInstructionRegionID(
          Kind, I, Extra);

  writeSingleStringMetadata(
      I, MDName, ID);

  return ID;
}

std::string
CoptContext::computeLoopRegionID(
    const Loop &L) const {

  const BasicBlock *Header = L.getHeader();

  if (!Header)
    report_fatal_error(
        "COPT: loop without a header");

  const Function *F =
      Header->getParent();

  return md5(
      formatv("region|loop|function={0}|loc={1}|depth={2}",
              F ? F->getName() : "<null>",
              loopLocation(L),
              L.getLoopDepth())
          .str());
}

std::string
CoptContext::getOrCreateLoopRegionID(
    Loop &L) const {

  BasicBlock *Header = L.getHeader();

  if (!Header)
    report_fatal_error(
        "COPT: loop without a header");

  for (Instruction &I : *Header)
    if (auto Existing =
            readSingleStringMetadata(
                I, CoptLoopRegionMD))
      return *Existing;

  if (!isDiscover())
    report_fatal_error(
        "COPT: attempted to create loop-region metadata "
        "outside discovery mode");

  std::string ID =
      computeLoopRegionID(L);

  for (Instruction &I : *Header)
    writeSingleStringMetadata(
        I,
        CoptLoopRegionMD,
        ID);

  return ID;
}

// ---------------------------------------------------------------------------
// Optimization-origin metadata
// ---------------------------------------------------------------------------

void CoptContext::tagInstructionOrigin(
    Instruction &I,
    StringRef DecisionID,
    StringRef RegionID) const {

  if (!isDiscover())
    report_fatal_error(
        "COPT: attempted to tag optimization origin "
        "outside discovery mode");

  LLVMContext &Ctx = I.getContext();

  Metadata *Ops[] = {
      MDString::get(Ctx, DecisionID),
      MDString::get(Ctx, RegionID),
  };

  I.setMetadata(
      CoptOriginMD,
      MDNode::get(Ctx, Ops));
}

std::optional<CoptOrigin>
CoptContext::getInstructionOrigin(
    const Instruction &I) const {

  MDNode *N = I.getMetadata(CoptOriginMD);

  if (!N || N->getNumOperands() < 2)
    return std::nullopt;

  auto *Decision =
      dyn_cast_or_null<MDString>(
          N->getOperand(0).get());

  auto *Region =
      dyn_cast_or_null<MDString>(
          N->getOperand(1).get());

  if (!Decision || !Region)
    return std::nullopt;

  CoptOrigin O;
  O.DecisionID = Decision->getString().str();
  O.RegionID = Region->getString().str();

  return O;
}

void CoptContext::populateInstructionProvenance(
    CoptCandidate &C,
    const Instruction &I) const {

  auto Origin = getInstructionOrigin(I);

  if (!Origin)
    return;

  C.CreatedByDecisionID = Origin->DecisionID;
  C.CreatedByRegionID = Origin->RegionID;

  appendUnique(C.Dependencies,
               Origin->DecisionID);
}

void CoptContext::populateLoopProvenance(
    CoptCandidate &C,
    Loop &L) const {

  // Lexical nesting.
  if (Loop *Parent = L.getParentLoop())
    C.StructuralParentRegionID =
        getOrCreateLoopRegionID(*Parent);

  // If the loop header itself consists of IR created by an earlier
  // transformation, treat that transformation as the creator of this region.
  if (BasicBlock *Header = L.getHeader()) {
    for (Instruction &I : *Header) {
      auto Origin = getInstructionOrigin(I);

      if (!Origin)
        continue;

      if (C.CreatedByDecisionID.empty()) {
        C.CreatedByDecisionID =
            Origin->DecisionID;
        C.CreatedByRegionID =
            Origin->RegionID;
      }

      appendUnique(C.Dependencies,
                   Origin->DecisionID);
    }
  }

  // More generally, collect every transformation origin represented inside
  // the loop. This captures cases where an inline changes an existing loop's
  // body/profitability without literally creating the loop header.
  for (BasicBlock *BB : L.blocks()) {
    for (Instruction &I : *BB) {
      auto Origin = getInstructionOrigin(I);

      if (!Origin)
        continue;

      appendUnique(C.Dependencies,
                   Origin->DecisionID);
    }
  }
}

// ---------------------------------------------------------------------------
// Plan loading / JSON output
// ---------------------------------------------------------------------------

void CoptContext::loadPlan() {
  auto BufferOrErr =
      MemoryBuffer::getFile(PlanPath);

  if (!BufferOrErr)
    report_fatal_error(
        Twine("COPT: cannot open plan file: ") +
        PlanPath);

  Expected<json::Value> Parsed =
      json::parse(
          BufferOrErr.get()->getBuffer());

  if (!Parsed)
    report_fatal_error(
        Twine("COPT: invalid JSON plan: ") +
        toString(Parsed.takeError()));

  json::Object *Root =
      Parsed->getAsObject();

  if (!Root)
    report_fatal_error(
        "COPT: plan root must be a JSON object");

  json::Object *Choices =
      Root->getObject("choices");

  if (!Choices)
    report_fatal_error(
        "COPT: plan must contain a 'choices' object");

  for (auto &KV : *Choices) {
    std::optional<StringRef> Choice =
        KV.second.getAsString();

    if (!Choice)
      report_fatal_error(
          Twine("COPT: choice for candidate '") +
          KV.first.str() +
          "' must be a string");

    Plan[KV.first.str()] =
        Choice->str();
  }
}

void CoptContext::writeCandidates() const {
  if (CandidatePath.empty())
    return;

  json::Array CandidateArray;

  for (const CoptCandidate &C : Candidates) {
    json::Object O;

    O["id"] = C.ID;
    O["kind"] = kindName(C.Kind);

    O["function"] = C.Function;
    O["location"] = C.Location;
    O["stage"] = C.Stage;

    O["region_kind"] =
        regionKindName(C.RegionKind);

    O["region_id"] =
        C.RegionID;

    O["structural_parent_region_id"] =
        C.StructuralParentRegionID;

    O["created_by_decision_id"] =
        C.CreatedByDecisionID;

    O["created_by_region_id"] =
        C.CreatedByRegionID;

    O["sequence"] =
        static_cast<int64_t>(C.Sequence);

    json::Array Dependencies;

    for (const std::string &D : C.Dependencies)
      Dependencies.push_back(D);

    O["dependencies"] =
        std::move(Dependencies);

    json::Array Alternatives;

    for (const CoptAlternative &A : C.Alternatives)
      Alternatives.push_back(
          alternativeToJSON(A));

    // "states" is the canonical Milestone-3 name.
    //
    // Keep "alternatives" out of the JSON from now on so external tools don't
    // accidentally treat transformation decisions as independent binary choices.
    O["states"] =
        std::move(Alternatives);

    CandidateArray.push_back(
        std::move(O));
  }

  json::Object Root;

    Root["version"] = 4;
    Root["mode"] = "integrated-region-states";
    Root["candidates"] =
      std::move(CandidateArray);

  std::error_code EC;

  raw_fd_ostream OS(
      CandidatePath,
      EC,
      sys::fs::OF_Text);

  if (EC)
    report_fatal_error(
        Twine("COPT: cannot write candidate file: ") +
        EC.message());

  OS << formatv(
      "{0:2}\n",
      json::Value(std::move(Root)));
}

void CoptContext::writeApplyReport() const {
  if (ApplyReportPath.empty())
    return;

  json::Array SeenArray;

  for (const std::string &ID : Seen)
    SeenArray.push_back(ID);

  json::Object AppliedObject;

  for (const auto &[ID, Choice] : Applied)
    AppliedObject[ID] = Choice;

  json::Array MissingArray;

  for (const auto &[ID, Choice] : Plan)
    if (!Seen.count(ID))
      MissingArray.push_back(ID);

  json::Object Root;

  Root["version"] = 3;
  Root["seen"] =
      std::move(SeenArray);
  Root["applied"] =
      std::move(AppliedObject);
  Root["missing"] =
      std::move(MissingArray);

  std::error_code EC;

  raw_fd_ostream OS(
      ApplyReportPath,
      EC,
      sys::fs::OF_Text);

  if (EC)
    report_fatal_error(
        Twine("COPT: cannot write apply report: ") +
        EC.message());

  OS << formatv(
      "{0:2}\n",
      json::Value(std::move(Root)));
}