#include "llvm/Analysis/CoptNativePolicy.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

namespace {

static std::string canonicalName(StringRef Name) {
  size_t Pos = Name.find(".__uniq.");
  if (Pos != StringRef::npos)
    return Name.substr(0, Pos).str();
  Pos = Name.find(".llvm.");
  if (Pos != StringRef::npos)
    return Name.substr(0, Pos).str();
  return Name.str();
}

static std::string md5(StringRef Input) {
  MD5 Hasher;
  MD5::MD5Result Result;
  SmallString<32> Hex;
  Hasher.update(Input);
  Hasher.final(Result);
  MD5::stringifyResult(Result, Hex);
  return std::string(Hex);
}

static bool envFlag(const char *Name, bool Default = false) {
  const char *Raw = std::getenv(Name);
  if (!Raw)
    return Default;
  StringRef S(Raw);
  return !(S.empty() || S == "0" || S.equals_insensitive("false") ||
           S.equals_insensitive("no"));
}

static unsigned directCallOrdinal(const CallBase &Wanted) {
  const Function *Caller = Wanted.getFunction();
  if (!Caller)
    return 0;
  unsigned Ord = 0;
  for (const Instruction &I : instructions(*Caller)) {
    const auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || !CB->getCalledFunction())
      continue;
    if (CB == &Wanted)
      return Ord;
    ++Ord;
  }
  return Ord;
}

struct DebugKey {
  unsigned Line = 0;
  unsigned Col = 0;
  unsigned Discriminator = 0;
};

static DebugKey debugKey(const Instruction *I) {
  if (!I)
    return {};
  if (DebugLoc DL = I->getDebugLoc())
    return {DL.getLine(), DL.getCol(), DL->getBaseDiscriminator()};
  return {};
}

static std::mutex LoadMutex;
static std::mutex TraceMutex;
static std::mutex StateMutex;

} // namespace

class CoptNativePolicy::Impl {
public:
  // External TSV-selected regions.
  std::unordered_set<std::string> Selected;

  // Internal Full-LTO selected regions. When active this plan has precedence
  // over the external file-backed plan.
  bool InternalPlanActive = false;
  std::unordered_set<std::string> InternalSelected;

  // In-memory shadow tracing used by the internal planner.
  bool InternalTraceActive = false;
  std::vector<CoptNativeTraceRecord> InternalTrace;

  std::unordered_map<std::string, double> UnrollProfileCount;
};

CoptScopedO2Policy::CoptScopedO2Policy(Function &Fn, bool Enable)
    : F(Fn), Enabled(Enable) {
  if (!Enabled)
    return;
  HadOptSize = F.hasFnAttribute(Attribute::OptimizeForSize);
  HadMinSize = F.hasFnAttribute(Attribute::MinSize);
  F.removeFnAttr(Attribute::OptimizeForSize);
  F.removeFnAttr(Attribute::MinSize);
}

CoptScopedO2Policy::~CoptScopedO2Policy() {
  if (!Enabled)
    return;
  F.removeFnAttr(Attribute::OptimizeForSize);
  F.removeFnAttr(Attribute::MinSize);
  if (HadOptSize)
    F.addFnAttr(Attribute::OptimizeForSize);
  if (HadMinSize)
    F.addFnAttr(Attribute::MinSize);
}

CoptNativePolicy &CoptNativePolicy::get() {
  static CoptNativePolicy Instance;
  return Instance;
}

CoptNativePolicy::CoptNativePolicy() : PImpl(new Impl()) {}

void CoptNativePolicy::loadRegionPlan() {
  std::lock_guard<std::mutex> Lock(LoadMutex);
  if (Loaded)
    return;
  Loaded = true;

  if (const char *P = std::getenv("COPT_NATIVE_REGION_FILE"))
    RegionFile = P;
  if (const char *P = std::getenv("COPT_NATIVE_TRACE_FILE"))
    TraceFile = P;
  if (TraceFile.empty())
    if (const char *P = std::getenv("COPT_INTERNAL_TRACE_FILE"))
      TraceFile = P;
  LogEnabled = envFlag("COPT_NATIVE_LOG", false);

  if (RegionFile.empty())
    return;

  std::ifstream In(RegionFile);
  if (!In)
    report_fatal_error(Twine("COPT-NATIVE: cannot open region policy file: ") +
                       RegionFile);

  std::string Raw;
  unsigned LineNo = 0;
  while (std::getline(In, Raw)) {
    ++LineNo;
    StringRef Line(Raw);
    Line = Line.trim();
    if (Line.empty() || Line.starts_with("#"))
      continue;

    SmallVector<StringRef, 4> Fields;
    Line.split(Fields, '\t', -1, false);
    if (Fields.size() < 3) {
      Fields.clear();
      Line.split(Fields, ',', -1, false);
    }
    if (Fields.size() < 3)
      report_fatal_error(Twine("COPT-NATIVE: malformed region policy line ") +
                         Twine(LineNo) + " in " + RegionFile);

    StringRef Kind = Fields[0].trim();
    StringRef ID = Fields[1].trim();
    StringRef Policy = Fields[2].trim();

    if (LineNo == 1 && Kind.equals_insensitive("kind") &&
        ID.equals_insensitive("region_id"))
      continue;

    if (!(Kind == "inline" || Kind == "unroll"))
      report_fatal_error(Twine("COPT-NATIVE: unsupported regional kind '") +
                         Kind + "' in " + RegionFile);
    if (!Policy.equals_insensitive("o2"))
      report_fatal_error(Twine("COPT-NATIVE: v1 regional policy only supports "
                               "monotonic promotion to O2; got '") +
                         Policy + "'");

    PImpl->Selected.insert((Twine(Kind) + ":" + ID).str());
  }

  if (LogEnabled)
    errs() << "[COPT-NATIVE] loaded " << PImpl->Selected.size()
           << " regional O2 promotions from " << RegionFile << "\n";
}

bool CoptNativePolicy::selected(StringRef Kind, StringRef RegionID) const {
  return PImpl->Selected.count((Twine(Kind) + ":" + RegionID).str()) != 0;
}

std::string CoptNativePolicy::inlineRegionID(const CallBase &CB) const {
  const Function *Caller = CB.getFunction();
  const Function *Callee = CB.getCalledFunction();
  DebugKey DK = debugKey(&CB);
  std::string CallerName = Caller ? canonicalName(Caller->getName()) : "<none>";
  std::string CalleeName = Callee ? canonicalName(Callee->getName()) : "<indirect>";
  unsigned Ordinal = directCallOrdinal(CB);

  // Prefer source identity because earlier policy differences can change the
  // current instruction ordinal.  Ordinal is only a fallback when debug
  // information is unavailable.
  std::string Site = DK.Line
                         ? (Twine(DK.Line) + ":" + Twine(DK.Col) + ":" +
                            Twine(DK.Discriminator)).str()
                         : (Twine("ordinal=") + Twine(Ordinal)).str();
  std::string Key =
      (Twine("inline|") + CallerName + "|" + CalleeName + "|" + Site).str();
  return md5(Key);
}

std::string CoptNativePolicy::unrollRegionID(const Loop &L) const {
  const BasicBlock *Header = L.getHeader();
  const Function *F = Header ? Header->getParent() : nullptr;
  const Instruction *Anchor = Header ? Header->getTerminator() : nullptr;
  DebugKey DK = debugKey(Anchor);

  std::string Fn = F ? canonicalName(F->getName()) : "<none>";
  std::string HeaderName = Header ? Header->getName().str() : "<none>";
  std::string Site = DK.Line
                         ? (Twine(DK.Line) + ":" + Twine(DK.Col) + ":" +
                            Twine(DK.Discriminator)).str()
                         : (Twine("header=") + HeaderName).str();
  std::string Key = (Twine("unroll|") + Fn + "|depth=" +
                     Twine(L.getLoopDepth()) + "|" + Site).str();
  return md5(Key);
}

bool CoptNativePolicy::useO2ForInline(const CallBase &CB) {
  const std::string ID = inlineRegionID(CB);
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    if (PImpl->InternalPlanActive)
      return PImpl->InternalSelected.count((Twine("inline:") + ID).str()) != 0;
  }

  loadRegionPlan();
  if (RegionFile.empty())
    return false;
  return selected("inline", ID);
}

bool CoptNativePolicy::useO2ForUnroll(const Loop &L) {
  const std::string ID = unrollRegionID(L);
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    if (PImpl->InternalPlanActive)
      return PImpl->InternalSelected.count((Twine("unroll:") + ID).str()) != 0;
  }

  loadRegionPlan();
  if (RegionFile.empty())
    return false;
  return selected("unroll", ID);
}

void CoptNativePolicy::appendTrace(StringRef Kind, StringRef RegionID,
                                   StringRef Function, StringRef Subject,
                                   StringRef Outcome, double ProfileCount,
                                   double MetricA, double MetricB) {
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    if (PImpl->InternalTraceActive) {
      CoptNativeTraceRecord R;
      R.Kind = Kind.str();
      R.RegionID = RegionID.str();
      R.Function = Function.str();
      R.Subject = Subject.str();
      R.Outcome = Outcome.str();
      R.ProfileCount = ProfileCount;
      R.MetricA = MetricA;
      R.MetricB = MetricB;
      PImpl->InternalTrace.push_back(std::move(R));
    }
  }

  loadRegionPlan();
  if (TraceFile.empty())
    return;

  std::lock_guard<std::mutex> Lock(TraceMutex);
  bool NeedHeader = false;
  {
    std::ifstream Existing(TraceFile, std::ios::binary | std::ios::ate);
    NeedHeader = !Existing || Existing.tellg() == 0;
  }

  std::ofstream Out(TraceFile, std::ios::app);
  if (!Out)
    report_fatal_error(Twine("COPT-NATIVE: cannot append trace file: ") +
                       TraceFile);
  if (NeedHeader)
    Out << "kind\tregion_id\tfunction\tsubject\toutcome\tprofile_count"
           "\tmetric_a\tmetric_b\n";
  Out << Kind.str() << '\t' << RegionID.str() << '\t' << Function.str() << '\t'
      << Subject.str() << '\t' << Outcome.str() << '\t' << ProfileCount << '\t'
      << MetricA << '\t' << MetricB << '\n';
}

void CoptNativePolicy::traceInline(const CallBase &CB, StringRef RegionID,
                                   bool Profitable, double ProfileCount,
                                   int Cost, int Threshold) {
  const Function *Caller = CB.getFunction();
  const Function *Callee = CB.getCalledFunction();
  appendTrace("inline", RegionID,
              Caller ? canonicalName(Caller->getName()) : "<none>",
              Callee ? canonicalName(Callee->getName()) : "<indirect>",
              Profitable ? "inline" : "no-inline", ProfileCount, Cost,
              Threshold);
}

void CoptNativePolicy::noteUnrollProfile(const Loop &L, double ProfileCount) {
  std::string RegionID = unrollRegionID(L);
  std::lock_guard<std::mutex> Lock(StateMutex);
  auto It = PImpl->UnrollProfileCount.find(RegionID);
  if (It == PImpl->UnrollProfileCount.end())
    PImpl->UnrollProfileCount.emplace(std::move(RegionID), ProfileCount);
  else
    It->second = std::max(It->second, ProfileCount);
}

void CoptNativePolicy::traceUnroll(const Loop &L, StringRef RegionID,
                                   unsigned Count) {
  const BasicBlock *Header = L.getHeader();
  const Function *F = Header ? Header->getParent() : nullptr;
  double ProfileCount = 0.0;
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    auto It = PImpl->UnrollProfileCount.find(RegionID.str());
    if (It != PImpl->UnrollProfileCount.end())
      ProfileCount = It->second;
  }
  std::string Subject = Header ? Header->getName().str() : "<none>";
  appendTrace("unroll", RegionID,
              F ? canonicalName(F->getName()) : "<none>",
              Subject, (Twine("count=") + Twine(Count)).str(), ProfileCount,
              Count, L.getLoopDepth());
}

void CoptNativePolicy::beginInternalTrace() {
  std::lock_guard<std::mutex> Lock(StateMutex);
  PImpl->InternalTrace.clear();
  PImpl->UnrollProfileCount.clear();
  PImpl->InternalTraceActive = true;
}

std::vector<CoptNativeTraceRecord> CoptNativePolicy::endInternalTrace() {
  std::lock_guard<std::mutex> Lock(StateMutex);
  PImpl->InternalTraceActive = false;
  std::vector<CoptNativeTraceRecord> Result;
  Result.swap(PImpl->InternalTrace);
  PImpl->UnrollProfileCount.clear();
  return Result;
}

void CoptNativePolicy::setInternalRegionPlan(
    ArrayRef<CoptNativeRegionSelection> Regions) {
  std::lock_guard<std::mutex> Lock(StateMutex);
  PImpl->InternalSelected.clear();
  for (const CoptNativeRegionSelection &R : Regions) {
    if (!(R.Kind == "inline" || R.Kind == "unroll"))
      report_fatal_error(Twine("COPT-NATIVE: invalid internal regional kind '") +
                         R.Kind + "'");
    PImpl->InternalSelected.insert((Twine(R.Kind) + ":" + R.RegionID).str());
  }
  PImpl->InternalPlanActive = true;

  if (LogEnabled)
    errs() << "[COPT-NATIVE] installed " << PImpl->InternalSelected.size()
           << " in-memory regional O2 promotions\n";
}

void CoptNativePolicy::clearInternalRegionPlan() {
  std::lock_guard<std::mutex> Lock(StateMutex);
  PImpl->InternalPlanActive = false;
  PImpl->InternalSelected.clear();
}

