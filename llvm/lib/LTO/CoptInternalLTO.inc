//===----------------------------------------------------------------------===//
// COPT internal Full-LTO planner v4.4 (exact coarse + regional replay).
//
// This file is intentionally included by llvm/lib/LTO/LTOBackend.cpp after
// runNewPMPassesOnce() has been defined.  It wraps that stock LTO pipeline with
// in-process shadow exploration and then runs the selected mixed policy once on
// the real module.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Analysis/CoptNativePolicy.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

static constexpr const char CoptInternalBuildTag[] = "COPT-INTERNAL-V4.4.0-REGIONAL";

// Keep every symbol in this block under a legacy-compatible prefix so this can be
// compiled in trees that still contain the older experimental COPT helpers.

enum class CoptV2Policy : uint8_t { Oz = 0, Os = 1, O2 = 2 };
enum class CoptV2PolicyMode : uint8_t { Off, AllOz, AllOs, AllO2, Selected };

static StringRef coptV2PolicyName(CoptV2Policy P) {
  switch (P) {
  case CoptV2Policy::Oz:
    return "oz";
  case CoptV2Policy::Os:
    return "os";
  case CoptV2Policy::O2:
    return "o2";
  }
  llvm_unreachable("unknown COPT internal policy");
}

static std::string coptV2CanonicalName(StringRef Name) {
  size_t Pos = Name.find(".__uniq.");
  if (Pos != StringRef::npos)
    return Name.substr(0, Pos).str();
  Pos = Name.find(".llvm.");
  if (Pos != StringRef::npos)
    return Name.substr(0, Pos).str();
  return Name.str();
}

struct CoptV2PolicyContext {
  CoptV2PolicyMode Mode = CoptV2PolicyMode::Off;
  std::unordered_map<std::string, CoptV2Policy> Selected;
};

static thread_local CoptV2PolicyContext CoptV2Context;

static void coptV2SetFunctionPolicy(Function &F, CoptV2Policy P) {
  switch (P) {
  case CoptV2Policy::Oz:
    F.addFnAttr(Attribute::OptimizeForSize);
    F.addFnAttr(Attribute::MinSize);
    break;
  case CoptV2Policy::Os:
    F.addFnAttr(Attribute::OptimizeForSize);
    F.removeFnAttr(Attribute::MinSize);
    break;
  case CoptV2Policy::O2:
    F.removeFnAttr(Attribute::OptimizeForSize);
    F.removeFnAttr(Attribute::MinSize);
    break;
  }
}

static CoptV2Policy coptV2PolicyForFunction(StringRef Name) {
  switch (CoptV2Context.Mode) {
  case CoptV2PolicyMode::AllOz:
    return CoptV2Policy::Oz;
  case CoptV2PolicyMode::AllOs:
    return CoptV2Policy::Os;
  case CoptV2PolicyMode::AllO2:
    return CoptV2Policy::O2;
  case CoptV2PolicyMode::Selected: {
    auto It = CoptV2Context.Selected.find(coptV2CanonicalName(Name));
    return It == CoptV2Context.Selected.end() ? CoptV2Policy::Oz
                                               : It->second;
  }
  case CoptV2PolicyMode::Off:
    return CoptV2Policy::O2;
  }
  llvm_unreachable("unknown COPT internal mode");
}

static void coptV2ApplyActivePolicy(Module &M) {
  if (CoptV2Context.Mode == CoptV2PolicyMode::Off)
    return;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    coptV2SetFunctionPolicy(F, coptV2PolicyForFunction(F.getName()));
  }
}

class CoptV2PolicyPass : public PassInfoMixin<CoptV2PolicyPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    if (CoptV2Context.Mode == CoptV2PolicyMode::Off)
      return PreservedAnalyses::all();
    coptV2ApplyActivePolicy(M);
    return PreservedAnalyses::none();
  }
};

// Forward-declared by the integration patch before runNewPMPassesOnce(). That
// function invokes this immediately after constructing PassBuilder, so the
// active policy is re-asserted at the same Full-LTO extension points used by
// the external prototype.
static void registerCoptInternalPolicyCallbacks(PassBuilder &PB) {
  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(CoptV2PolicyPass());
      });

  PB.registerFullLinkTimeOptimizationEarlyEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(CoptV2PolicyPass());
      });

  PB.registerOptimizerEarlyEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel,
         ThinOrFullLTOPhase) {
        MPM.addPass(CoptV2PolicyPass());
      });
}

static cl::opt<bool> CoptV2EnableFlag(
    "copt-internal", cl::init(false), cl::Hidden,
    cl::desc("Run COPT policy discovery/selection inside Full LTO"));
static cl::opt<double> CoptV2GrowthFlag(
    "copt-size-growth-percent", cl::init(-1.0), cl::Hidden,
    cl::desc("COPT allowed size growth percentage relative to baseline"));
static cl::opt<uint64_t> CoptV2BaselineFlag(
    "copt-baseline-size-bytes", cl::init(0), cl::Hidden,
    cl::desc("COPT true baseline linked size in bytes"));
static cl::opt<uint64_t> CoptV2MaxBytesFlag(
    "copt-max-size-bytes", cl::init(0), cl::Hidden,
    cl::desc("COPT absolute maximum linked size in bytes"));
static cl::opt<std::string> CoptV2ReportFlag(
    "copt-internal-report", cl::init(""), cl::Hidden,
    cl::desc("COPT internal planner JSON report path"));

static bool coptV2EnvFlag(const char *Name, bool Default = false) {
  const char *Raw = std::getenv(Name);
  if (!Raw)
    return Default;
  StringRef S(Raw);
  return !(S.empty() || S == "0" || S.equals_insensitive("false") ||
           S.equals_insensitive("no") || S.equals_insensitive("off"));
}

static std::optional<double> coptV2EnvDouble(const char *Name) {
  const char *Raw = std::getenv(Name);
  if (!Raw || !*Raw)
    return std::nullopt;
  errno = 0;
  char *End = nullptr;
  double V = std::strtod(Raw, &End);
  if (errno || End == Raw || *End != '\0' || !std::isfinite(V))
    report_fatal_error(Twine("COPT-INTERNAL: invalid floating-point value for ") +
                       Name);
  return V;
}

static std::optional<uint64_t> coptV2EnvUInt64(const char *Name) {
  const char *Raw = std::getenv(Name);
  if (!Raw || !*Raw)
    return std::nullopt;
  uint64_t V = 0;
  if (StringRef(Raw).getAsInteger(10, V))
    report_fatal_error(Twine("COPT-INTERNAL: invalid integer value for ") + Name);
  return V;
}

static unsigned coptV2EnvUnsigned(const char *Name, unsigned Default) {
  auto V = coptV2EnvUInt64(Name);
  if (!V)
    return Default;
  if (*V > std::numeric_limits<unsigned>::max())
    report_fatal_error(Twine("COPT-INTERNAL: value too large for ") + Name);
  return static_cast<unsigned>(*V);
}

static std::string coptV2EnvString(const char *Name, StringRef Default = "") {
  const char *Raw = std::getenv(Name);
  return Raw ? std::string(Raw) : Default.str();
}

struct CoptV2Options {
  bool Enabled = false;
  bool Log = true;
  bool ForceAllOz = false;
  std::optional<CoptV2Policy> ForcePolicy;
  bool RequireDeferredPreLink = true;
  bool PerTUPreLink = true;
  std::string FunctionModelPath;
  std::string FixedPolicyPath;
  std::string RegionFilePath;
  double RequestedGrowthPercent = -1.0;
  uint64_t BaselineBytes = 0;
  uint64_t RequestedMaxBytes = 0;
  uint64_t EffectiveMaxBytes = 0;
  uint64_t PolicyOzAnchorBytes = 0;
  std::optional<uint64_t> PolicyOsBytes;
  std::optional<uint64_t> PolicyO2Bytes;
  double BudgetScale = 1.0;
  double OsUtilityFactor = 0.5;
  unsigned DPBuckets = 4096;
  unsigned MaxRefinedFunctions = 16;
  unsigned MaxDifferentialRegions = 128;
  unsigned MaxRegionsPerFunction = 16;
  unsigned MaxPartialPlansPerFunction = 8;
  double RegionalMinResidualRatio = 0.10;
  uint64_t RegionalMinResidualBytes = 1024;
  std::string ReportPath;
};

static std::optional<CoptV2Policy> coptV2ParseOptionalPolicy(StringRef Raw) {
  Raw = Raw.trim();
  if (Raw.empty())
    return std::nullopt;
  if (Raw.equals_insensitive("oz"))
    return CoptV2Policy::Oz;
  if (Raw.equals_insensitive("os"))
    return CoptV2Policy::Os;
  if (Raw.equals_insensitive("o2"))
    return CoptV2Policy::O2;
  report_fatal_error(Twine("COPT-INTERNAL: invalid COPT_INTERNAL_FORCE_POLICY='") +
                     Raw + "'; expected oz, os, or o2");
}

static CoptV2Options coptV2LoadOptions() {
  CoptV2Options O;
  O.Enabled = CoptV2EnableFlag.getValue() || coptV2EnvFlag("COPT_INTERNAL", false);
  if (!O.Enabled)
    return O;

  O.Log = coptV2EnvFlag("COPT_INTERNAL_LOG", true);
  O.ForceAllOz = coptV2EnvFlag("COPT_INTERNAL_FORCE_ALL_OZ", false);
  if (const char *P = std::getenv("COPT_INTERNAL_FORCE_POLICY"))
    O.ForcePolicy = coptV2ParseOptionalPolicy(P);
  if (O.ForceAllOz && O.ForcePolicy && *O.ForcePolicy != CoptV2Policy::Oz)
    report_fatal_error("COPT-INTERNAL: FORCE_ALL_OZ conflicts with FORCE_POLICY");
  if (O.ForceAllOz)
    O.ForcePolicy = CoptV2Policy::Oz;
  O.RequireDeferredPreLink =
      coptV2EnvFlag("COPT_INTERNAL_REQUIRE_DEFERRED_PRELINK", true);
  O.PerTUPreLink =
      coptV2EnvFlag("COPT_INTERNAL_PER_TU_PRELINK", true);
  O.FunctionModelPath = coptV2EnvString("COPT_INTERNAL_FUNCTION_MODEL");
  O.FixedPolicyPath = coptV2EnvString("COPT_INTERNAL_FIXED_POLICY");
  O.RegionFilePath = coptV2EnvString("COPT_INTERNAL_REGION_FILE");
  if (O.ForcePolicy && !O.FixedPolicyPath.empty())
    report_fatal_error(
        "COPT-INTERNAL: COPT_INTERNAL_FIXED_POLICY conflicts with "
        "COPT_INTERNAL_FORCE_POLICY/FORCE_ALL_OZ");
  O.BaselineBytes =
      coptV2EnvUInt64("COPT_BASELINE_SIZE_BYTES").value_or(CoptV2BaselineFlag.getValue());
  if (!O.BaselineBytes)
    report_fatal_error(
        "COPT-INTERNAL: COPT_BASELINE_SIZE_BYTES (or "
        "-copt-baseline-size-bytes) is required");

  if (auto P = coptV2EnvDouble("COPT_SIZE_GROWTH_PERCENT"))
    O.RequestedGrowthPercent = *P;
  else
    O.RequestedGrowthPercent = CoptV2GrowthFlag.getValue();

  if (O.RequestedGrowthPercent < 0.0 && CoptV2MaxBytesFlag.getValue() == 0 &&
      !coptV2EnvUInt64("COPT_MAX_SIZE_BYTES"))
    report_fatal_error(
        "COPT-INTERNAL: provide COPT_SIZE_GROWTH_PERCENT=<any nonnegative "
        "percentage> or COPT_MAX_SIZE_BYTES=<absolute bytes>");

  if (O.RequestedGrowthPercent < 0.0)
    O.RequestedGrowthPercent = 0.0;

  if (auto M = coptV2EnvUInt64("COPT_MAX_SIZE_BYTES"))
    O.RequestedMaxBytes = *M;
  else if (CoptV2MaxBytesFlag.getValue() != 0)
    O.RequestedMaxBytes = CoptV2MaxBytesFlag.getValue();
  else
    O.RequestedMaxBytes = static_cast<uint64_t>(std::floor(
        static_cast<double>(O.BaselineBytes) *
        (1.0 + O.RequestedGrowthPercent / 100.0)));

  if (O.RequestedMaxBytes < O.BaselineBytes)
    report_fatal_error(
        "COPT-INTERNAL: requested maximum size is below the supplied baseline");

  O.BudgetScale = coptV2EnvDouble("COPT_INTERNAL_BUDGET_SCALE").value_or(1.0);
  if (!(O.BudgetScale > 0.0 && O.BudgetScale <= 1.0))
    report_fatal_error("COPT-INTERNAL: COPT_INTERNAL_BUDGET_SCALE must be in (0,1]");

  const double RequestedHeadroom =
      static_cast<double>(O.RequestedMaxBytes - O.BaselineBytes);
  O.EffectiveMaxBytes = O.BaselineBytes + static_cast<uint64_t>(
      std::floor(RequestedHeadroom * O.BudgetScale));

  O.PolicyOzAnchorBytes =
      coptV2EnvUInt64("COPT_INTERNAL_POLICY_OZ_SIZE_BYTES")
          .value_or(O.BaselineBytes);
  O.PolicyOsBytes = coptV2EnvUInt64("COPT_INTERNAL_POLICY_OS_SIZE_BYTES");
  O.PolicyO2Bytes = coptV2EnvUInt64("COPT_INTERNAL_POLICY_O2_SIZE_BYTES");

  O.OsUtilityFactor =
      coptV2EnvDouble("COPT_INTERNAL_OS_UTILITY_FACTOR").value_or(0.5);
  O.DPBuckets = coptV2EnvUnsigned("COPT_INTERNAL_DP_BUCKETS", 4096);
  O.MaxRefinedFunctions =
      coptV2EnvUnsigned("COPT_INTERNAL_MAX_REFINED_FUNCTIONS", 16);
  O.MaxDifferentialRegions =
      coptV2EnvUnsigned("COPT_INTERNAL_MAX_DIFFERENTIAL_REGIONS", 128);
  O.MaxRegionsPerFunction =
      coptV2EnvUnsigned("COPT_INTERNAL_MAX_REGIONS_PER_FUNCTION", 16);
  O.MaxPartialPlansPerFunction =
      coptV2EnvUnsigned("COPT_INTERNAL_MAX_PARTIAL_PLANS_PER_FUNCTION", 8);
  O.RegionalMinResidualRatio =
      coptV2EnvDouble("COPT_INTERNAL_REGIONAL_MIN_RESIDUAL_RATIO")
          .value_or(0.10);
  O.RegionalMinResidualBytes =
      coptV2EnvUInt64("COPT_INTERNAL_REGIONAL_MIN_RESIDUAL_BYTES")
          .value_or(1024);

  O.ReportPath = coptV2EnvString("COPT_INTERNAL_REPORT", CoptV2ReportFlag.getValue());

  if (O.DPBuckets < 32)
    report_fatal_error("COPT-INTERNAL: DP bucket count must be at least 32");
  if (!(O.OsUtilityFactor >= 0.0 && O.OsUtilityFactor <= 1.0))
    report_fatal_error("COPT-INTERNAL: Os utility factor must be in [0,1]");
  if (O.RegionalMinResidualRatio < 0.0)
    report_fatal_error("COPT-INTERNAL: regional residual ratio must be nonnegative");

  // The old external policy mechanism would overwrite this planner's attrs.
  if (const char *P = std::getenv("COPT_POLICY_MODE")) {
    StringRef S(P);
    if (!S.empty() && !S.equals_insensitive("off"))
      report_fatal_error(
          "COPT-INTERNAL: COPT_POLICY_MODE must be unset/off; the internal "
          "planner owns function policy");
  }
  if (const char *P = std::getenv("COPT_NATIVE_REGION_FILE"))
    if (*P)
      report_fatal_error(
          "COPT-INTERNAL: unset COPT_NATIVE_REGION_FILE; the internal planner "
          "owns the regional plan");
  if (const char *P = std::getenv("COPT_NATIVE_TRACE_FILE"))
    if (*P)
      report_fatal_error(
          "COPT-INTERNAL: unset COPT_NATIVE_TRACE_FILE; shadow traces are kept "
          "in memory");

  return O;
}

struct CoptV2FunctionScore {
  double SizeCost = 0.0;
  double ProfileWeight = 0.0;
  uint64_t InstructionCount = 0;
};

struct CoptV2Endpoint {
  CoptV2Policy Policy = CoptV2Policy::Oz;
  double TotalSizeCost = 0.0;
  uint64_t InstructionCount = 0;
  uint64_t FunctionsWithProfile = 0;
  std::unordered_map<std::string, CoptV2FunctionScore> Functions;
  std::vector<CoptNativeTraceRecord> Trace;
};

static CoptV2Endpoint coptV2ScoreModule(
    Module &M, TargetMachine *TM, CoptV2Policy Policy,
    std::vector<CoptNativeTraceRecord> Trace) {
  PassBuilder PB(TM);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  CoptV2Endpoint Result;
  Result.Policy = Policy;
  Result.Trace = std::move(Trace);

  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;

    const std::string Name = coptV2CanonicalName(F.getName());
    auto &FS = Result.Functions[Name];
    auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
    auto &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);

    double EntryCount = 0.0;
    if (auto Count = BFI.getBlockProfileCount(&F.getEntryBlock(),
                                               /*AllowSynthetic=*/true))
      EntryCount = static_cast<double>(*Count);

    if (EntryCount <= 0.0) {
      // Some transformed functions lose a directly recoverable entry count.
      // A block count is still better than treating a profiled function as
      // completely cold; use the largest count observed in the function.
      for (BasicBlock &BB : F)
        if (auto Count = BFI.getBlockProfileCount(&BB,
                                                   /*AllowSynthetic=*/true))
          EntryCount = std::max(EntryCount, static_cast<double>(*Count));
    }

    FS.ProfileWeight = std::max(FS.ProfileWeight, EntryCount);
    if (EntryCount > 0.0)
      ++Result.FunctionsWithProfile;

    for (Instruction &I : instructions(F)) {
      ++FS.InstructionCount;
      ++Result.InstructionCount;
      InstructionCost C =
          TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);
      double V = C.isValid() ? static_cast<double>(C.getValue()) : 1.0;
      // Negative code-size costs are not meaningful as absolute endpoint size.
      V = std::max(0.0, V);
      FS.SizeCost += V;
      Result.TotalSizeCost += V;
    }
  }
  return Result;
}

static double coptV2SafeScale(double ByteDelta, double CostDelta,
                              double FallbackScale) {
  if (std::abs(CostDelta) < 1.0e-9)
    return FallbackScale;
  const double S = ByteDelta / CostDelta;
  if (!std::isfinite(S) || S <= 0.0)
    return FallbackScale;
  return std::clamp(S, FallbackScale * 0.05, FallbackScale * 20.0);
}

static const CoptV2FunctionScore &
coptV2GetFunctionScore(const CoptV2Endpoint &E, StringRef Name) {
  static const CoptV2FunctionScore Zero;
  auto It = E.Functions.find(Name.str());
  return It == E.Functions.end() ? Zero : It->second;
}

struct CoptV2Option {
  CoptV2Policy Policy = CoptV2Policy::Oz;
  double Cost = 0.0;    // estimated bytes relative to policy-Oz anchor
  double Utility = 0.0; // normalized profile utility
};

struct CoptV2FunctionCandidate {
  std::string Name;
  double Hotness = 0.0;
  std::array<CoptV2Option, 3> Options;
};

struct CoptV2Model {
  std::vector<CoptV2FunctionCandidate> Functions;
  double ScaleOs = 1.0;
  double ScaleO2 = 1.0;
  double TotalProfileWeight = 0.0;
};

static CoptV2Model coptV2BuildFunctionModel(const CoptV2Options &O,
                                             const CoptV2Endpoint &Oz,
                                             const CoptV2Endpoint &Os,
                                             const CoptV2Endpoint &O2) {
  CoptV2Model M;

  const double BaseScale =
      Oz.TotalSizeCost > 0.0
          ? static_cast<double>(O.PolicyOzAnchorBytes) / Oz.TotalSizeCost
          : 1.0;

  if (O.PolicyOsBytes) {
    M.ScaleOs = coptV2SafeScale(
        static_cast<double>(*O.PolicyOsBytes) - O.PolicyOzAnchorBytes,
        Os.TotalSizeCost - Oz.TotalSizeCost, BaseScale);
  } else {
    M.ScaleOs = BaseScale;
  }

  if (O.PolicyO2Bytes) {
    M.ScaleO2 = coptV2SafeScale(
        static_cast<double>(*O.PolicyO2Bytes) - O.PolicyOzAnchorBytes,
        O2.TotalSizeCost - Oz.TotalSizeCost, BaseScale);
  } else {
    M.ScaleO2 = BaseScale;
  }

  std::unordered_set<std::string> Names;
  for (const auto &KV : Oz.Functions)
    Names.insert(KV.first);
  for (const auto &KV : Os.Functions)
    Names.insert(KV.first);
  for (const auto &KV : O2.Functions)
    Names.insert(KV.first);

  struct Temp {
    std::string Name;
    double Weight = 0.0;
  };
  std::vector<Temp> Profiled;
  for (const std::string &Name : Names) {
    const double W = std::max({coptV2GetFunctionScore(Oz, Name).ProfileWeight,
                               coptV2GetFunctionScore(Os, Name).ProfileWeight,
                               coptV2GetFunctionScore(O2, Name).ProfileWeight});
    if (W <= 0.0)
      continue;
    Profiled.push_back({Name, W});
    M.TotalProfileWeight += W;
  }

  if (M.TotalProfileWeight <= 0.0)
    report_fatal_error(
        "COPT-INTERNAL: shadow pipelines produced no usable SamplePGO/BFI "
        "function counts");

  M.Functions.reserve(Profiled.size());
  for (const Temp &T : Profiled) {
    const auto &Fz = coptV2GetFunctionScore(Oz, T.Name);
    const auto &Fs = coptV2GetFunctionScore(Os, T.Name);
    const auto &F2 = coptV2GetFunctionScore(O2, T.Name);
    const double H = T.Weight / M.TotalProfileWeight;

    CoptV2FunctionCandidate F;
    F.Name = T.Name;
    F.Hotness = H;
    F.Options[0] = {CoptV2Policy::Oz, 0.0, 0.0};
    F.Options[1] = {CoptV2Policy::Os,
                    (Fs.SizeCost - Fz.SizeCost) * M.ScaleOs,
                    H * O.OsUtilityFactor};
    F.Options[2] = {CoptV2Policy::O2,
                    (F2.SizeCost - Fz.SizeCost) * M.ScaleO2, H};
    M.Functions.push_back(std::move(F));
  }

  std::sort(M.Functions.begin(), M.Functions.end(),
            [](const CoptV2FunctionCandidate &A,
               const CoptV2FunctionCandidate &B) {
              if (A.Hotness != B.Hotness)
                return A.Hotness > B.Hotness;
              return A.Name < B.Name;
            });
  return M;
}

struct CoptV2Back {
  int PrevBucket = -1;
  unsigned Choice = 0;
};

static std::unordered_map<std::string, CoptV2Policy>
coptV2SolveFunctions(const CoptV2Model &M, double Headroom,
                     unsigned Buckets) {
  std::unordered_map<std::string, CoptV2Policy> Selected;
  if (M.Functions.empty())
    return Selected;

  std::vector<double> MinCost(M.Functions.size(), 0.0);
  double MinTotal = 0.0;
  for (size_t I = 0; I < M.Functions.size(); ++I) {
    MinCost[I] = std::min({M.Functions[I].Options[0].Cost,
                           M.Functions[I].Options[1].Cost,
                           M.Functions[I].Options[2].Cost});
    MinTotal += MinCost[I];
  }

  double ShiftedBudget = Headroom - MinTotal;
  if (ShiftedBudget < -1.0e-6) {
    // The model itself cannot fit its minimum-cost point. Keep the conservative
    // policy anchor; exact linked-size validation remains authoritative.
    for (const auto &F : M.Functions)
      Selected[F.Name] = CoptV2Policy::Oz;
    return Selected;
  }
  ShiftedBudget = std::max(0.0, ShiftedBudget);
  const double Width = ShiftedBudget > 0.0 ? ShiftedBudget / Buckets : 1.0;
  const double NegInf = -std::numeric_limits<double>::infinity();

  std::vector<double> DP(Buckets + 1, NegInf), Next(Buckets + 1, NegInf);
  DP[0] = 0.0;
  std::vector<std::vector<CoptV2Back>> Back(
      M.Functions.size(), std::vector<CoptV2Back>(Buckets + 1));

  for (size_t I = 0; I < M.Functions.size(); ++I) {
    std::fill(Next.begin(), Next.end(), NegInf);
    std::vector<CoptV2Back> Layer(Buckets + 1);

    for (unsigned B = 0; B <= Buckets; ++B) {
      if (!std::isfinite(DP[B]))
        continue;
      for (unsigned C = 0; C < 3; ++C) {
        const CoptV2Option &Opt = M.Functions[I].Options[C];
        const double Shifted = std::max(0.0, Opt.Cost - MinCost[I]);
        const unsigned W =
            Width > 0.0
                ? static_cast<unsigned>(std::ceil(Shifted / Width - 1.0e-12))
                : 0;
        if (W > Buckets - B)
          continue;
        const unsigned NB = B + W;
        const double NU = DP[B] + Opt.Utility;
        if (!std::isfinite(Next[NB]) || NU > Next[NB] + 1.0e-15) {
          Next[NB] = NU;
          Layer[NB] = {static_cast<int>(B), C};
        }
      }
    }

    // Pareto prune: increasing estimated size must strictly improve utility.
    double Best = NegInf;
    for (unsigned B = 0; B <= Buckets; ++B) {
      if (!std::isfinite(Next[B]))
        continue;
      if (Next[B] <= Best + 1.0e-15) {
        Next[B] = NegInf;
        Layer[B] = {};
      } else {
        Best = Next[B];
      }
    }

    DP.swap(Next);
    Back[I].swap(Layer);
  }

  int BestBucket = -1;
  double BestUtility = NegInf;
  for (unsigned B = 0; B <= Buckets; ++B) {
    if (!std::isfinite(DP[B]))
      continue;
    if (BestBucket < 0 || DP[B] > BestUtility + 1.0e-15) {
      BestBucket = static_cast<int>(B);
      BestUtility = DP[B];
    }
  }
  if (BestBucket < 0)
    report_fatal_error("COPT-INTERNAL: function DP produced no solution");

  int B = BestBucket;
  for (size_t RI = M.Functions.size(); RI-- > 0;) {
    const CoptV2Back &K = Back[RI][B];
    if (K.PrevBucket < 0)
      report_fatal_error("COPT-INTERNAL: broken function DP backpointer");
    Selected[M.Functions[RI].Name] =
        M.Functions[RI].Options[K.Choice].Policy;
    B = K.PrevBucket;
  }
  return Selected;
}

static double coptV2FunctionCost(
    const CoptV2Model &M,
    const std::unordered_map<std::string, CoptV2Policy> &Selected) {
  double Cost = 0.0;
  for (const auto &F : M.Functions) {
    auto It = Selected.find(F.Name);
    const CoptV2Policy P =
        It == Selected.end() ? CoptV2Policy::Oz : It->second;
    Cost += F.Options[static_cast<unsigned>(P)].Cost;
  }
  return Cost;
}

static double coptV2Aggressiveness(const CoptNativeTraceRecord &R) {
  if (R.Kind == "inline")
    return R.Outcome == "inline" ? 1.0 : 0.0;
  if (R.Kind == "unroll") {
    StringRef S(R.Outcome);
    if (S.starts_with("count=")) {
      double V = 0.0;
      if (!S.drop_front(6).getAsDouble(V))
        return std::max(0.0, V);
    }
    return std::max(0.0, R.MetricA);
  }
  return 0.0;
}

static double coptV2NativeInterest(const CoptNativeTraceRecord &R) {
  if (R.Kind == "inline") {
    if (R.Outcome != "inline")
      return 0.0;
    const double Threshold = std::max(1.0, std::abs(R.MetricB));
    const double Margin = R.MetricB - R.MetricA;
    return std::clamp(1.0 + Margin / Threshold, 0.5, 2.0);
  }
  if (R.Kind == "unroll") {
    const double Count = std::max(1.0, coptV2Aggressiveness(R));
    return std::clamp(std::log2(Count + 1.0), 0.5, 2.0);
  }
  return 0.5;
}

static double coptV2RegionUtilityWeight(const CoptNativeTraceRecord &R) {
  const double Native = coptV2NativeInterest(R);
  return R.ProfileCount > 0.0 ? R.ProfileCount * Native : Native;
}

static double coptV2RegionSizeWeight(const CoptNativeTraceRecord &R) {
  if (R.Kind == "unroll")
    return std::max(1.0, coptV2Aggressiveness(R) - 1.0);
  return 1.0;
}

using CoptV2TraceMap =
    std::unordered_map<std::string, CoptNativeTraceRecord>;

static std::string coptV2TraceKey(const CoptNativeTraceRecord &R) {
  return R.Kind + ":" + R.RegionID;
}

static CoptV2TraceMap
coptV2IndexTrace(ArrayRef<CoptNativeTraceRecord> Records) {
  CoptV2TraceMap Out;
  for (const CoptNativeTraceRecord &R0 : Records) {
    if (!(R0.Kind == "inline" || R0.Kind == "unroll") || R0.RegionID.empty())
      continue;
    CoptNativeTraceRecord R = R0;
    R.Function = coptV2CanonicalName(R.Function);
    const std::string K = coptV2TraceKey(R);
    auto It = Out.find(K);
    if (It == Out.end()) {
      Out.emplace(K, std::move(R));
      continue;
    }
    if (coptV2Aggressiveness(R) > coptV2Aggressiveness(It->second) + 1.0e-12) {
      R.ProfileCount = std::max(R.ProfileCount, It->second.ProfileCount);
      It->second = std::move(R);
    } else {
      It->second.ProfileCount = std::max(It->second.ProfileCount, R.ProfileCount);
    }
  }
  return Out;
}

struct CoptV2Region {
  CoptNativeTraceRecord Record;
  double UtilityWeight = 0.0;
  double SizeWeight = 1.0;
};

static std::vector<CoptV2Region> coptV2DifferentialRegions(
    StringRef FunctionName, const CoptV2Endpoint &Lower,
    const CoptV2Endpoint &Upper) {
  const CoptV2TraceMap L = coptV2IndexTrace(Lower.Trace);
  const CoptV2TraceMap U = coptV2IndexTrace(Upper.Trace);
  std::vector<CoptV2Region> Result;

  for (const auto &KV : U) {
    const CoptNativeTraceRecord &Hi0 = KV.second;
    if (Hi0.Function != FunctionName)
      continue;
    auto It = L.find(KV.first);
    if (It == L.end())
      continue;
    const CoptNativeTraceRecord &Lo = It->second;
    if (Lo.Function != Hi0.Function)
      continue;
    if (coptV2Aggressiveness(Hi0) <=
        coptV2Aggressiveness(Lo) + 1.0e-12)
      continue;
    if (Hi0.Kind == "inline" &&
        !(Lo.Outcome != "inline" && Hi0.Outcome == "inline"))
      continue;

    CoptNativeTraceRecord Hi = Hi0;
    Hi.ProfileCount = Lo.ProfileCount > 0.0 ? Lo.ProfileCount : Hi0.ProfileCount;
    CoptV2Region R;
    R.Record = std::move(Hi);
    R.UtilityWeight = coptV2RegionUtilityWeight(R.Record);
    R.SizeWeight = coptV2RegionSizeWeight(R.Record);
    Result.push_back(std::move(R));
  }

  std::sort(Result.begin(), Result.end(), [](const CoptV2Region &A,
                                             const CoptV2Region &B) {
    const double AD = A.UtilityWeight / std::max(1.0e-9, A.SizeWeight);
    const double BD = B.UtilityWeight / std::max(1.0e-9, B.SizeWeight);
    if (AD != BD)
      return AD > BD;
    if (A.UtilityWeight != B.UtilityWeight)
      return A.UtilityWeight > B.UtilityWeight;
    if (A.Record.Kind != B.Record.Kind)
      return A.Record.Kind < B.Record.Kind;
    return A.Record.RegionID < B.Record.RegionID;
  });
  return Result;
}

struct CoptV2RegionalAlternative {
  std::string Function;
  double Cost = 0.0;
  double Utility = 0.0;
  std::vector<CoptV2Region> Regions;
};

struct CoptV2RegionalGroup {
  std::string Function;
  std::vector<CoptV2RegionalAlternative> Alternatives;
};

static const CoptV2FunctionCandidate *
coptV2FindFunctionCandidate(const CoptV2Model &M, StringRef Name) {
  for (const auto &F : M.Functions)
    if (F.Name == Name)
      return &F;
  return nullptr;
}

static std::vector<unsigned> coptV2PrefixLengths(unsigned N,
                                                  unsigned MaxPlans) {
  std::vector<unsigned> L;
  if (!N || !MaxPlans)
    return L;
  if (N <= MaxPlans) {
    for (unsigned I = 1; I <= N; ++I)
      L.push_back(I);
    return L;
  }
  L.push_back(1);
  L.push_back(N);
  for (unsigned I = 0; I < MaxPlans; ++I) {
    unsigned K = static_cast<unsigned>(
        std::llround(static_cast<double>(I + 1) * N / MaxPlans));
    K = std::clamp(K, 1u, N);
    L.push_back(K);
  }
  std::sort(L.begin(), L.end());
  L.erase(std::unique(L.begin(), L.end()), L.end());
  return L;
}

static std::vector<CoptV2RegionalGroup> coptV2BuildRegionalGroups(
    const CoptV2Options &O, const CoptV2Model &M,
    const std::unordered_map<std::string, CoptV2Policy> &Selected,
    const CoptV2Endpoint &Current, const CoptV2Endpoint &O2,
    uint64_t &DifferentialTotal, uint64_t &DifferentialProfiled) {
  struct RankedFunction {
    double Density = 0.0;
    double Gain = 0.0;
    double Cost = 0.0;
    std::string Name;
    CoptV2Policy Base = CoptV2Policy::Oz;
    std::vector<CoptV2Region> Regions;
  };

  std::vector<RankedFunction> Ranked;
  DifferentialTotal = 0;
  DifferentialProfiled = 0;

  for (const auto &F : M.Functions) {
    auto It = Selected.find(F.Name);
    const CoptV2Policy Base =
        It == Selected.end() ? CoptV2Policy::Oz : It->second;
    if (Base == CoptV2Policy::O2)
      continue;

    // Match the external v1.5 refiner: the lower trace is the actual coarse
    // mixed-policy context, not the uniform Oz/Os endpoint. This preserves
    // cross-function/LTO interactions in regional differential discovery.
    auto Regions = coptV2DifferentialRegions(F.Name, Current, O2);
    DifferentialTotal += Regions.size();
    for (const auto &R : Regions)
      if (R.Record.ProfileCount > 0.0)
        ++DifferentialProfiled;
    if (Regions.empty())
      continue;

    const CoptV2Option &BaseOpt = F.Options[static_cast<unsigned>(Base)];
    const CoptV2Option &Upper = F.Options[2];
    const double Cost = Upper.Cost - BaseOpt.Cost;
    const double Gain = Upper.Utility - BaseOpt.Utility;
    if (Cost <= 1.0e-9 || Gain <= 1.0e-15)
      continue;
    Ranked.push_back({Gain / Cost, Gain, Cost, F.Name, Base,
                      std::move(Regions)});
  }

  std::sort(Ranked.begin(), Ranked.end(), [](const RankedFunction &A,
                                             const RankedFunction &B) {
    if (A.Density != B.Density)
      return A.Density > B.Density;
    if (A.Gain != B.Gain)
      return A.Gain > B.Gain;
    return A.Name < B.Name;
  });

  std::vector<CoptV2RegionalGroup> Groups;
  unsigned RegionBudgetUsed = 0;
  for (RankedFunction &RF : Ranked) {
    if (Groups.size() >= O.MaxRefinedFunctions)
      break;
    const unsigned Retained =
        std::min<unsigned>(RF.Regions.size(), O.MaxRegionsPerFunction);
    if (!Retained)
      continue;
    if (!Groups.empty() &&
        RegionBudgetUsed + Retained > O.MaxDifferentialRegions)
      continue;

    double TotalSizeWeight = 0.0;
    double TotalUtilityWeight = 0.0;
    for (const auto &R : RF.Regions) {
      TotalSizeWeight += R.SizeWeight;
      TotalUtilityWeight += R.UtilityWeight;
    }
    if (TotalSizeWeight <= 0.0 || TotalUtilityWeight <= 0.0)
      continue;

    CoptV2RegionalGroup G;
    G.Function = RF.Name;
    G.Alternatives.push_back({RF.Name, 0.0, 0.0, {}}); // frozen baseline

    const auto Prefixes =
        coptV2PrefixLengths(Retained, O.MaxPartialPlansPerFunction);
    for (unsigned K : Prefixes) {
      double SW = 0.0, UW = 0.0;
      std::vector<CoptV2Region> Rs;
      for (unsigned I = 0; I < K; ++I) {
        SW += RF.Regions[I].SizeWeight;
        UW += RF.Regions[I].UtilityWeight;
        Rs.push_back(RF.Regions[I]);
      }
      const double SizeFraction = std::min(0.98, SW / TotalSizeWeight);
      const double UtilityFraction = std::min(0.98, UW / TotalUtilityWeight);
      G.Alternatives.push_back(
          {RF.Name, RF.Cost * SizeFraction, RF.Gain * UtilityFraction,
           std::move(Rs)});
    }

    std::sort(G.Alternatives.begin(), G.Alternatives.end(),
              [](const CoptV2RegionalAlternative &A,
                 const CoptV2RegionalAlternative &B) {
                if (A.Cost != B.Cost)
                  return A.Cost < B.Cost;
                return A.Utility > B.Utility;
              });
    std::vector<CoptV2RegionalAlternative> Kept;
    double BestUtility = -std::numeric_limits<double>::infinity();
    for (auto &A : G.Alternatives) {
      if (A.Utility <= BestUtility + 1.0e-15)
        continue;
      BestUtility = A.Utility;
      Kept.push_back(std::move(A));
    }
    G.Alternatives = std::move(Kept);
    Groups.push_back(std::move(G));
    RegionBudgetUsed += Retained;
    if (RegionBudgetUsed >= O.MaxDifferentialRegions)
      break;
  }
  return Groups;
}

static std::vector<CoptV2RegionalAlternative>
coptV2SolveRegional(const std::vector<CoptV2RegionalGroup> &Groups,
                    double Budget, unsigned Buckets) {
  if (Groups.empty() || Budget <= 0.0)
    return {};
  const double Width = Budget > 0.0 ? Budget / Buckets : 1.0;
  const double NegInf = -std::numeric_limits<double>::infinity();
  std::vector<double> DP(Buckets + 1, NegInf), Next(Buckets + 1, NegInf);
  DP[0] = 0.0;
  std::vector<std::vector<CoptV2Back>> Back(
      Groups.size(), std::vector<CoptV2Back>(Buckets + 1));

  for (size_t I = 0; I < Groups.size(); ++I) {
    std::fill(Next.begin(), Next.end(), NegInf);
    std::vector<CoptV2Back> Layer(Buckets + 1);
    for (unsigned B = 0; B <= Buckets; ++B) {
      if (!std::isfinite(DP[B]))
        continue;
      const auto &G = Groups[I];
      for (unsigned C = 0; C < G.Alternatives.size(); ++C) {
        const auto &A = G.Alternatives[C];
        const unsigned W =
            static_cast<unsigned>(std::ceil(A.Cost / Width - 1.0e-12));
        if (W > Buckets - B)
          continue;
        const unsigned NB = B + W;
        const double NU = DP[B] + A.Utility;
        if (!std::isfinite(Next[NB]) || NU > Next[NB] + 1.0e-15) {
          Next[NB] = NU;
          Layer[NB] = {static_cast<int>(B), C};
        }
      }
    }
    double Best = NegInf;
    for (unsigned B = 0; B <= Buckets; ++B) {
      if (!std::isfinite(Next[B]))
        continue;
      if (Next[B] <= Best + 1.0e-15) {
        Next[B] = NegInf;
        Layer[B] = {};
      } else {
        Best = Next[B];
      }
    }
    DP.swap(Next);
    Back[I].swap(Layer);
  }

  int BestBucket = -1;
  double BestUtility = NegInf;
  for (unsigned B = 0; B <= Buckets; ++B) {
    if (!std::isfinite(DP[B]))
      continue;
    if (BestBucket < 0 || DP[B] > BestUtility + 1.0e-15) {
      BestBucket = static_cast<int>(B);
      BestUtility = DP[B];
    }
  }
  if (BestBucket < 0)
    return {};

  std::vector<CoptV2RegionalAlternative> Chosen;
  int B = BestBucket;
  for (size_t RI = Groups.size(); RI-- > 0;) {
    const CoptV2Back &K = Back[RI][B];
    if (K.PrevBucket < 0)
      report_fatal_error("COPT-INTERNAL: broken regional DP backpointer");
    Chosen.push_back(Groups[RI].Alternatives[K.Choice]);
    B = K.PrevBucket;
  }
  std::reverse(Chosen.begin(), Chosen.end());
  return Chosen;
}

static void coptV2SetAllPolicy(CoptV2Policy P) {
  CoptV2Context.Selected.clear();
  switch (P) {
  case CoptV2Policy::Oz:
    CoptV2Context.Mode = CoptV2PolicyMode::AllOz;
    break;
  case CoptV2Policy::Os:
    CoptV2Context.Mode = CoptV2PolicyMode::AllOs;
    break;
  case CoptV2Policy::O2:
    CoptV2Context.Mode = CoptV2PolicyMode::AllO2;
    break;
  }
}

static void coptV2SetSelectedPolicy(
    const std::unordered_map<std::string, CoptV2Policy> &Selected) {
  CoptV2Context.Mode = CoptV2PolicyMode::Selected;
  CoptV2Context.Selected = Selected;
}

static void coptV2ClearPolicyContext() {
  CoptV2Context.Mode = CoptV2PolicyMode::Off;
  CoptV2Context.Selected.clear();
}

static std::vector<CoptNativeRegionSelection>
coptV2LoadRegionPlan(StringRef Path) {
  std::vector<CoptNativeRegionSelection> Out;
  if (Path.empty())
    return Out;

  auto BufOrErr = MemoryBuffer::getFile(Path);
  if (!BufOrErr)
    report_fatal_error(Twine("COPT-INTERNAL: cannot open regional policy: ") +
                       Path);

  SmallVector<StringRef, 256> Lines;
  (*BufOrErr)->getBuffer().split(Lines, '\n', -1, false);
  std::unordered_set<std::string> Seen;
  unsigned LineNo = 0;
  for (StringRef Raw : Lines) {
    ++LineNo;
    StringRef Line = Raw.trim();
    if (Line.empty() || Line.starts_with("#"))
      continue;
    SmallVector<StringRef, 8> Fields;
    Line.split(Fields, '\t', -1, false);
    if (Fields.size() < 3)
      report_fatal_error(Twine("COPT-INTERNAL: malformed regional policy line ") +
                         Twine(LineNo) + " in " + Path);
    StringRef Kind = Fields[0].trim();
    StringRef ID = Fields[1].trim();
    StringRef Policy = Fields[2].trim();
    if (LineNo == 1 && Kind.equals_insensitive("kind") &&
        ID.equals_insensitive("region_id"))
      continue;
    if (!(Kind == "inline" || Kind == "unroll"))
      report_fatal_error(Twine("COPT-INTERNAL: unsupported regional kind '") +
                         Kind + "'");
    if (!Policy.equals_insensitive("o2"))
      report_fatal_error("COPT-INTERNAL: regional replay only supports O2");
    std::string Key = (Twine(Kind) + ":" + ID).str();
    if (!Seen.insert(Key).second)
      continue;
    Out.push_back({Kind.str(), ID.str()});
  }
  return Out;
}

static void coptV2WriteReport(
    const CoptV2Options &O, const CoptV2Endpoint &Oz,
    const CoptV2Endpoint &Os, const CoptV2Endpoint &O2,
    const CoptV2Model &M,
    const std::unordered_map<std::string, CoptV2Policy> &Selected,
    double CoarseEstimatedBytes, double ResidualBytes,
    uint64_t DifferentialTotal, uint64_t DifferentialProfiled,
    ArrayRef<CoptV2RegionalAlternative> Regional,
    ArrayRef<CoptNativeRegionSelection> SelectedRegions) {
  if (O.ReportPath.empty())
    return;

  json::Object Root;
  Root["version"] = 3;
  Root["algorithm"] = "internal-deferred-prelink-external-equivalent-policy";
  Root["baseline_bytes"] = static_cast<int64_t>(O.BaselineBytes);
  Root["requested_max_bytes"] = static_cast<int64_t>(O.RequestedMaxBytes);
  Root["effective_max_bytes"] = static_cast<int64_t>(O.EffectiveMaxBytes);
  Root["requested_growth_percent"] = O.RequestedGrowthPercent;
  Root["budget_scale"] = O.BudgetScale;
  Root["force_all_oz"] = O.ForceAllOz;
  Root["deferred_prelink_required"] = O.RequireDeferredPreLink;
  Root["function_model_path"] = O.FunctionModelPath;
  Root["fixed_policy_path"] = O.FixedPolicyPath;
  Root["region_file_path"] = O.RegionFilePath;
  Root["policy_oz_anchor_bytes"] = static_cast<int64_t>(O.PolicyOzAnchorBytes);
  Root["shadow_oz_size_cost"] = Oz.TotalSizeCost;
  Root["shadow_os_size_cost"] = Os.TotalSizeCost;
  Root["shadow_o2_size_cost"] = O2.TotalSizeCost;
  Root["scale_os"] = M.ScaleOs;
  Root["scale_o2"] = M.ScaleO2;
  Root["profiled_functions"] = static_cast<int64_t>(M.Functions.size());
  Root["coarse_estimated_bytes"] = CoarseEstimatedBytes;
  Root["regional_residual_estimated_bytes"] = ResidualBytes;
  Root["differential_regions_total"] = static_cast<int64_t>(DifferentialTotal);
  Root["differential_regions_with_profile_count"] =
      static_cast<int64_t>(DifferentialProfiled);
  Root["differential_regions_profile_fallback"] =
      static_cast<int64_t>(DifferentialTotal - DifferentialProfiled);
  Root["selected_region_count"] =
      static_cast<int64_t>(SelectedRegions.size());

  uint64_t NZ = 0, NS = 0, N2 = 0;
  json::Array FunctionPolicies;
  for (const auto &F : M.Functions) {
    auto It = Selected.find(F.Name);
    const CoptV2Policy P =
        It == Selected.end() ? CoptV2Policy::Oz : It->second;
    switch (P) {
    case CoptV2Policy::Oz:
      ++NZ;
      break;
    case CoptV2Policy::Os:
      ++NS;
      break;
    case CoptV2Policy::O2:
      ++N2;
      break;
    }
    json::Object J;
    J["function"] = F.Name;
    J["policy"] = coptV2PolicyName(P).str();
    J["hotness"] = F.Hotness;
    J["estimated_cost_bytes"] = F.Options[static_cast<unsigned>(P)].Cost;
    FunctionPolicies.push_back(std::move(J));
  }
  Root["functions_oz"] = static_cast<int64_t>(NZ);
  Root["functions_os"] = static_cast<int64_t>(NS);
  Root["functions_o2"] = static_cast<int64_t>(N2);
  Root["function_policies"] = std::move(FunctionPolicies);

  json::Array Regions;
  for (const auto &A : Regional) {
    for (const auto &R : A.Regions) {
      json::Object J;
      J["function"] = A.Function;
      J["kind"] = R.Record.Kind;
      J["region_id"] = R.Record.RegionID;
      J["subject"] = R.Record.Subject;
      J["profile_count"] = R.Record.ProfileCount;
      J["profile_guided"] = R.Record.ProfileCount > 0.0;
      J["native_interest"] = coptV2NativeInterest(R.Record);
      Regions.push_back(std::move(J));
    }
  }
  Root["selected_regions"] = std::move(Regions);

  std::error_code EC;
  raw_fd_ostream OS(O.ReportPath, EC, sys::fs::OF_Text);
  if (EC)
    report_fatal_error(Twine("COPT-INTERNAL: cannot write report ") +
                       O.ReportPath + ": " + EC.message());
  OS << formatv("{0:2}\n", json::Value(std::move(Root)));
}

static bool coptV2HasDeferredPreLinkMarker(const Module &M) {
  return M.getModuleFlag("COPTDeferredPreLink") != nullptr;
}

static bool coptV2HasRawPreLinkMarker(const Module &M) {
  return M.getModuleFlag("COPTRawPreLink") != nullptr;
}

static bool coptV2HasPerTUPreLinkMarker(const Module &M) {
  return M.getModuleFlag("COPTPerTUPreLinkDone") != nullptr;
}


static void coptV2WriteForcedEndpointReport(const CoptV2Options &O,
                                            CoptV2Policy P,
                                            const Module &Mod) {
  if (O.ReportPath.empty())
    return;

  json::Object Root;
  Root["version"] = 3;
  Root["mode"] = "forced-endpoint";
  Root["forced_policy"] = coptV2PolicyName(P).str();
  Root["module"] = Mod.getModuleIdentifier();
  Root["deferred_prelink_required"] = O.RequireDeferredPreLink;
  Root["deferred_prelink_marker"] = coptV2HasDeferredPreLinkMarker(Mod);
  Root["per_tu_prelink_marker"] = coptV2HasPerTUPreLinkMarker(Mod);

  std::error_code EC;
  raw_fd_ostream OS(O.ReportPath, EC, sys::fs::OF_Text);
  if (EC)
    report_fatal_error(Twine("COPT-INTERNAL: cannot write forced-endpoint report ") +
                       O.ReportPath + ": " + EC.message());
  OS << formatv("{0:2}\n", json::Value(std::move(Root)));
}

// Replay the normal O2 FullLTOPreLink pipeline inside the same LTO process.
// The frontend emitted only the minimal O0 LTO-prelink form, so policy attrs
// are now visible before both pre-link and post-link optimization.
static void coptV2RunDeferredPreLink(const Config &Conf, Module &Mod,
                                     TargetMachine *TM) {
  std::optional<PGOOptions> PGOOpt;
  if (!Conf.SampleProfile.empty())
    PGOOpt = PGOOptions(
        Conf.SampleProfile, "", Conf.ProfileRemapping,
        /*MemoryProfile=*/"", PGOOptions::SampleUse,
        PGOOptions::NoCSAction, PGOOptions::ColdFuncOpt::Default,
        /*DebugInfoForProfiling=*/false,
        /*PseudoProbeForProfiling=*/true);
  else if (Conf.AddFSDiscriminator)
    PGOOpt = PGOOptions("", "", "", /*MemoryProfile=*/"",
                        PGOOptions::NoAction, PGOOptions::NoCSAction,
                        PGOOptions::ColdFuncOpt::Default, true);

  TM->setPGOOption(PGOOpt);

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(Mod.getContext(), Conf.DebugPassManager,
                              Conf.VerifyEach);
  SI.registerCallbacks(PIC, &MAM);

  PassBuilder PB(TM, Conf.PTO, PGOOpt, &PIC);
  registerCoptInternalPolicyCallbacks(PB);
  RegisterPassPlugins(Conf.PassPlugins, PB);

  // LLVM 22 no longer exposes the vector-library selection through
  // TargetOptions::VecLib.  These COPT/SPEC builds do not request a custom
  // -fveclib, so use TargetLibraryInfoImpl's default NoLibrary setting.
  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      new TargetLibraryInfoImpl(TM->getTargetTriple()));
  if (Conf.Freestanding)
    TLII->disableAllFunctions();
  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  if (!Conf.AAPipeline.empty()) {
    AAManager AA;
    if (auto Err = PB.parseAAPipeline(AA, Conf.AAPipeline))
      report_fatal_error(Twine("COPT-INTERNAL: unable to parse AA pipeline: ") +
                         toString(std::move(Err)));
    FAM.registerPass([&] { return std::move(AA); });
  }

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  if (!Conf.DisableVerify)
    MPM.addPass(VerifierPass());
  MPM.addPass(PB.buildLTOPreLinkDefaultPipeline(OptimizationLevel::O2));
  if (!Conf.DisableVerify)
    MPM.addPass(VerifierPass());
  MPM.run(Mod, MAM);
}

// Optional exact fixed policy map used by the actual-linked-size correction
// loop. Missing functions default to Oz, matching the external policy driver.
static std::unordered_map<std::string, CoptV2Policy>
coptV2LoadFixedPolicy(StringRef Path) {
  std::unordered_map<std::string, CoptV2Policy> Selected;
  if (Path.empty())
    return Selected;

  auto MBOrErr = MemoryBuffer::getFile(Path);
  if (!MBOrErr)
    report_fatal_error(Twine("COPT-INTERNAL: cannot open fixed policy ") +
                       Path + ": " + MBOrErr.getError().message());

  SmallVector<StringRef, 0> Lines;
  (*MBOrErr)->getBuffer().split(Lines, '\n', -1, false);
  bool First = true;
  for (StringRef Line : Lines) {
    Line = Line.trim();
    if (Line.empty() || Line.starts_with("#"))
      continue;
    const char Sep = Line.contains('\t') ? '\t' : ',';
    auto Pair = Line.split(Sep);
    StringRef Name = Pair.first.trim();
    StringRef Policy = Pair.second.trim();
    if (First && Name.equals_insensitive("function") &&
        Policy.equals_insensitive("policy")) {
      First = false;
      continue;
    }
    First = false;
    if (Name.empty() || Policy.empty())
      report_fatal_error(Twine("COPT-INTERNAL: malformed fixed-policy row: ") +
                         Line);
    auto P = coptV2ParseOptionalPolicy(Policy);
    if (!P)
      report_fatal_error(Twine("COPT-INTERNAL: empty fixed policy for ") + Name);
    Selected[coptV2CanonicalName(Name)] = *P;
  }
  return Selected;
}

// Optional exact external-style model. The calibration utility writes the
// same function options used by copt_allocator.py:
// function<TAB>sample_count<TAB>hotness<TAB>cost_os<TAB>cost_o2
//         <TAB>utility_os<TAB>utility_o2
static std::optional<CoptV2Model>
coptV2LoadReferenceFunctionModel(const CoptV2Options &O) {
  if (O.FunctionModelPath.empty())
    return std::nullopt;

  auto MBOrErr = MemoryBuffer::getFile(O.FunctionModelPath);
  if (!MBOrErr)
    report_fatal_error(Twine("COPT-INTERNAL: cannot open function model ") +
                       O.FunctionModelPath + ": " +
                       MBOrErr.getError().message());

  CoptV2Model M;
  StringRef Text = (*MBOrErr)->getBuffer();
  SmallVector<StringRef, 0> Lines;
  Text.split(Lines, '\n', -1, false);

  bool HeaderSeen = false;
  for (StringRef Line : Lines) {
    Line = Line.trim();
    if (Line.empty() || Line.starts_with("#"))
      continue;
    SmallVector<StringRef, 8> C;
    Line.split(C, '\t');
    if (!HeaderSeen) {
      HeaderSeen = true;
      if (!C.empty() && C[0].equals_insensitive("function"))
        continue;
    }
    if (C.size() < 7)
      report_fatal_error(
          Twine("COPT-INTERNAL: malformed function model row: ") + Line);

    double Sample = 0.0, H = 0.0, CostOs = 0.0, CostO2 = 0.0;
    double UOs = 0.0, UO2 = 0.0;
    if (C[1].getAsDouble(Sample) || C[2].getAsDouble(H) ||
        C[3].getAsDouble(CostOs) || C[4].getAsDouble(CostO2) ||
        C[5].getAsDouble(UOs) || C[6].getAsDouble(UO2))
      report_fatal_error(
          Twine("COPT-INTERNAL: invalid numeric function model row: ") + Line);

    CoptV2FunctionCandidate F;
    F.Name = coptV2CanonicalName(C[0]);
    F.Hotness = H;
    F.Options[0] = {CoptV2Policy::Oz, 0.0, 0.0};
    F.Options[1] = {CoptV2Policy::Os, CostOs, UOs};
    F.Options[2] = {CoptV2Policy::O2, CostO2, UO2};
    M.Functions.push_back(std::move(F));
    M.TotalProfileWeight += Sample;
  }

  if (M.Functions.empty())
    report_fatal_error("COPT-INTERNAL: reference function model is empty");

  std::sort(M.Functions.begin(), M.Functions.end(),
            [](const CoptV2FunctionCandidate &A,
               const CoptV2FunctionCandidate &B) {
              if (A.Hotness != B.Hotness)
                return A.Hotness > B.Hotness;
              return A.Name < B.Name;
            });
  return M;
}

#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
static CoptV2Endpoint coptV2RunEndpoint(
    const Config &Conf, Module &Input, TargetMachine *TM,
    CoptV2Policy Policy, const DenseSet<StringRef> &BitcodeLibFuncs) {
#else
static CoptV2Endpoint coptV2RunEndpoint(const Config &Conf, Module &Input,
                                        TargetMachine *TM,
                                        CoptV2Policy Policy) {
#endif
  std::unique_ptr<Module> Shadow = CloneModule(Input);
  coptV2SetAllPolicy(Policy);
  coptV2ApplyActivePolicy(*Shadow);

  CoptNativePolicy &Native = CoptNativePolicy::get();
  Native.setInternalRegionPlan(ArrayRef<CoptNativeRegionSelection>()); // suppress any file-backed regional plan
  Native.beginInternalTrace();

  coptV2RunDeferredPreLink(Conf, *Shadow, TM);

#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
  runNewPMPassesOnce(Conf, *Shadow, TM, /*OptLevel=*/2,
                     /*IsThinLTO=*/false, /*ExportSummary=*/nullptr,
                     /*ImportSummary=*/nullptr, BitcodeLibFuncs);
#else
  runNewPMPassesOnce(Conf, *Shadow, TM, /*OptLevel=*/2,
                     /*IsThinLTO=*/false, /*ExportSummary=*/nullptr,
                     /*ImportSummary=*/nullptr);
#endif

  auto Trace = Native.endInternalTrace();
  Native.clearInternalRegionPlan();
  CoptV2Endpoint Result =
      coptV2ScoreModule(*Shadow, TM, Policy, std::move(Trace));
  coptV2ClearPolicyContext();
  return Result;
}

// Shadow the selected coarse mixed policy so regional discovery observes the
// same lower context as the external frozen-function refiner.
#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
static CoptV2Endpoint coptV2RunSelectedEndpoint(
    const Config &Conf, Module &Input, TargetMachine *TM,
    const std::unordered_map<std::string, CoptV2Policy> &Selected,
    const DenseSet<StringRef> &BitcodeLibFuncs) {
#else
static CoptV2Endpoint coptV2RunSelectedEndpoint(
    const Config &Conf, Module &Input, TargetMachine *TM,
    const std::unordered_map<std::string, CoptV2Policy> &Selected) {
#endif
  std::unique_ptr<Module> Shadow = CloneModule(Input);
  coptV2SetSelectedPolicy(Selected);
  coptV2ApplyActivePolicy(*Shadow);

  CoptNativePolicy &Native = CoptNativePolicy::get();
  Native.setInternalRegionPlan(ArrayRef<CoptNativeRegionSelection>());
  Native.beginInternalTrace();

  coptV2RunDeferredPreLink(Conf, *Shadow, TM);
#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
  runNewPMPassesOnce(Conf, *Shadow, TM, /*OptLevel=*/2,
                     /*IsThinLTO=*/false, /*ExportSummary=*/nullptr,
                     /*ImportSummary=*/nullptr, BitcodeLibFuncs);
#else
  runNewPMPassesOnce(Conf, *Shadow, TM, /*OptLevel=*/2,
                     /*IsThinLTO=*/false, /*ExportSummary=*/nullptr,
                     /*ImportSummary=*/nullptr);
#endif

  auto Trace = Native.endInternalTrace();
  Native.clearInternalRegionPlan();
  CoptV2Endpoint Result =
      coptV2ScoreModule(*Shadow, TM, CoptV2Policy::O2, std::move(Trace));
  coptV2ClearPolicyContext();
  return Result;
}

#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
static void runNewPMPasses(
    const Config &Conf, Module &Mod, TargetMachine *TM, unsigned OptLevel,
    bool IsThinLTO, ModuleSummaryIndex *ExportSummary,
    const ModuleSummaryIndex *ImportSummary,
    const DenseSet<StringRef> &BitcodeLibFuncs) {
#else
static void runNewPMPasses(const Config &Conf, Module &Mod, TargetMachine *TM,
                           unsigned OptLevel, bool IsThinLTO,
                           ModuleSummaryIndex *ExportSummary,
                           const ModuleSummaryIndex *ImportSummary) {
#endif
  const CoptV2Options O = coptV2LoadOptions();
  if (!O.Enabled) {
#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
    runNewPMPassesOnce(Conf, Mod, TM, OptLevel, IsThinLTO, ExportSummary,
                       ImportSummary, BitcodeLibFuncs);
#else
    runNewPMPassesOnce(Conf, Mod, TM, OptLevel, IsThinLTO, ExportSummary,
                       ImportSummary);
#endif
    return;
  }

  if (O.Log)
    errs() << "[COPT-INTERNAL] build-tag=" << CoptInternalBuildTag << "\n";

  // A global environment variable is convenient for build systems such as the
  // LLVM test-suite, but it also reaches helper LTO links (timeit, fpcmp,
  // etc.). Those links do not carry the benchmark SamplePGO profile and must
  // remain completely stock. The experiment runner additionally requires the
  // requested benchmark to emit a planner report, so a missing benchmark
  // profile cannot be mistaken for a successful COPT run.
  if (Conf.SampleProfile.empty()) {
    if (O.Log)
      errs() << "[COPT-INTERNAL] bypass unprofiled LTO module="
             << Mod.getModuleIdentifier() << "\n";
#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
    runNewPMPassesOnce(Conf, Mod, TM, OptLevel, IsThinLTO, ExportSummary,
                       ImportSummary, BitcodeLibFuncs);
#else
    runNewPMPassesOnce(Conf, Mod, TM, OptLevel, IsThinLTO, ExportSummary,
                       ImportSummary);
#endif
    return;
  }

  if (IsThinLTO)
    report_fatal_error("COPT-INTERNAL: v2 currently requires Full LTO (-flto)");
  if (OptLevel != 2)
    report_fatal_error(
        "COPT-INTERNAL: invoke the final COPT pipeline at O2; policy controls "
        "supply the Oz/Os/O2 heterogeneity internally");
  if (!Conf.OptPipeline.empty())
    report_fatal_error(
        "COPT-INTERNAL: custom -passes/-lto-newpm-passes pipelines are not "
        "supported by the v2 shadow planner");

  if (O.Log) {
    errs() << "[COPT-INTERNAL] options enabled=1 force_policy="
           << (O.ForcePolicy ? coptV2PolicyName(*O.ForcePolicy) : StringRef("<none>"))
           << " force_all_oz=" << (O.ForceAllOz ? 1 : 0)
           << " deferred_required=" << (O.RequireDeferredPreLink ? 1 : 0)
           << " per_tu_prelink=" << (O.PerTUPreLink ? 1 : 0)
           << " fixed_policy="
           << (O.FixedPolicyPath.empty() ? StringRef("<none>")
                                         : StringRef(O.FixedPolicyPath))
           << " report=" << (O.ReportPath.empty() ? StringRef("<none>")
                                                   : StringRef(O.ReportPath))
           << "\n";
  }

  if (O.RequireDeferredPreLink && !coptV2HasDeferredPreLinkMarker(Mod))
    report_fatal_error(
        "COPT-INTERNAL v3: input module was not produced with deferred "
        "pre-link optimization. Set COPT_INTERNAL_DEFER_PRELINK=1 for the "
        "frontend compile.");

  // Endpoint-equivalence/calibration mode. Produce an actual linked endpoint
  // while reusing the same deferred frontend bitcode objects.
  if (O.ForcePolicy) {
    if (O.Log)
      errs() << "[COPT-INTERNAL] forced endpoint policy="
             << coptV2PolicyName(*O.ForcePolicy) << "\n";
    coptV2SetAllPolicy(*O.ForcePolicy);
    coptV2ApplyActivePolicy(Mod);
    CoptNativePolicy::get().setInternalRegionPlan(
        ArrayRef<CoptNativeRegionSelection>());
    if (!O.PerTUPreLink)
      coptV2RunDeferredPreLink(Conf, Mod, TM);
    else if (O.Log)
      errs() << "[COPT-INTERNAL] combined prelink skipped: per-TU prelink already applied\n";
#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
    runNewPMPassesOnce(Conf, Mod, TM, /*OptLevel=*/2, /*IsThinLTO=*/false,
                       ExportSummary, ImportSummary, BitcodeLibFuncs);
#else
    runNewPMPassesOnce(Conf, Mod, TM, /*OptLevel=*/2, /*IsThinLTO=*/false,
                       ExportSummary, ImportSummary);
#endif
    CoptNativePolicy::get().clearInternalRegionPlan();
    coptV2WriteForcedEndpointReport(O, *O.ForcePolicy, Mod);
    coptV2ClearPolicyContext();
    return;
  }

  // Exact-correction escape hatch. This deliberately skips shadow search and
  // runs the real Full-LTO module with the most conservative function policy.
  // The outer validation harness may use this only after a model-selected
  // binary exceeded its hard linked-size limit.
  if (O.ForceAllOz) {
    if (O.Log)
      errs() << "[COPT-INTERNAL] exact-correction fallback: all-Oz policy\n";
    coptV2SetAllPolicy(CoptV2Policy::Oz);
    coptV2ApplyActivePolicy(Mod);
    CoptNativePolicy::get().setInternalRegionPlan(
        ArrayRef<CoptNativeRegionSelection>());
    if (!O.PerTUPreLink)
      coptV2RunDeferredPreLink(Conf, Mod, TM);
    else if (O.Log)
      errs() << "[COPT-INTERNAL] combined prelink skipped: per-TU prelink already applied\n";
#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
    runNewPMPassesOnce(Conf, Mod, TM, /*OptLevel=*/2, /*IsThinLTO=*/false,
                       ExportSummary, ImportSummary, BitcodeLibFuncs);
#else
    runNewPMPassesOnce(Conf, Mod, TM, /*OptLevel=*/2, /*IsThinLTO=*/false,
                       ExportSummary, ImportSummary);
#endif
    CoptNativePolicy::get().clearInternalRegionPlan();
    coptV2ClearPolicyContext();
    return;
  }

  if (O.Log) {
    errs() << "[COPT-INTERNAL] begin module=" << Mod.getModuleIdentifier()
           << " baseline=" << O.BaselineBytes
           << " requested-max=" << O.RequestedMaxBytes
           << " effective-max=" << O.EffectiveMaxBytes
           << " growth=" << O.RequestedGrowthPercent << "%\n";
  }

  // v4 replication mode: the exact endpoint-derived function model is already
  // available before the final link. The same deterministic MCKP assignment
  // has already been applied independently to every regular-LTO input module.
  // Recompute it here only to reassert policy for post-link and write a report.
  if (O.PerTUPreLink) {
    auto ReferenceModel = coptV2LoadReferenceFunctionModel(O);
    if (!ReferenceModel)
      report_fatal_error(
          "COPT-INTERNAL v4: per-TU final solve requires "
          "COPT_INTERNAL_FUNCTION_MODEL");

    CoptV2Model Model = std::move(*ReferenceModel);
    const double Headroom =
        static_cast<double>(O.EffectiveMaxBytes) - O.PolicyOzAnchorBytes;
    auto Selected = O.FixedPolicyPath.empty()
                        ? coptV2SolveFunctions(Model, Headroom, O.DPBuckets)
                        : coptV2LoadFixedPolicy(O.FixedPolicyPath);

    const double CoarseEstimatedBytes =
        static_cast<double>(O.PolicyOzAnchorBytes) +
        coptV2FunctionCost(Model, Selected);
    const double Residual =
        std::max(0.0, static_cast<double>(O.EffectiveMaxBytes) -
                          CoarseEstimatedBytes);

    uint64_t NZ = 0, NS = 0, N2 = 0;
    for (const auto &KV : Selected) {
      switch (KV.second) {
      case CoptV2Policy::Oz: ++NZ; break;
      case CoptV2Policy::Os: ++NS; break;
      case CoptV2Policy::O2: ++N2; break;
      }
    }

    if (O.Log && !O.FixedPolicyPath.empty())
      errs() << "[COPT-INTERNAL] fixed policy replay path="
             << O.FixedPolicyPath << "\n";

    if (O.Log)
      errs() << "[COPT-INTERNAL] per-tu coarse assignment "
             << "policies(oz/os/o2)=" << NZ << "/" << NS << "/" << N2
             << " coarse-est=" << CoarseEstimatedBytes
             << " residual-est=" << Residual << "\n";

    coptV2SetSelectedPolicy(Selected);
    coptV2ApplyActivePolicy(Mod);
    std::vector<CoptNativeRegionSelection> ReplayRegions =
        coptV2LoadRegionPlan(O.RegionFilePath);
    CoptNativePolicy::get().setInternalRegionPlan(ReplayRegions);
    if (O.Log && !O.RegionFilePath.empty())
      errs() << "[COPT-INTERNAL] regional replay path=" << O.RegionFilePath
             << " regions=" << ReplayRegions.size() << "\n";

#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
    runNewPMPassesOnce(Conf, Mod, TM, /*OptLevel=*/2,
                       /*IsThinLTO=*/false, ExportSummary, ImportSummary,
                       BitcodeLibFuncs);
#else
    runNewPMPassesOnce(Conf, Mod, TM, /*OptLevel=*/2,
                       /*IsThinLTO=*/false, ExportSummary, ImportSummary);
#endif

    CoptV2Endpoint EmptyOz, EmptyOs, EmptyO2;
    EmptyOz.Policy = CoptV2Policy::Oz;
    EmptyOs.Policy = CoptV2Policy::Os;
    EmptyO2.Policy = CoptV2Policy::O2;
    EmptyOz.TotalSizeCost = static_cast<double>(O.PolicyOzAnchorBytes);
    EmptyOs.TotalSizeCost =
        static_cast<double>(O.PolicyOsBytes.value_or(O.PolicyOzAnchorBytes));
    EmptyO2.TotalSizeCost =
        static_cast<double>(O.PolicyO2Bytes.value_or(O.PolicyOzAnchorBytes));
    std::vector<CoptV2RegionalAlternative> NoRegional;
    std::vector<CoptNativeRegionSelection> NoRegions;
    coptV2WriteReport(O, EmptyOz, EmptyOs, EmptyO2, Model, Selected,
                      CoarseEstimatedBytes, Residual,
                      /*DifferentialTotal=*/0,
                      /*DifferentialProfiled=*/0,
                      NoRegional, NoRegions);

    CoptNativePolicy::get().clearInternalRegionPlan();
    coptV2ClearPolicyContext();
    return;
  }

  // Endpoint discovery is entirely in-process. Each shadow starts from the
  // same pristine combined Full-LTO module; no code generation or linker
  // invocation is performed for a shadow.
#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
  CoptV2Endpoint Oz =
      coptV2RunEndpoint(Conf, Mod, TM, CoptV2Policy::Oz, BitcodeLibFuncs);
  CoptV2Endpoint Os =
      coptV2RunEndpoint(Conf, Mod, TM, CoptV2Policy::Os, BitcodeLibFuncs);
  CoptV2Endpoint O2 =
      coptV2RunEndpoint(Conf, Mod, TM, CoptV2Policy::O2, BitcodeLibFuncs);
#else
  CoptV2Endpoint Oz = coptV2RunEndpoint(Conf, Mod, TM, CoptV2Policy::Oz);
  CoptV2Endpoint Os = coptV2RunEndpoint(Conf, Mod, TM, CoptV2Policy::Os);
  CoptV2Endpoint O2 = coptV2RunEndpoint(Conf, Mod, TM, CoptV2Policy::O2);
#endif

  std::optional<CoptV2Model> ReferenceModel =
      coptV2LoadReferenceFunctionModel(O);
  CoptV2Model Model = ReferenceModel
                          ? std::move(*ReferenceModel)
                          : coptV2BuildFunctionModel(O, Oz, Os, O2);
  const double Headroom =
      static_cast<double>(O.EffectiveMaxBytes) - O.PolicyOzAnchorBytes;
  auto Selected = coptV2SolveFunctions(Model, Headroom, O.DPBuckets);

  // The external regional stage compares the actual coarse mixed-policy trace
  // against the all-O2 trace. Recreate that coarse context in-process.
#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
  CoptV2Endpoint Current = coptV2RunSelectedEndpoint(
      Conf, Mod, TM, Selected, BitcodeLibFuncs);
#else
  CoptV2Endpoint Current =
      coptV2RunSelectedEndpoint(Conf, Mod, TM, Selected);
#endif

  const double CoarseEstimatedBytes =
      static_cast<double>(O.PolicyOzAnchorBytes) +
      coptV2FunctionCost(Model, Selected);
  const double Residual =
      std::max(0.0, static_cast<double>(O.EffectiveMaxBytes) -
                        CoarseEstimatedBytes);
  const double BudgetSpan = std::max(
      0.0, static_cast<double>(O.EffectiveMaxBytes) - O.PolicyOzAnchorBytes);
  const double RegionalTrigger = std::max(
      static_cast<double>(O.RegionalMinResidualBytes),
      BudgetSpan * O.RegionalMinResidualRatio);

  uint64_t DifferentialTotal = 0;
  uint64_t DifferentialProfiled = 0;
  std::vector<CoptV2RegionalAlternative> RegionalChosen;
  std::vector<CoptNativeRegionSelection> SelectedRegions;

  if (Residual > RegionalTrigger && Residual > 0.0) {
    auto Groups = coptV2BuildRegionalGroups(
        O, Model, Selected, Current, O2, DifferentialTotal,
        DifferentialProfiled);
    RegionalChosen = coptV2SolveRegional(Groups, Residual, O.DPBuckets);
    for (const auto &A : RegionalChosen)
      for (const auto &R : A.Regions)
        SelectedRegions.push_back({R.Record.Kind, R.Record.RegionID});
  } else {
    // Still count differentials for diagnostics even when refinement is not
    // triggered by residual headroom.
    auto Ignored = coptV2BuildRegionalGroups(
        O, Model, Selected, Current, O2, DifferentialTotal,
        DifferentialProfiled);
    (void)Ignored;
  }

  if (O.Log) {
    uint64_t NZ = 0, NS = 0, N2 = 0;
    for (const auto &KV : Selected) {
      switch (KV.second) {
      case CoptV2Policy::Oz:
        ++NZ;
        break;
      case CoptV2Policy::Os:
        ++NS;
        break;
      case CoptV2Policy::O2:
        ++N2;
        break;
      }
    }
    errs() << "[COPT-INTERNAL] shadows oz=" << Oz.TotalSizeCost
           << " os=" << Os.TotalSizeCost << " o2=" << O2.TotalSizeCost
           << " profiled-functions=" << Model.Functions.size() << "\n";
    errs() << "[COPT-INTERNAL] coarse-est=" << CoarseEstimatedBytes
           << " residual-est=" << Residual << " policies(oz/os/o2)=" << NZ
           << "/" << NS << "/" << N2 << " regions=" << SelectedRegions.size()
           << " differentials=" << DifferentialTotal << "\n";
  }

  coptV2WriteReport(O, Oz, Os, O2, Model, Selected, CoarseEstimatedBytes,
                    Residual, DifferentialTotal, DifferentialProfiled,
                    RegionalChosen, SelectedRegions);

  // Stage boundary: the function policy is now frozen. The regional plan can
  // only relax selected inline/unroll decisions to native O2 behavior.
  coptV2SetSelectedPolicy(Selected);
  coptV2ApplyActivePolicy(Mod);
  CoptNativePolicy::get().setInternalRegionPlan(SelectedRegions);

  coptV2RunDeferredPreLink(Conf, Mod, TM);

#if COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
  runNewPMPassesOnce(Conf, Mod, TM, /*OptLevel=*/2, /*IsThinLTO=*/false,
                     ExportSummary, ImportSummary, BitcodeLibFuncs);
#else
  runNewPMPassesOnce(Conf, Mod, TM, /*OptLevel=*/2, /*IsThinLTO=*/false,
                     ExportSummary, ImportSummary);
#endif

  CoptNativePolicy::get().clearInternalRegionPlan();
  coptV2ClearPolicyContext();
}


//===----------------------------------------------------------------------===//
// COPT v4.1 pre-irsymtab regular-LTO input transformation.
//===----------------------------------------------------------------------===//
Expected<std::unique_ptr<MemoryBuffer>>
llvm::lto::coptPreprocessRegularLTOInput(MemoryBufferRef Object) {
  const CoptV2Options O = coptV2LoadOptions();
  if (!O.Enabled || !O.PerTUPreLink)
    return std::unique_ptr<MemoryBuffer>();

  LLVMContext Ctx;
  Expected<std::unique_ptr<Module>> MOrErr = parseBitcodeFile(Object, Ctx);
  if (!MOrErr)
    return MOrErr.takeError();

  std::unique_ptr<Module> Mod = std::move(*MOrErr);
  if (!coptV2HasDeferredPreLinkMarker(*Mod))
    return std::unique_ptr<MemoryBuffer>();

  if (!coptV2HasRawPreLinkMarker(*Mod))
    return createStringError(
        inconvertibleErrorCode(),
        "COPT-INTERNAL v4.2: deferred object was produced by the obsolete "
        "O0-prelink path; rebuild with a fresh v4.2 work root");

  if (Error Err = Mod->materializeAll())
    return std::move(Err);

  Config Conf;
  Conf.SampleProfile = coptV2EnvString("COPT_INTERNAL_SAMPLE_PROFILE");
  Conf.ProfileRemapping =
      coptV2EnvString("COPT_INTERNAL_PROFILE_REMAPPING");
  Conf.CPU = coptV2EnvString("COPT_INTERNAL_TARGET_CPU");

  // Match the ordinary clang -O2 tuning used by the external SPEC endpoints.
  Conf.PTO.LoopUnrolling = true;
  Conf.PTO.LoopInterleaving = true;
  Conf.PTO.LoopVectorization = true;
  Conf.PTO.SLPVectorization = true;
  Conf.PTO.LoopInterchange = false;
  Conf.PTO.LoopFusion = false;
  Conf.PTO.MergeFunctions = false;
  Conf.PTO.CallGraphProfile = true;
  Conf.PTO.UnifiedLTO = false;

  if (Conf.SampleProfile.empty())
    return createStringError(
        inconvertibleErrorCode(),
        "COPT-INTERNAL v4.2: COPT_INTERNAL_SAMPLE_PROFILE is required");

  std::string TargetError;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(Mod->getTargetTriple(), TargetError);
  if (!TheTarget)
    return make_error<StringError>(TargetError, inconvertibleErrorCode());

  std::unique_ptr<TargetMachine> TM =
      createTargetMachine(Conf, TheTarget, *Mod);

  if (O.ForcePolicy) {
    coptV2SetAllPolicy(*O.ForcePolicy);
  } else if (O.ForceAllOz) {
    coptV2SetAllPolicy(CoptV2Policy::Oz);
  } else if (!O.FixedPolicyPath.empty()) {
    coptV2SetSelectedPolicy(coptV2LoadFixedPolicy(O.FixedPolicyPath));
  } else {
    auto ReferenceModel = coptV2LoadReferenceFunctionModel(O);
    if (!ReferenceModel)
      return createStringError(
          inconvertibleErrorCode(),
          "COPT-INTERNAL v4.2: final mixed build requires "
          "COPT_INTERNAL_FUNCTION_MODEL");

    const double Headroom =
        static_cast<double>(O.EffectiveMaxBytes) - O.PolicyOzAnchorBytes;
    auto Selected =
        coptV2SolveFunctions(*ReferenceModel, Headroom, O.DPBuckets);
    coptV2SetSelectedPolicy(Selected);
  }

  coptV2ApplyActivePolicy(*Mod);

  if (O.Log) {
    uint64_t Oz = 0, Os = 0, O2 = 0;
    for (Function &F : *Mod) {
      if (F.isDeclaration())
        continue;
      if (F.hasFnAttribute(Attribute::MinSize))
        ++Oz;
      else if (F.hasFnAttribute(Attribute::OptimizeForSize))
        ++Os;
      else
        ++O2;
    }
    errs() << "[COPT-INTERNAL] per-tu prelink module="
           << Mod->getModuleIdentifier()
           << " stage=pre-irsymtab-raw"
           << " functions(oz/os/o2)=" << Oz << "/" << Os << "/" << O2
           << "\n";
  }

  std::vector<CoptNativeRegionSelection> ReplayRegions =
      coptV2LoadRegionPlan(O.RegionFilePath);
  CoptNativePolicy::get().setInternalRegionPlan(ReplayRegions);
  coptV2RunDeferredPreLink(Conf, *Mod, TM.get());
  CoptNativePolicy::get().clearInternalRegionPlan();

  Mod->addModuleFlag(Module::Override, "COPTPerTUPreLinkDone", uint32_t(1));

  ProfileSummaryInfo PSI(*Mod);
  ModuleSummaryIndex Index =
      buildModuleSummaryIndex(*Mod, nullptr, &PSI);

  SmallVector<char, 0> Bitcode;
  raw_svector_ostream OS(Bitcode);
  WriteBitcodeToFile(*Mod, OS,
                     /*ShouldPreserveUseListOrder=*/false, &Index);

  // Use LLVM's public owning-copy API. InputFile keeps this buffer alive.
  auto Result = MemoryBuffer::getMemBufferCopy(
      StringRef(Bitcode.data(), Bitcode.size()),
      Object.getBufferIdentifier());
  if (!Result)
    return createStringError(
        inconvertibleErrorCode(),
        "COPT-INTERNAL v4.2: unable to allocate rewritten bitcode buffer");

  coptV2ClearPolicyContext();
  return Result;
}
