//===-LTOBackend.cpp - LLVM Link Time Optimizer Backend -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the "backend" phase of LTO, i.e. it performs
// optimization and code generation on a loaded module. It is generally used
// internally by the LTO class but can also be used independently, for example
// to implement a standalone ThinLTO backend.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOBackend.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CGData/CodeGenData.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LTO/LTO.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ModuleSymbolTable.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/Analysis/ConstrainedOptimization.h" // copt_impl: ADDED
#include "llvm/Transforms/Utils/Cloning.h" // copt_impl: ADDED
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CoptFunctionPlans.h"
#include "llvm/Analysis/CoptSolver.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/IPO/Inliner.h"

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/LoopInstSimplify.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"

#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/IPO/Inliner.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <system_error>
#include <unordered_map>
#include <optional>
#include <unordered_set>

#include <algorithm>

using namespace llvm;
using namespace lto;

#define DEBUG_TYPE "lto-backend"

enum class LTOBitcodeEmbedding {
  DoNotEmbed = 0,
  EmbedOptimized = 1,
  EmbedPostMergePreOptimized = 2
};

static cl::opt<LTOBitcodeEmbedding> EmbedBitcode(
    "lto-embed-bitcode", cl::init(LTOBitcodeEmbedding::DoNotEmbed),
    cl::values(clEnumValN(LTOBitcodeEmbedding::DoNotEmbed, "none",
                          "Do not embed"),
               clEnumValN(LTOBitcodeEmbedding::EmbedOptimized, "optimized",
                          "Embed after all optimization passes"),
               clEnumValN(LTOBitcodeEmbedding::EmbedPostMergePreOptimized,
                          "post-merge-pre-opt",
                          "Embed post merge, but before optimizations")),
    cl::desc("Embed LLVM bitcode in object files produced by LTO"));

static cl::opt<bool> ThinLTOAssumeMerged(
    "thinlto-assume-merged", cl::init(false),
    cl::desc("Assume the input has already undergone ThinLTO function "
             "importing and the other pre-optimization pipeline changes."));

namespace llvm {
extern cl::opt<bool> NoPGOWarnMismatch;
}

[[noreturn]] static void reportOpenError(StringRef Path, Twine Msg) {
  errs() << "failed to open " << Path << ": " << Msg << '\n';
  errs().flush();
  exit(1);
}

Error Config::addSaveTemps(std::string OutputFileName, bool UseInputModulePath,
                           const DenseSet<StringRef> &SaveTempsArgs) {
  ShouldDiscardValueNames = false;

  std::error_code EC;
  if (SaveTempsArgs.empty() || SaveTempsArgs.contains("resolution")) {
    ResolutionFile =
        std::make_unique<raw_fd_ostream>(OutputFileName + "resolution.txt", EC,
                                         sys::fs::OpenFlags::OF_TextWithCRLF);
    if (EC) {
      ResolutionFile.reset();
      return errorCodeToError(EC);
    }
  }

  auto setHook = [&](std::string PathSuffix, ModuleHookFn &Hook) {
    // Keep track of the hook provided by the linker, which also needs to run.
    ModuleHookFn LinkerHook = Hook;
    Hook = [=](unsigned Task, const Module &M) {
      // If the linker's hook returned false, we need to pass that result
      // through.
      if (LinkerHook && !LinkerHook(Task, M))
        return false;

      std::string PathPrefix;
      // If this is the combined module (not a ThinLTO backend compile) or the
      // user hasn't requested using the input module's path, emit to a file
      // named from the provided OutputFileName with the Task ID appended.
      if (M.getModuleIdentifier() == "ld-temp.o" || !UseInputModulePath) {
        PathPrefix = OutputFileName;
        if (Task != (unsigned)-1)
          PathPrefix += utostr(Task) + ".";
      } else
        PathPrefix = M.getModuleIdentifier() + ".";
      std::string Path = PathPrefix + PathSuffix + ".bc";
      std::error_code EC;
      raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::OF_None);
      // Because -save-temps is a debugging feature, we report the error
      // directly and exit.
      if (EC)
        reportOpenError(Path, EC.message());
      WriteBitcodeToFile(M, OS, /*ShouldPreserveUseListOrder=*/false);
      return true;
    };
  };

  auto SaveCombinedIndex =
      [=](const ModuleSummaryIndex &Index,
          const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols) {
        std::string Path = OutputFileName + "index.bc";
        std::error_code EC;
        raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::OF_None);
        // Because -save-temps is a debugging feature, we report the error
        // directly and exit.
        if (EC)
          reportOpenError(Path, EC.message());
        writeIndexToFile(Index, OS);

        Path = OutputFileName + "index.dot";
        raw_fd_ostream OSDot(Path, EC, sys::fs::OpenFlags::OF_Text);
        if (EC)
          reportOpenError(Path, EC.message());
        Index.exportToDot(OSDot, GUIDPreservedSymbols);
        return true;
      };

  if (SaveTempsArgs.empty()) {
    setHook("0.preopt", PreOptModuleHook);
    setHook("1.promote", PostPromoteModuleHook);
    setHook("2.internalize", PostInternalizeModuleHook);
    setHook("3.import", PostImportModuleHook);
    setHook("4.opt", PostOptModuleHook);
    setHook("5.precodegen", PreCodeGenModuleHook);
    CombinedIndexHook = SaveCombinedIndex;
  } else {
    if (SaveTempsArgs.contains("preopt"))
      setHook("0.preopt", PreOptModuleHook);
    if (SaveTempsArgs.contains("promote"))
      setHook("1.promote", PostPromoteModuleHook);
    if (SaveTempsArgs.contains("internalize"))
      setHook("2.internalize", PostInternalizeModuleHook);
    if (SaveTempsArgs.contains("import"))
      setHook("3.import", PostImportModuleHook);
    if (SaveTempsArgs.contains("opt"))
      setHook("4.opt", PostOptModuleHook);
    if (SaveTempsArgs.contains("precodegen"))
      setHook("5.precodegen", PreCodeGenModuleHook);
    if (SaveTempsArgs.contains("combinedindex"))
      CombinedIndexHook = SaveCombinedIndex;
  }

  return Error::success();
}

#define HANDLE_EXTENSION(Ext)                                                  \
  llvm::PassPluginLibraryInfo get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"
#undef HANDLE_EXTENSION

static void RegisterPassPlugins(ArrayRef<std::string> PassPlugins,
                                PassBuilder &PB) {
#define HANDLE_EXTENSION(Ext)                                                  \
  get##Ext##PluginInfo().RegisterPassBuilderCallbacks(PB);
#include "llvm/Support/Extension.def"
#undef HANDLE_EXTENSION

  // Load requested pass plugins and let them register pass builder callbacks
  for (auto &PluginFN : PassPlugins) {
    auto PassPlugin = PassPlugin::Load(PluginFN);
    if (!PassPlugin)
      reportFatalUsageError(PassPlugin.takeError());
    PassPlugin->registerPassBuilderCallbacks(PB);
  }
}

static std::unique_ptr<TargetMachine>
createTargetMachine(const Config &Conf, const Target *TheTarget, Module &M) {
  const Triple &TheTriple = M.getTargetTriple();
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(TheTriple);
  for (const std::string &A : Conf.MAttrs)
    Features.AddFeature(A);

  std::optional<Reloc::Model> RelocModel;
  if (Conf.RelocModel)
    RelocModel = *Conf.RelocModel;
  else if (M.getModuleFlag("PIC Level"))
    RelocModel =
        M.getPICLevel() == PICLevel::NotPIC ? Reloc::Static : Reloc::PIC_;

  std::optional<CodeModel::Model> CodeModel;
  if (Conf.CodeModel)
    CodeModel = *Conf.CodeModel;
  else
    CodeModel = M.getCodeModel();

  TargetOptions TargetOpts = Conf.Options;
  if (TargetOpts.MCOptions.ABIName.empty()) {
    TargetOpts.MCOptions.ABIName = M.getTargetABIFromMD();
  }

  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      TheTriple, Conf.CPU, Features.getString(), TargetOpts, RelocModel,
      CodeModel, Conf.CGOptLevel));

  assert(TM && "Failed to create target machine");

  if (std::optional<uint64_t> LargeDataThreshold = M.getLargeDataThreshold())
    TM->setLargeDataThreshold(*LargeDataThreshold);

  return TM;
}

// COPT_INTERNAL_LTO_V2: BEGIN
static void registerCoptInternalPolicyCallbacks(PassBuilder &PB);
// COPT_INTERNAL_LTO_V2: END forward declaration

static void runNewPMPassesOnce(const Config &Conf, Module &Mod, TargetMachine *TM,
                           unsigned OptLevel, bool IsThinLTO,
                           ModuleSummaryIndex *ExportSummary,
                           const ModuleSummaryIndex *ImportSummary) {
  std::optional<PGOOptions> PGOOpt;
  if (!Conf.SampleProfile.empty())
    PGOOpt = PGOOptions(Conf.SampleProfile, "", Conf.ProfileRemapping,
                        /*MemoryProfile=*/"", PGOOptions::SampleUse,
                        PGOOptions::NoCSAction,
                        PGOOptions::ColdFuncOpt::Default, true);
  else if (Conf.RunCSIRInstr) {
    PGOOpt = PGOOptions("", Conf.CSIRProfile, Conf.ProfileRemapping,
                        /*MemoryProfile=*/"", PGOOptions::IRUse,
                        PGOOptions::CSIRInstr, PGOOptions::ColdFuncOpt::Default,
                        Conf.AddFSDiscriminator);
  } else if (!Conf.CSIRProfile.empty()) {
    PGOOpt =
        PGOOptions(Conf.CSIRProfile, "", Conf.ProfileRemapping,
                   /*MemoryProfile=*/"", PGOOptions::IRUse, PGOOptions::CSIRUse,
                   PGOOptions::ColdFuncOpt::Default, Conf.AddFSDiscriminator);
    NoPGOWarnMismatch = !Conf.PGOWarnMismatch;
  } else if (Conf.AddFSDiscriminator) {
    PGOOpt = PGOOptions("", "", "", /*MemoryProfile=*/"", PGOOptions::NoAction,
                        PGOOptions::NoCSAction,
                        PGOOptions::ColdFuncOpt::Default, true);
  }
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

  // COPT internal policy must register before external pass plugins so its
  // function attributes are available at Full-LTO extension points.
  registerCoptInternalPolicyCallbacks(PB);

  RegisterPassPlugins(Conf.PassPlugins, PB);

  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      new TargetLibraryInfoImpl(TM->getTargetTriple()));
  if (Conf.Freestanding)
    TLII->disableAllFunctions();
  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  // Parse a custom AA pipeline if asked to.
  if (!Conf.AAPipeline.empty()) {
    AAManager AA;
    if (auto Err = PB.parseAAPipeline(AA, Conf.AAPipeline)) {
      report_fatal_error(Twine("unable to parse AA pipeline description '") +
                         Conf.AAPipeline + "': " + toString(std::move(Err)));
    }
    // Register the AA manager first so that our version is the one used.
    FAM.registerPass([&] { return std::move(AA); });
  }

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;

  if (!Conf.DisableVerify)
    MPM.addPass(VerifierPass());

  OptimizationLevel OL;

  switch (OptLevel) {
  default:
    llvm_unreachable("Invalid optimization level");
  case 0:
    OL = OptimizationLevel::O0;
    break;
  case 1:
    OL = OptimizationLevel::O1;
    break;
  case 2:
    OL = OptimizationLevel::O2;
    break;
  case 3:
    OL = OptimizationLevel::O3;
    break;
  }

  // Parse a custom pipeline if asked to.
  if (!Conf.OptPipeline.empty()) {
    if (auto Err = PB.parsePassPipeline(MPM, Conf.OptPipeline)) {
      report_fatal_error(Twine("unable to parse pass pipeline description '") +
                         Conf.OptPipeline + "': " + toString(std::move(Err)));
    }
  } else if (IsThinLTO) {
    MPM.addPass(PB.buildThinLTODefaultPipeline(OL, ImportSummary));
  } else {
    MPM.addPass(PB.buildLTODefaultPipeline(OL, ExportSummary));
  }

  if (!Conf.DisableVerify)
    MPM.addPass(VerifierPass());

  if (PrintPipelinePasses) {
    std::string PipelineStr;
    raw_string_ostream OS(PipelineStr);
    MPM.printPipeline(OS, [&PIC](StringRef ClassName) {
      auto PassName = PIC.getPassNameForClassName(ClassName);
      return PassName.empty() ? ClassName : PassName;
    });
    outs() << "pipeline-passes: " << PipelineStr << '\n';
  }

  MPM.run(Mod, MAM);
}

// COPT_INTERNAL_LTO_V2: internal shadow planner wrapper.
#define COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS 0
#include "CoptInternalLTO.inc"
#undef COPT_INTERNAL_V2_HAS_BITCODE_LIBFUNCS
// COPT_INTERNAL_LTO_V2: END


// COPT_PROFILE_UTILITY_V2
//
// IMPORTANT:
//   SizeCost and PerfCost below are diagnostic/static-model quantities.
//   PerfCost is NOT used as the global optimization objective anymore.
//
// Global value is driven by profile hotness:
//   utility(plan) = normalized function entry hotness * plan aggressiveness.
//
// The hard size constraint remains the real linked-byte check in LLD.
// TTI CodeSize remains only a cheap internal size estimator used to shortlist
// and refine candidates before the authoritative linked-size check.
struct CoptFunctionScore {
  double SizeCost = 0.0;

  // Retained only for diagnostics/evaluation reports.
  double PerfCost = 0.0;

  // SamplePGO/BFI-derived function-entry execution importance.
  double Hotness = 0.0;

  uint64_t InstructionCount = 0;
};

struct CoptModuleScore {
  double SizeCost = 0.0;

  // Diagnostic only.
  double PerfCost = 0.0;

  // Sum of per-function entry hotness. Used to normalize values to [0, 1].
  double TotalHotness = 0.0;

  uint64_t InstructionCount = 0;
  uint64_t ProfiledBlocks = 0;
  uint64_t FallbackBlocks = 0;

  std::unordered_map<
      std::string,
      CoptFunctionScore>
      Functions;
};

static CoptModuleScore
scoreCoptModule(
    Module &M,
    TargetMachine *TM) {

  PassBuilder PB(TM);

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);

  PB.crossRegisterProxies(
      LAM,
      FAM,
      CGAM,
      MAM);

  CoptModuleScore Result;

  for (Function &F : M) {
    if (F.isDeclaration() ||
        F.empty())
      continue;

    auto &TTI =
        FAM.getResult<TargetIRAnalysis>(F);

    auto &BFI =
        FAM.getResult<BlockFrequencyAnalysis>(F);

    CoptFunctionScore &FunctionScore =
        Result.Functions[
            F.getName().str()];

    // --------------------------------------------------------------------
    // Profile hotness
    // --------------------------------------------------------------------
    //
    // Prefer the SamplePGO/synthetic profile count of the function entry.
    // This answers "how dynamically important is this function?" separately
    // from LLVM's static instruction cost estimates.
    const BasicBlock &Entry =
        F.getEntryBlock();

    const uint64_t EntryFreq =
        std::max<uint64_t>(
            1,
            BFI.getBlockFreq(
                   &Entry)
                .getFrequency());

    double EntryCountScale =
        1.0;

    if (std::optional<uint64_t> EntryCount =
            BFI.getBlockProfileCount(
                &Entry,
                /*AllowSynthetic=*/true))
      EntryCountScale =
          std::max(
              1.0,
              static_cast<double>(
                  *EntryCount));

    FunctionScore.Hotness =
        EntryCountScale;

    Result.TotalHotness +=
        EntryCountScale;

    // --------------------------------------------------------------------
    // Diagnostic TTI totals + cheap size estimator
    // --------------------------------------------------------------------
    //
    // TCK_CodeSize is only a shortlist/refinement estimate.
    // TCK_RecipThroughput is retained only for diagnostics.
    for (BasicBlock &BB : F) {
      double Weight = 0.0;

      if (std::optional<uint64_t> Count =
              BFI.getBlockProfileCount(
                  &BB,
                  /*AllowSynthetic=*/true)) {

        Weight =
            static_cast<double>(
                *Count);

        ++Result.ProfiledBlocks;
      } else {

        const uint64_t BBFreq =
            BFI.getBlockFreq(
                   &BB)
                .getFrequency();

        const double RelativeFreq =
            static_cast<double>(
                BBFreq) /
            static_cast<double>(
                EntryFreq);

        Weight =
            EntryCountScale *
            RelativeFreq;

        ++Result.FallbackBlocks;
      }

      for (Instruction &I : BB) {
        ++Result.InstructionCount;
        ++FunctionScore.InstructionCount;

        InstructionCost Size =
            TTI.getInstructionCost(
                &I,
                TargetTransformInfo::TCK_CodeSize);

        const double SizeValue =
            Size.isValid()
                ? static_cast<double>(
                      Size.getValue())
                : 1.0;

        Result.SizeCost +=
            SizeValue;

        FunctionScore.SizeCost +=
            SizeValue;

        InstructionCost Perf =
            TTI.getInstructionCost(
                &I,
                TargetTransformInfo::TCK_RecipThroughput);

        const double PerfValue =
            Perf.isValid()
                ? static_cast<double>(
                      Perf.getValue())
                : 1.0;

        const double WeightedPerf =
            PerfValue *
            Weight;

        Result.PerfCost +=
            WeightedPerf;

        FunctionScore.PerfCost +=
            WeightedPerf;
      }
    }
  }

  return Result;
}


static const CoptFunctionScore &
getCoptFunctionScoreOrZero(
    const CoptModuleScore &Score,
    StringRef FunctionName) {

  static const CoptFunctionScore Zero;

  auto It =
      Score.Functions.find(
          FunctionName.str());

  if (It == Score.Functions.end())
    return Zero;

  return It->second;
}

using CoptCandidateIndex =
    std::unordered_map<
        std::string,
        const CoptCandidate *>;

static CoptCandidateIndex
buildCoptCandidateIndex(
    ArrayRef<CoptCandidate> Candidates) {

  CoptCandidateIndex Index;

  Index.reserve(
      Candidates.size());

  for (const CoptCandidate &C :
       Candidates)
    Index.emplace(
        C.ID,
        &C);

  return Index;
}

static const CoptCandidate *
findCoptCandidateByID(
    const CoptCandidateIndex &Index,
    const std::string &ID) {

  auto It =
      Index.find(ID);

  if (It ==
      Index.end())
    return nullptr;

  return It->second;
}

static const CoptAlternative *
findCoptStateByName(
    const CoptCandidate &C,
    StringRef Name) {

  for (const CoptAlternative &A :
       C.Alternatives)
    if (A.Name == Name)
      return &A;

  return nullptr;
}

static double
coptDecisionAggressiveness(
    const CoptCandidate &C,
    StringRef StateName) {

  const CoptAlternative *Selected =
      findCoptStateByName(
          C,
          StateName);

  if (!Selected)
    return 0.0;

  if (Selected->LowerEndpoint)
    return 0.0;

  if (Selected->UpperEndpoint)
    return 1.0;

  const CoptAlternative *Lower =
      nullptr;

  const CoptAlternative *Upper =
      nullptr;

  for (const CoptAlternative &A :
       C.Alternatives) {

    if (!A.Replayable)
      continue;

    if (A.LowerEndpoint)
      Lower = &A;

    if (A.UpperEndpoint)
      Upper = &A;
  }

  if (!Lower || !Upper)
    return 0.0;

  auto MetricFraction =
      [&](StringRef Metric)
          -> std::optional<double> {

    auto LI =
        Lower->Metrics.find(
            Metric.str());

    auto UI =
        Upper->Metrics.find(
            Metric.str());

    auto SI =
        Selected->Metrics.find(
            Metric.str());

    if (LI == Lower->Metrics.end() ||
        UI == Upper->Metrics.end() ||
        SI == Selected->Metrics.end())
      return std::nullopt;

    const double Den =
        UI->second -
        LI->second;

    if (std::abs(Den) < 1.0e-12)
      return std::nullopt;

    return std::clamp(
        (SI->second -
         LI->second) /
            Den,
        0.0,
        1.0);
  };

  if (auto F =
          MetricFraction(
              "unroll_count"))
    return *F;

  if (auto F =
          MetricFraction(
              "vf"))
    return *F;

  std::vector<const CoptAlternative *>
      Replayable;

  for (const CoptAlternative &A :
       C.Alternatives)
    if (A.Replayable)
      Replayable.push_back(&A);

  if (Replayable.size() < 2)
    return 0.0;

  size_t LowerIndex =
      0;

  size_t UpperIndex =
      Replayable.size() - 1;

  size_t SelectedIndex =
      0;

  for (size_t I = 0;
       I < Replayable.size();
       ++I) {

    if (Replayable[I]->LowerEndpoint)
      LowerIndex = I;

    if (Replayable[I]->UpperEndpoint)
      UpperIndex = I;

    if (Replayable[I]->Name ==
        StateName)
      SelectedIndex = I;
  }

  if (UpperIndex == LowerIndex)
    return 0.0;

  return std::clamp(
      static_cast<double>(
          SelectedIndex - LowerIndex) /
          static_cast<double>(
              UpperIndex - LowerIndex),
      0.0,
      1.0);
}

static double
coptPlanAggressiveness(
    const CoptFunctionPlan &Plan,
    const CoptCandidateIndex &CandidateByID) {

  if (Plan.IsLowerEndpoint)
    return 0.0;

  if (Plan.IsUpperEndpoint)
    return 1.0;

  if (Plan.Choices.empty())
    return 0.0;

  double Sum = 0.0;
  unsigned Count = 0;

  for (const CoptPlanChoice &Choice :
       Plan.Choices) {

    const CoptCandidate *C =
        findCoptCandidateByID(
            CandidateByID,
            Choice.DecisionID);

    if (!C)
      continue;

    Sum +=
        coptDecisionAggressiveness(
            *C,
            Choice.State);

    ++Count;
  }

  if (Count == 0)
    return 0.0;

  return std::clamp(
      Sum /
          static_cast<double>(Count),
      0.0,
      1.0);
}

static std::vector<CoptPlanValue>
buildInitialCoptPlanValues(
    ArrayRef<CoptFunctionPlan> Plans,
    ArrayRef<CoptCandidate> Candidates,
    const CoptModuleScore &Lower,
    const CoptModuleScore &Upper) {

  std::vector<CoptPlanValue> Values;
  Values.reserve(Plans.size());

  const CoptCandidateIndex CandidateByID =
      buildCoptCandidateIndex(
          Candidates);

  const double ModuleHotness =
      std::max(
          Lower.TotalHotness,
          1.0);

  for (const CoptFunctionPlan &Plan :
       Plans) {

    CoptPlanValue V;

    V.Function =
        Plan.Function;

    V.PlanID =
        Plan.ID;

    V.LowerEndpoint =
        Plan.IsLowerEndpoint;

    V.UpperEndpoint =
        Plan.IsUpperEndpoint;

    if (Plan.IsLowerEndpoint) {
      V.SizeDelta = 0.0;
      V.PerformanceBenefit = 0.0;
      V.Exact = true;

      Values.push_back(
          std::move(V));

      continue;
    }

    const CoptFunctionScore &LowerF =
        getCoptFunctionScoreOrZero(
            Lower,
            Plan.Function);

    const CoptFunctionScore &UpperF =
        getCoptFunctionScoreOrZero(
            Upper,
            Plan.Function);

    const double UpperSizeDelta =
        UpperF.SizeCost -
        LowerF.SizeCost;

    const double Alpha =
        coptPlanAggressiveness(
            Plan,
            CandidateByID);

    // Cost-model v2:
    // LLVM's upper/reference pipeline supplies the native aggressive states.
    // COPT values those states by where the dynamic execution actually is,
    // instead of translating IR reciprocal-throughput sums into "speed".
    //
    // U(plan) =
    //   normalized function-entry SamplePGO/BFI hotness
    //   * native-plan aggressiveness.
    const double FunctionHotness =
        std::max(
            LowerF.Hotness,
            0.0) /
        ModuleHotness;

    V.PerformanceBenefit =
        FunctionHotness *
        Alpha;

    // Cheap size estimate for exploration. Exact probes refine this static
    // estimate; the final LLD linked-byte check remains authoritative.
    V.SizeDelta =
        Alpha *
        UpperSizeDelta;

    V.Exact = false;

    Values.push_back(
        std::move(V));
  }

  return Values;
}

static CoptPlanValue *
findCoptPlanValue(
    std::vector<CoptPlanValue> &Values,
    StringRef Function,
    StringRef PlanID) {

  for (CoptPlanValue &V :
       Values)
    if (V.Function == Function &&
        V.PlanID == PlanID)
      return &V;

  return nullptr;
}

static std::optional<double>
getCoptDoubleEnv(
    const char *Name) {

  const char *Raw =
      std::getenv(Name);

  if (!Raw)
    return std::nullopt;

  errno = 0;

  char *End = nullptr;

  double Value =
      std::strtod(
          Raw,
          &End);

  if (errno != 0 ||
      End == Raw ||
      *End != '\0' ||
      !std::isfinite(Value))
    report_fatal_error(
        Twine("COPT: invalid floating-point value for ") +
        Name);

  return Value;
}

static unsigned
getCoptUnsignedEnv(
    const char *Name,
    unsigned DefaultValue) {

  const char *Raw =
      std::getenv(Name);

  if (!Raw)
    return DefaultValue;

  StringRef S(Raw);

  unsigned Value = 0;

  if (S.getAsInteger(
          10,
          Value))
    report_fatal_error(
        Twine("COPT: invalid unsigned integer for ") +
        Name);

  return Value;
}

static std::optional<uint64_t>
getCoptUInt64Env(
    const char *Name) {

  const char *Raw =
      std::getenv(Name);

  if (!Raw)
    return std::nullopt;

  StringRef S(Raw);

  uint64_t Value = 0;

  if (S.getAsInteger(
          10,
          Value))
    report_fatal_error(
        Twine("COPT: invalid unsigned 64-bit integer for ") +
        Name);

  return Value;
}


// copt_impl: BEGIN
// ============================================================================
// COPT Oz-base -> selective local O3 delta helpers
//
// Replace the previous COPT helper block beginning with:
//
//   using CoptFunctionSet =
//
// and ending at the closing brace of:
//
//   runCoptO3UpgradePipeline(...)
//
// with this entire block.
//
// This version intentionally DOES NOT call:
//   PB.buildModuleSimplificationPipeline(...)
//   PB.buildModuleOptimizationPipeline(...)
//
// Function specialization is temporarily excluded from the delta pipeline.
// The active target-delta transformations in this correctness-first version are:
//   * selective O3 inlining
//   * non-trivial loop unswitching
//   * loop vectorization
//   * loop unrolling
//   * local cleanup
// ============================================================================

using CoptFunctionSet =
    std::unordered_set<std::string>;

static CoptFunctionSet
collectAllCoptDefinedFunctions(
    Module &M) {

  CoptFunctionSet Result;

  for (Function &F : M) {
    if (F.isDeclaration() ||
        F.empty())
      continue;

    Result.insert(
        F.getName().str());
  }

  return Result;
}

static CoptFunctionSet
collectCoptUpgradeFunctions(
    ArrayRef<CoptCandidate> Candidates,
    const std::unordered_map<
        std::string,
        std::string> &Choices) {

  CoptFunctionSet Result;

  for (const CoptCandidate &Candidate :
       Candidates) {

    auto ChoiceIt =
        Choices.find(
            Candidate.ID);

    // Missing means conservative/lower.  This is important when a replay
    // pipeline exposes a decision that was not part of the discovery plan.
    if (ChoiceIt ==
        Choices.end())
      continue;

    const CoptAlternative *Chosen =
        nullptr;

    for (const CoptAlternative &Alt :
         Candidate.Alternatives) {
      if (Alt.Name ==
          ChoiceIt->second) {
        Chosen = &Alt;
        break;
      }
    }

    if (!Chosen)
      report_fatal_error(
          Twine(
              "COPT: plan requests unknown state '") +
          ChoiceIt->second +
          "' for candidate " +
          Candidate.ID);

    if (!Chosen->LowerEndpoint)
      Result.insert(
          Candidate.Function);
  }

  return Result;
}

static void
prepareCoptSelectedFunctionsForO3(
    Module &M,
    const CoptFunctionSet &Selected) {

  for (const std::string &Name :
       Selected) {

    Function *F =
        M.getFunction(Name);

    if (!F ||
        F->isDeclaration())
      continue;

    // The module was produced by -Oz.  Selected functions are being promoted
    // toward the O3 target policy, so remove only the compiler's size-policy
    // attributes.  Preserve semantic/user attributes such as noinline/optnone.
    F->removeFnAttr(
        Attribute::OptimizeForSize);

    F->removeFnAttr(
        Attribute::MinSize);
  }
}

static unsigned
blockCoptInliningFromUnselectedCallers(
    Module &M,
    const CoptFunctionSet &Selected) {

  static constexpr StringLiteral
      TempNoInlineMD =
          "copt.ozbase.temp.noinline";

  unsigned Added = 0;

  MDNode *Tag =
      MDNode::get(
          M.getContext(),
          {});

  for (Function &F : M) {
    if (F.isDeclaration() ||
        F.empty() ||
        Selected.count(
            F.getName().str()) != 0)
      continue;

    for (Instruction &I :
         instructions(F)) {

      auto *CB =
          dyn_cast<CallBase>(&I);

      if (!CB ||
          CB->isNoInline())
        continue;

      // Inlining is controlled by caller.  An upgraded caller may use the O3
      // policy, but an unselected caller must remain at the already-produced
      // Oz baseline.
      CB->addFnAttr(
          Attribute::NoInline);

      CB->setMetadata(
          TempNoInlineMD,
          Tag);

      ++Added;
    }
  }

  return Added;
}

static void
removeCoptTemporaryNoInline(
    Module &M) {

  static constexpr StringLiteral
      TempNoInlineMD =
          "copt.ozbase.temp.noinline";

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    for (Instruction &I :
         instructions(F)) {

      auto *CB =
          dyn_cast<CallBase>(&I);

      if (!CB ||
          !CB->getMetadata(
              TempNoInlineMD))
        continue;

      CB->removeFnAttr(
          Attribute::NoInline);

      CB->setMetadata(
          TempNoInlineMD,
          nullptr);
    }
  }
}

static FunctionPassManager
buildCoptO3LocalDeltaPipeline(
    const PipelineTuningOptions &PTO) {

  FunctionPassManager FPM;

  // Minimal local canonicalization.  Do not replay the complete O3 function
  // pipeline: the point is to expose only COPT-controlled target deltas on top
  // of an already-optimized Oz function.
  FPM.addPass(
      InstCombinePass());

  // O3-only non-trivial unswitching.  This is a Loop pass, so run it through
  // the normal loop adaptor with MemorySSA enabled.
  LoopPassManager UnswitchLPM;

  UnswitchLPM.addPass(
      LoopInstSimplifyPass());

  UnswitchLPM.addPass(
      LoopSimplifyCFGPass());

  UnswitchLPM.addPass(
      SimpleLoopUnswitchPass(
          /*NonTrivial=*/true));

  FPM.addPass(
      createFunctionToLoopPassAdaptor(
          std::move(
              UnswitchLPM),
          /*UseMemorySSA=*/true));

  FPM.addPass(
      SimplifyCFGPass());

  FPM.addPass(
      InstCombinePass());

  // O3 loop-vectorization policy.
  //
  // LoopVectorizeOptions takes "disable" booleans. false/false means allow
  // vectorization and interleaving.
  FPM.addPass(
      LoopVectorizePass(
          LoopVectorizeOptions(
              /*InterleaveOnlyWhenForced=*/false,
              /*VectorizeOnlyWhenForced=*/false)));

  // O3 late unroll policy.  This is the pass containing the COPT multi-state
  // unroll hook, so evaluation/apply choices remain authoritative.
  FPM.addPass(
      LoopUnrollPass(
          LoopUnrollOptions(
              /*OptLevel=*/3,
              /*OnlyWhenForced=*/false,
              PTO.ForgetAllSCEVInLoopUnroll)));

  // Local cleanup only.  None of these can transform another function.
  FPM.addPass(
      InstCombinePass());

  FPM.addPass(
      SimplifyCFGPass());

  FPM.addPass(
      ADCEPass());

  return FPM;
}

static void
runCoptO3UpgradePipeline(
    const Config &Conf,
    Module &M,
    TargetMachine *TM,
    const CoptFunctionSet &Selected) {

  if (Selected.empty())
    return;

  if (!TM)
    report_fatal_error(
        "COPT: target machine is null in O3 upgrade pipeline");

  // This is a supplemental delta pipeline over an already-completed Oz
  // Full-LTO base, not LLVM's normal FullLTOPostLink default pipeline.
  constexpr ThinOrFullLTOPhase Phase =
      ThinOrFullLTOPhase::None;

  prepareCoptSelectedFunctionsForO3(
      M,
      Selected);

  PipelineTuningOptions PTO =
      Conf.PTO;

  PTO.LoopInterleaving = true;
  PTO.LoopVectorization = true;
  PTO.LoopUnrolling = true;

  // SLP is intentionally excluded until it has a COPT decision hook.
  PTO.SLPVectorization = false;

  // The Oz base already emitted/updated the call-graph profile.
  PTO.CallGraphProfile = false;

  PassBuilder PB(
      TM,
      PTO);

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(
      MAM);

  PB.registerCGSCCAnalyses(
      CGAM);

  PB.registerFunctionAnalyses(
      FAM);

  PB.registerLoopAnalyses(
      LAM);

  PB.crossRegisterProxies(
      LAM,
      FAM,
      CGAM,
      MAM);

  // ------------------------------------------------------------------------
  // 1. Selective cross-function inlining.
  // ------------------------------------------------------------------------
  //
  // This is the only module/CGSCC transformation in this correctness-first
  // delta pipeline.  Calls in every unselected caller are temporarily marked
  // noinline, so only selected callers can receive O3 inlining decisions.
  const unsigned BlockedCallsites =
      blockCoptInliningFromUnselectedCallers(
          M,
          Selected);

  InlineParams IP =
      getInlineParams(
          /*OptLevel=*/3,
          /*SizeOptLevel=*/0);

  ModulePassManager InlineMPM;

  InlineMPM.addPass(
      RequireAnalysisPass<
          ProfileSummaryAnalysis,
          Module>());

  InlineMPM.addPass(
      ModuleInlinerWrapperPass(
          IP,
          /*MandatoryFirst=*/false,
          InlineContext{
              Phase,
              InlinePass::CGSCCInliner}));

  InlineMPM.run(
      M,
      MAM);

  removeCoptTemporaryNoInline(
      M);

  // ------------------------------------------------------------------------
  // 2. Local target-delta passes.
  // ------------------------------------------------------------------------
  //
  // DO NOT use buildModuleSimplificationPipeline/buildModuleOptimizationPipeline
  // here.  Exact evaluation must not mutate 500+ unrelated functions merely
  // because one function plan was selected.
  FunctionPassManager LocalFPM =
      buildCoptO3LocalDeltaPipeline(
          PTO);

  unsigned AppliedFunctions = 0;

  for (const std::string &Name :
       Selected) {

    Function *F =
        M.getFunction(
            Name);

    // A selected function may disappear after inlining/DCE in the inliner.
    if (!F ||
        F->isDeclaration() ||
        F->empty())
      continue;

    LocalFPM.run(
        *F,
        FAM);

    ++AppliedFunctions;
  }

  errs()
      << "[COPT] local O3 delta: selected="
      << Selected.size()
      << ", applied-functions="
      << AppliedFunctions
      << ", temporarily-blocked-calls="
      << BlockedCallsites
      << "\n";
}

// copt_impl: END

static std::optional<CoptModuleScore>
materializeAndScoreCoptChoices(
    const Config &Conf,
    Module &Pristine,
    TargetMachine *TM,
    unsigned OptLevel,
    const ModuleSummaryIndex *ImportSummary,
    const std::unordered_map<
        std::string,
        std::string> &Choices) {

  (void)OptLevel;
  (void)ImportSummary;

  CoptContext &COPT =
      CoptContext::get();

  // IMPORTANT:
  //
  // Pristine is no longer the pre-optimization LTO module. The integrated
  // wrapper first runs the real Oz base pipeline on Mod. Every call to this
  // function therefore clones the already-optimized Oz baseline.
  std::unique_ptr<Module> EvalMod =
      CloneModule(Pristine);

  CoptFunctionSet SelectedFunctions =
      collectCoptUpgradeFunctions(
          COPT.candidates(),
          Choices);

  COPT.beginIntegratedEvaluation(
      *EvalMod,
      Choices);

  // Global-lower has no selected non-lower decision. In that case this is a
  // pure clone+score operation: no optimizer is run after Oz.
  if (!SelectedFunctions.empty())
    runCoptO3UpgradePipeline(
        Conf,
        *EvalMod,
        TM,
        SelectedFunctions);

  const bool Invalid =
      COPT.evaluationInvalid();

  const std::string InvalidReason =
      COPT.evaluationInvalidReason().str();

  COPT.endIntegratedEvaluation(
      *EvalMod);

  if (Invalid) {
    errs()
        << "[COPT] Rejecting non-replayable exact candidate: "
        << InvalidReason
        << "\n";

    return std::nullopt;
  }

  return scoreCoptModule(
      *EvalMod,
      TM);
}
static void
writeCoptEvaluationReport(
    StringRef Path,
    StringRef FunctionName,
    const CoptFunctionPlan &Plan,
    const CoptModuleScore &Lower,
    const CoptModuleScore &Candidate) {

  if (Path.empty())
    return;

  json::Object LowerJSON;
  LowerJSON["size_cost"] =
      Lower.SizeCost;
  LowerJSON["perf_cost_diagnostic_only"] =
      Lower.PerfCost;
  LowerJSON["total_profile_hotness"] =
      Lower.TotalHotness;
  LowerJSON["instruction_count"] =
      static_cast<int64_t>(
          Lower.InstructionCount);
  LowerJSON["profiled_blocks"] =
      static_cast<int64_t>(
          Lower.ProfiledBlocks);
  LowerJSON["fallback_blocks"] =
      static_cast<int64_t>(
          Lower.FallbackBlocks);

  json::Object CandidateJSON;
  CandidateJSON["size_cost"] =
      Candidate.SizeCost;
  CandidateJSON["perf_cost_diagnostic_only"] =
      Candidate.PerfCost;
  CandidateJSON["total_profile_hotness"] =
      Candidate.TotalHotness;
  CandidateJSON["instruction_count"] =
      static_cast<int64_t>(
          Candidate.InstructionCount);
  CandidateJSON["profiled_blocks"] =
      static_cast<int64_t>(
          Candidate.ProfiledBlocks);
  CandidateJSON["fallback_blocks"] =
      static_cast<int64_t>(
          Candidate.FallbackBlocks);

  json::Object DeltaJSON;

  DeltaJSON["size_cost"] =
      Candidate.SizeCost -
      Lower.SizeCost;

  DeltaJSON["perf_cost_diagnostic_only"] =
      Candidate.PerfCost -
      Lower.PerfCost;

  DeltaJSON["tti_performance_benefit_diagnostic_only"] =
      Lower.PerfCost -
      Candidate.PerfCost;

  DeltaJSON["instruction_count"] =
      static_cast<int64_t>(
          Candidate.InstructionCount) -
      static_cast<int64_t>(
          Lower.InstructionCount);

  json::Object Root;

  Root["version"] = 1;
  Root["function"] =
      FunctionName;
  Root["plan_id"] =
      Plan.ID;
  Root["lower_endpoint"] =
      Plan.IsLowerEndpoint;
  Root["upper_endpoint"] =
      Plan.IsUpperEndpoint;

  Root["global_lower"] =
      std::move(LowerJSON);

  Root["candidate"] =
      std::move(CandidateJSON);

  Root["delta"] =
      std::move(DeltaJSON);

  std::error_code EC;

  raw_fd_ostream OS(
      Path,
      EC,
      sys::fs::OF_Text);

  if (EC)
    report_fatal_error(
        Twine("COPT: cannot write evaluation report: ") +
        EC.message());

  OS << formatv(
      "{0:2}\n",
      json::Value(std::move(Root)));
}




// copt_impl: BEGIN
// COPT_INTERNAL_LTO_V2: legacy runNewPMPasses wrapper replaced by internal v2.

// copt_impl: END

static bool isEmptyModule(const Module &Mod) {
  // Module is empty if it has no functions, no globals, no inline asm and no
  // named metadata (aliases and ifuncs require functions or globals so we
  // don't need to check those explicitly).
  return Mod.empty() && Mod.global_empty() && Mod.named_metadata_empty() &&
         Mod.getModuleInlineAsm().empty();
}

bool lto::opt(const Config &Conf, TargetMachine *TM, unsigned Task, Module &Mod,
              bool IsThinLTO, ModuleSummaryIndex *ExportSummary,
              const ModuleSummaryIndex *ImportSummary,
              const std::vector<uint8_t> &CmdArgs) {
  llvm::TimeTraceScope timeScope("opt");
  if (EmbedBitcode == LTOBitcodeEmbedding::EmbedPostMergePreOptimized) {
    // FIXME: the motivation for capturing post-merge bitcode and command line
    // is replicating the compilation environment from bitcode, without needing
    // to understand the dependencies (the functions to be imported). This
    // assumes a clang - based invocation, case in which we have the command
    // line.
    // It's not very clear how the above motivation would map in the
    // linker-based case, so we currently don't plumb the command line args in
    // that case.
    if (CmdArgs.empty())
      LLVM_DEBUG(
          dbgs() << "Post-(Thin)LTO merge bitcode embedding was requested, but "
                    "command line arguments are not available");
    llvm::embedBitcodeInModule(Mod, llvm::MemoryBufferRef(),
                               /*EmbedBitcode*/ true, /*EmbedCmdline*/ true,
                               /*Cmdline*/ CmdArgs);
  }
  // No need to run any opt passes if the module is empty.
  // In theory these passes should take almost no time for an empty
  // module, however, this guards against doing any unnecessary summary-based
  // analysis in the case of a ThinLTO build where this might be an empty
  // regular LTO combined module, with a large combined index from ThinLTO.
  if (!isEmptyModule(Mod)) {
    // FIXME: Plumb the combined index into the new pass manager.
    runNewPMPasses(Conf, Mod, TM, Conf.OptLevel, IsThinLTO, ExportSummary,
                   ImportSummary);
  }
  return !Conf.PostOptModuleHook || Conf.PostOptModuleHook(Task, Mod);
}

static void codegen(const Config &Conf, TargetMachine *TM,
                    AddStreamFn AddStream, unsigned Task, Module &Mod,
                    const ModuleSummaryIndex &CombinedIndex) {
  llvm::TimeTraceScope timeScope("codegen");
  if (Conf.PreCodeGenModuleHook && !Conf.PreCodeGenModuleHook(Task, Mod))
    return;

  if (EmbedBitcode == LTOBitcodeEmbedding::EmbedOptimized)
    llvm::embedBitcodeInModule(Mod, llvm::MemoryBufferRef(),
                               /*EmbedBitcode*/ true,
                               /*EmbedCmdline*/ false,
                               /*CmdArgs*/ std::vector<uint8_t>());

  std::unique_ptr<ToolOutputFile> DwoOut;
  SmallString<1024> DwoFile(Conf.SplitDwarfOutput);
  if (!Conf.DwoDir.empty()) {
    std::error_code EC;
    if (auto EC = llvm::sys::fs::create_directories(Conf.DwoDir))
      report_fatal_error(Twine("Failed to create directory ") + Conf.DwoDir +
                         ": " + EC.message());

    DwoFile = Conf.DwoDir;
    sys::path::append(DwoFile, std::to_string(Task) + ".dwo");
    TM->Options.MCOptions.SplitDwarfFile = std::string(DwoFile);
  } else
    TM->Options.MCOptions.SplitDwarfFile = Conf.SplitDwarfFile;

  if (!DwoFile.empty()) {
    std::error_code EC;
    DwoOut = std::make_unique<ToolOutputFile>(DwoFile, EC, sys::fs::OF_None);
    if (EC)
      report_fatal_error(Twine("Failed to open ") + DwoFile + ": " +
                         EC.message());
  }

  Expected<std::unique_ptr<CachedFileStream>> StreamOrErr =
      AddStream(Task, Mod.getModuleIdentifier());
  if (Error Err = StreamOrErr.takeError())
    report_fatal_error(std::move(Err));
  std::unique_ptr<CachedFileStream> &Stream = *StreamOrErr;
  TM->Options.ObjectFilenameForDebug = Stream->ObjectPathName;

  // Create the codegen pipeline in its own scope so it gets deleted before
  // Stream->commit() is called. The commit function of CacheStream deletes
  // the raw stream, which is too early as streamers (e.g. MCAsmStreamer)
  // keep the pointer and may use it until their destruction. See #138194.
  {
    legacy::PassManager CodeGenPasses;
    TargetLibraryInfoImpl TLII(Mod.getTargetTriple());
    CodeGenPasses.add(new TargetLibraryInfoWrapperPass(TLII));
    // No need to make index available if the module is empty.
    // In theory these passes should not use the index for an empty
    // module, however, this guards against doing any unnecessary summary-based
    // analysis in the case of a ThinLTO build where this might be an empty
    // regular LTO combined module, with a large combined index from ThinLTO.
    if (!isEmptyModule(Mod))
      CodeGenPasses.add(
          createImmutableModuleSummaryIndexWrapperPass(&CombinedIndex));
    if (Conf.PreCodeGenPassesHook)
      Conf.PreCodeGenPassesHook(CodeGenPasses);
    if (TM->addPassesToEmitFile(CodeGenPasses, *Stream->OS,
                                DwoOut ? &DwoOut->os() : nullptr,
                                Conf.CGFileType))
      report_fatal_error("Failed to setup codegen");
    CodeGenPasses.run(Mod);

    if (DwoOut)
      DwoOut->keep();
  }

  if (Error Err = Stream->commit())
    report_fatal_error(std::move(Err));
}

static void splitCodeGen(const Config &C, TargetMachine *TM,
                         AddStreamFn AddStream,
                         unsigned ParallelCodeGenParallelismLevel, Module &Mod,
                         const ModuleSummaryIndex &CombinedIndex) {
  DefaultThreadPool CodegenThreadPool(
      heavyweight_hardware_concurrency(ParallelCodeGenParallelismLevel));
  unsigned ThreadCount = 0;
  const Target *T = &TM->getTarget();

  const auto HandleModulePartition =
      [&](std::unique_ptr<Module> MPart) {
        // We want to clone the module in a new context to multi-thread the
        // codegen. We do it by serializing partition modules to bitcode
        // (while still on the main thread, in order to avoid data races) and
        // spinning up new threads which deserialize the partitions into
        // separate contexts.
        // FIXME: Provide a more direct way to do this in LLVM.
        SmallString<0> BC;
        raw_svector_ostream BCOS(BC);
        WriteBitcodeToFile(*MPart, BCOS);

        // Enqueue the task
        CodegenThreadPool.async(
            [&](const SmallString<0> &BC, unsigned ThreadId) {
              LTOLLVMContext Ctx(C);
              Expected<std::unique_ptr<Module>> MOrErr =
                  parseBitcodeFile(MemoryBufferRef(BC.str(), "ld-temp.o"), Ctx);
              if (!MOrErr)
                report_fatal_error("Failed to read bitcode");
              std::unique_ptr<Module> MPartInCtx = std::move(MOrErr.get());

              std::unique_ptr<TargetMachine> TM =
                  createTargetMachine(C, T, *MPartInCtx);

              codegen(C, TM.get(), AddStream, ThreadId, *MPartInCtx,
                      CombinedIndex);
            },
            // Pass BC using std::move to ensure that it get moved rather than
            // copied into the thread's context.
            std::move(BC), ThreadCount++);
      };

  // Try target-specific module splitting first, then fallback to the default.
  if (!TM->splitModule(Mod, ParallelCodeGenParallelismLevel,
                       HandleModulePartition)) {
    SplitModule(Mod, ParallelCodeGenParallelismLevel, HandleModulePartition,
                false);
  }

  // Because the inner lambda (which runs in a worker thread) captures our local
  // variables, we need to wait for the worker threads to terminate before we
  // can leave the function scope.
  CodegenThreadPool.wait();
}

static Expected<const Target *> initAndLookupTarget(const Config &C,
                                                    Module &Mod) {
  if (!C.OverrideTriple.empty())
    Mod.setTargetTriple(Triple(C.OverrideTriple));
  else if (Mod.getTargetTriple().empty())
    Mod.setTargetTriple(Triple(C.DefaultTriple));

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(Mod.getTargetTriple(), Msg);
  if (!T)
    return make_error<StringError>(Msg, inconvertibleErrorCode());
  return T;
}

Error lto::finalizeOptimizationRemarks(LLVMRemarkFileHandle DiagOutputFile) {
  // Make sure we flush the diagnostic remarks file in case the linker doesn't
  // call the global destructors before exiting.
  if (!DiagOutputFile)
    return Error::success();
  DiagOutputFile.finalize();
  DiagOutputFile->keep();
  DiagOutputFile->os().flush();
  return Error::success();
}

Error lto::backend(const Config &C, AddStreamFn AddStream,
                   unsigned ParallelCodeGenParallelismLevel, Module &Mod,
                   ModuleSummaryIndex &CombinedIndex) {
  llvm::TimeTraceScope timeScope("LTO backend");
  Expected<const Target *> TOrErr = initAndLookupTarget(C, Mod);
  if (!TOrErr)
    return TOrErr.takeError();

  std::unique_ptr<TargetMachine> TM = createTargetMachine(C, *TOrErr, Mod);

  LLVM_DEBUG(dbgs() << "Running regular LTO\n");
  if (!C.CodeGenOnly) {
    if (!opt(C, TM.get(), 0, Mod, /*IsThinLTO=*/false,
             /*ExportSummary=*/&CombinedIndex, /*ImportSummary=*/nullptr,
             /*CmdArgs*/ std::vector<uint8_t>()))
      return Error::success();
  }

  if (ParallelCodeGenParallelismLevel == 1) {
    codegen(C, TM.get(), AddStream, 0, Mod, CombinedIndex);
  } else {
    splitCodeGen(C, TM.get(), AddStream, ParallelCodeGenParallelismLevel, Mod,
                 CombinedIndex);
  }
  return Error::success();
}

static void dropDeadSymbols(Module &Mod, const GVSummaryMapTy &DefinedGlobals,
                            const ModuleSummaryIndex &Index) {
  llvm::TimeTraceScope timeScope("Drop dead symbols");
  std::vector<GlobalValue*> DeadGVs;
  for (auto &GV : Mod.global_values())
    if (GlobalValueSummary *GVS = DefinedGlobals.lookup(GV.getGUID()))
      if (!Index.isGlobalValueLive(GVS)) {
        DeadGVs.push_back(&GV);
        convertToDeclaration(GV);
      }

  // Now that all dead bodies have been dropped, delete the actual objects
  // themselves when possible.
  for (GlobalValue *GV : DeadGVs) {
    GV->removeDeadConstantUsers();
    // Might reference something defined in native object (i.e. dropped a
    // non-prevailing IR def, but we need to keep the declaration).
    if (GV->use_empty())
      GV->eraseFromParent();
  }
}

Error lto::thinBackend(const Config &Conf, unsigned Task, AddStreamFn AddStream,
                       Module &Mod, const ModuleSummaryIndex &CombinedIndex,
                       const FunctionImporter::ImportMapTy &ImportList,
                       const GVSummaryMapTy &DefinedGlobals,
                       MapVector<StringRef, BitcodeModule> *ModuleMap,
                       bool CodeGenOnly, AddStreamFn IRAddStream,
                       const std::vector<uint8_t> &CmdArgs) {
  llvm::TimeTraceScope timeScope("Thin backend", Mod.getModuleIdentifier());
  Expected<const Target *> TOrErr = initAndLookupTarget(Conf, Mod);
  if (!TOrErr)
    return TOrErr.takeError();

  std::unique_ptr<TargetMachine> TM = createTargetMachine(Conf, *TOrErr, Mod);

  // Setup optimization remarks.
  auto DiagFileOrErr = lto::setupLLVMOptimizationRemarks(
      Mod.getContext(), Conf.RemarksFilename, Conf.RemarksPasses,
      Conf.RemarksFormat, Conf.RemarksWithHotness, Conf.RemarksHotnessThreshold,
      Task);
  if (!DiagFileOrErr)
    return DiagFileOrErr.takeError();
  auto DiagnosticOutputFile = std::move(*DiagFileOrErr);

  // Set the partial sample profile ratio in the profile summary module flag of
  // the module, if applicable.
  Mod.setPartialSampleProfileRatio(CombinedIndex);

  LLVM_DEBUG(dbgs() << "Running ThinLTO\n");
  if (CodeGenOnly) {
    // If CodeGenOnly is set, we only perform code generation and skip
    // optimization. This value may differ from Conf.CodeGenOnly.
    codegen(Conf, TM.get(), AddStream, Task, Mod, CombinedIndex);
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));
  }

  if (Conf.PreOptModuleHook && !Conf.PreOptModuleHook(Task, Mod))
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

  auto OptimizeAndCodegen =
      [&](Module &Mod, TargetMachine *TM,
          LLVMRemarkFileHandle DiagnosticOutputFile) {
        // Perform optimization and code generation for ThinLTO.
        if (!opt(Conf, TM, Task, Mod, /*IsThinLTO=*/true,
                 /*ExportSummary=*/nullptr, /*ImportSummary=*/&CombinedIndex,
                 CmdArgs))
          return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

        // Save the current module before the first codegen round.
        // Note that the second codegen round runs only `codegen()` without
        // running `opt()`. We're not reaching here as it's bailed out earlier
        // with `CodeGenOnly` which has been set in `SecondRoundThinBackend`.
        if (IRAddStream)
          cgdata::saveModuleForTwoRounds(Mod, Task, IRAddStream);

        codegen(Conf, TM, AddStream, Task, Mod, CombinedIndex);
        return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));
      };

  if (ThinLTOAssumeMerged)
    return OptimizeAndCodegen(Mod, TM.get(), std::move(DiagnosticOutputFile));

  // When linking an ELF shared object, dso_local should be dropped. We
  // conservatively do this for -fpic.
  bool ClearDSOLocalOnDeclarations =
      TM->getTargetTriple().isOSBinFormatELF() &&
      TM->getRelocationModel() != Reloc::Static &&
      Mod.getPIELevel() == PIELevel::Default;
  renameModuleForThinLTO(Mod, CombinedIndex, ClearDSOLocalOnDeclarations);

  dropDeadSymbols(Mod, DefinedGlobals, CombinedIndex);

  thinLTOFinalizeInModule(Mod, DefinedGlobals, /*PropagateAttrs=*/true);

  if (Conf.PostPromoteModuleHook && !Conf.PostPromoteModuleHook(Task, Mod))
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

  if (!DefinedGlobals.empty())
    thinLTOInternalizeModule(Mod, DefinedGlobals);

  if (Conf.PostInternalizeModuleHook &&
      !Conf.PostInternalizeModuleHook(Task, Mod))
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

  auto ModuleLoader = [&](StringRef Identifier) {
    llvm::TimeTraceScope moduleLoaderScope("Module loader", Identifier);
    assert(Mod.getContext().isODRUniquingDebugTypes() &&
           "ODR Type uniquing should be enabled on the context");
    if (ModuleMap) {
      auto I = ModuleMap->find(Identifier);
      assert(I != ModuleMap->end());
      return I->second.getLazyModule(Mod.getContext(),
                                     /*ShouldLazyLoadMetadata=*/true,
                                     /*IsImporting*/ true);
    }

    ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> MBOrErr =
        llvm::MemoryBuffer::getFile(Identifier);
    if (!MBOrErr)
      return Expected<std::unique_ptr<llvm::Module>>(make_error<StringError>(
          Twine("Error loading imported file ") + Identifier + " : ",
          MBOrErr.getError()));

    Expected<BitcodeModule> BMOrErr = findThinLTOModule(**MBOrErr);
    if (!BMOrErr)
      return Expected<std::unique_ptr<llvm::Module>>(make_error<StringError>(
          Twine("Error loading imported file ") + Identifier + " : " +
              toString(BMOrErr.takeError()),
          inconvertibleErrorCode()));

    Expected<std::unique_ptr<Module>> MOrErr =
        BMOrErr->getLazyModule(Mod.getContext(),
                               /*ShouldLazyLoadMetadata=*/true,
                               /*IsImporting*/ true);
    if (MOrErr)
      (*MOrErr)->setOwnedMemoryBuffer(std::move(*MBOrErr));
    return MOrErr;
  };

  {
    llvm::TimeTraceScope importScope("Import functions");
    FunctionImporter Importer(CombinedIndex, ModuleLoader,
                              ClearDSOLocalOnDeclarations);
    if (Error Err = Importer.importFunctions(Mod, ImportList).takeError())
      return Err;
  }

  // Do this after any importing so that imported code is updated.
  updateMemProfAttributes(Mod, CombinedIndex);
  updatePublicTypeTestCalls(Mod, CombinedIndex.withWholeProgramVisibility());

  if (Conf.PostImportModuleHook && !Conf.PostImportModuleHook(Task, Mod))
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

  return OptimizeAndCodegen(Mod, TM.get(), std::move(DiagnosticOutputFile));
}

BitcodeModule *lto::findThinLTOModule(MutableArrayRef<BitcodeModule> BMs) {
  if (ThinLTOAssumeMerged && BMs.size() == 1)
    return BMs.begin();

  for (BitcodeModule &BM : BMs) {
    Expected<BitcodeLTOInfo> LTOInfo = BM.getLTOInfo();
    if (LTOInfo && LTOInfo->IsThinLTO)
      return &BM;
  }
  return nullptr;
}

Expected<BitcodeModule> lto::findThinLTOModule(MemoryBufferRef MBRef) {
  Expected<std::vector<BitcodeModule>> BMsOrErr = getBitcodeModuleList(MBRef);
  if (!BMsOrErr)
    return BMsOrErr.takeError();

  // The bitcode file may contain multiple modules, we want the one that is
  // marked as being the ThinLTO module.
  if (const BitcodeModule *Bm = lto::findThinLTOModule(*BMsOrErr))
    return *Bm;

  return make_error<StringError>("Could not find module summary",
                                 inconvertibleErrorCode());
}

bool lto::initImportList(const Module &M,
                         const ModuleSummaryIndex &CombinedIndex,
                         FunctionImporter::ImportMapTy &ImportList) {
  if (ThinLTOAssumeMerged)
    return true;
  // We can simply import the values mentioned in the combined index, since
  // we should only invoke this using the individual indexes written out
  // via a WriteIndexesThinBackend.
  for (const auto &GlobalList : CombinedIndex) {
    // Ignore entries for undefined references.
    if (GlobalList.second.SummaryList.empty())
      continue;

    auto GUID = GlobalList.first;
    for (const auto &Summary : GlobalList.second.SummaryList) {
      // Skip the summaries for the importing module. These are included to
      // e.g. record required linkage changes.
      if (Summary->modulePath() == M.getModuleIdentifier())
        continue;
      // Add an entry to provoke importing by thinBackend.
      ImportList.addGUID(Summary->modulePath(), GUID, Summary->importType());
    }
  }
  return true;
}
