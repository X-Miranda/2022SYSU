#pragma once

#include "LLMHelper.hpp"
#include <functional>
#include <initializer_list>
#include <llvm/IR/PassManager.h>
#include <string>

class PassSequencePredict : public llvm::PassInfoMixin<PassSequencePredict>
{
public:
  struct PassInfo
  {
    std::string mClassName;
    std::string mHppPath;
    std::string mCppPath;
    std::string mSummaryPath;
    std::function<void(llvm::ModulePassManager&)> mAddPass;
  };

  PassSequencePredict(llvm::StringRef apiKey,
                      llvm::StringRef baseURL,
                      std::initializer_list<PassInfo> passesInfo);

  llvm::PreservedAnalyses run(llvm::Module& mod,
                              llvm::ModuleAnalysisManager& mam);

private:
  LLMHelper mHelper;

  std::vector<PassInfo> mPassesInfo;

  std::string pass_summary(PassInfo& passInfo);
};
