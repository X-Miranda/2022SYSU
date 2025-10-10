#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include <llvm/Support/raw_ostream.h>

class DSEPass : public llvm::PassInfoMixin<DSEPass> {
public:

    explicit DSEPass(llvm::raw_ostream& out)
    : mOut(out) {}
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

private:
    llvm::raw_ostream& mOut;  
};