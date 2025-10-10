#ifndef DCE_HPP
#define DCE_HPP

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/Analysis/LoopInfo.h"
#include <vector>

class DCEPass : public llvm::PassInfoMixin<DCEPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
    
private:
    // 判断指令是否为死代码
    bool isDeadInstruction(llvm::Instruction* I);
    
    // 删除基本块中的死代码
    bool eliminateDeadCodeInBlock(llvm::BasicBlock& BB);
    
    // 递归删除循环中的死代码
    bool eliminateDeadCodeInLoop(llvm::Loop* L, llvm::LoopInfo& LI);
};

#endif // DCE_HPP