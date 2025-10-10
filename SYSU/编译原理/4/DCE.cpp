#include "DCE.hpp"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

// 判断指令是否为死代码
bool DCEPass::isDeadInstruction(Instruction* I) {
    // 有副作用的指令不能删除
    if (I->mayHaveSideEffects()) {
        // 存储指令特殊处理：如果存储的值和地址都未被使用，且无副作用，可以删除
        if (auto* SI = dyn_cast<StoreInst>(I)) {
            return SI->getValueOperand()->use_empty() &&
                   SI->getPointerOperand()->use_empty();
        }
        return false;
    }
    
    // 无副作用的指令，如果没有使用，可以删除
    if (I->use_empty()) {
        return true;
    }
    
    // 特殊处理终止指令
    if (I->isTerminator()) {
        return false;
    }
    
    return false;
}

// 删除基本块中的死代码
bool DCEPass::eliminateDeadCodeInBlock(BasicBlock& BB) {
    bool changed = false;
    std::vector<Instruction*> deadInstructions;
    
    // 收集死指令
    for (Instruction& I : BB) {
        if (isDeadInstruction(&I)) {
            deadInstructions.push_back(&I);
        }
    }
    
    // 删除死指令
    for (Instruction* I : deadInstructions) {
        I->eraseFromParent();
        changed = true;
    }
    
    return changed;
}

// 删除循环中的死代码
bool DCEPass::eliminateDeadCodeInLoop(Loop* L, LoopInfo& LI) {
    bool changed = false;
    
    // 遍历循环中的所有基本块
    for (BasicBlock* BB : L->getBlocks()) {
        changed |= eliminateDeadCodeInBlock(*BB);
    }
    
    // 递归处理子循环
    for (Loop* SubLoop : L->getSubLoops()) {
        changed |= eliminateDeadCodeInLoop(SubLoop, LI);
    }
    
    return changed;
}

PreservedAnalyses DCEPass::run(Module& M, ModuleAnalysisManager& MAM) {
    bool changed = false;
    int eliminatedCount = 0;
    
    // 初始化分析管理器
    FunctionAnalysisManager& FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    
    // 第一遍：函数级别的死代码消除
    for (Function& F : M) {
        if (F.isDeclaration()) continue;
        
        // 获取循环信息
        LoopInfo& LI = FAM.getResult<LoopAnalysis>(F);
        
        // 处理整个函数
        for (BasicBlock& BB : F) {
            changed |= eliminateDeadCodeInBlock(BB);
        }
        
        // 处理循环
        for (Loop* L : LI) {
            changed |= eliminateDeadCodeInLoop(L, LI);
        }
    }
    
    // 第二遍：处理可能新产生的死代码
    if (changed) {
        for (Function& F : M) {
            if (F.isDeclaration()) continue;
            
            bool functionChanged;
            do {
                functionChanged = false;
                for (BasicBlock& BB : F) {
                    functionChanged |= eliminateDeadCodeInBlock(BB);
                }
                changed |= functionChanged;
            } while (functionChanged);
        }
    }

    // 输出统计信息
    errs() << "Dead Code Elimination\n";
    errs() << "=====================\n";
    errs() << "Eliminated instructions: " << eliminatedCount << "\n";
    
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}