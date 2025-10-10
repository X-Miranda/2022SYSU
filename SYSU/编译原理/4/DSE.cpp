#include "DSE.hpp"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/Local.h"
#include <vector>
#include <queue>
#include <algorithm>

using namespace llvm;

// 检查指令是否安全删除
static bool isSafeToDelete(Instruction* I) {
    // 有副作用的指令不能删除
    if (I->mayHaveSideEffects()) {
        // 特殊处理存储指令
        if (auto* SI = dyn_cast<StoreInst>(I)) {
            Value* ptr = SI->getPointerOperand();
            // 如果地址有多个使用或者是复杂表达式，不能删除
            if (ptr->hasNUsesOrMore(2) || !isa<AllocaInst>(ptr)) {
                return false;
            }
        }
        return false;
    }
    return true;
}

// 删除死存储及其依赖链
static void removeDeadStoreAndDeps(StoreInst* SI, std::vector<Instruction*>& toRemove) {
    std::queue<Instruction*> worklist;
    worklist.push(SI);
    
    while (!worklist.empty()) {
        Instruction* current = worklist.front();
        worklist.pop();
        
        // 如果指令已经在删除列表中，跳过
        if (std::find(toRemove.begin(), toRemove.end(), current) != toRemove.end()) {
            continue;
        }
        
        // 添加到删除列表
        toRemove.push_back(current);
        
        // 检查值操作数是否也可以删除
        if (auto* valueInst = dyn_cast<Instruction>(SI->getValueOperand())) {
            // 如果该值只在这个存储指令中使用
            if (valueInst->hasOneUse()) {
                worklist.push(valueInst);
            }
        }
    }
}

// 函数级别的死存储消除
static int eliminateDeadStorage(Function &F) {
    int numRemoved = 0;
    std::vector<Instruction*> toRemove;
    
    // 第一遍：收集要删除的指令
    for (auto &BB : F) {
        for (auto &I : BB) {
            // 未使用的 alloca
            if (auto* AI = dyn_cast<AllocaInst>(&I)) {
                if (AI->use_empty()) {
                    toRemove.push_back(AI);
                }
            }
            // 死存储
            else if (auto* SI = dyn_cast<StoreInst>(&I)) {
                Value* dest = SI->getPointerOperand();
                
                // 如果目标地址只有一个使用（就是这个存储指令）
                if (dest->hasOneUse()) {
                    // 如果是 alloca 或全局变量，可以安全删除
                    if (isa<AllocaInst>(dest) || isa<GlobalVariable>(dest)) {
                        removeDeadStoreAndDeps(SI, toRemove);
                    }
                }
            }
        }
    }
    
    // 第二遍：安全删除指令
    for (Instruction* I : toRemove) {
        // 确保指令仍然存在（即仍有父基本块）且可以安全删除
        if (I->getParent() && isSafeToDelete(I)) {
            I->eraseFromParent();
            numRemoved++;
        }
    }
    
    return numRemoved;
}

// DSEPass 运行函数
PreservedAnalyses DSEPass::run(Module &M, ModuleAnalysisManager &) {
    int totalRemoved = 0;
    
    for (auto &F : M) {
        totalRemoved += eliminateDeadStorage(F);
    }
    
    outs() << "Dead Store Elimination\n";
    outs() << "=======================\n";
    outs() << "Removed instructions: " << totalRemoved << "\n";
    
    return totalRemoved > 0 ? PreservedAnalyses::none() : PreservedAnalyses::all();
}