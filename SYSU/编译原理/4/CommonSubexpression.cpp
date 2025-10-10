#include "CommonSubexpression.hpp"
#include <map>

using namespace llvm;

PreservedAnalyses CommonSubexpression::run(Module& mod, ModuleAnalysisManager& mam) {
    int eliminatedCount = 0;
    std::vector<Instruction*> toRemove;

    for (auto& func : mod) {
        for (auto& bb : func) {
            // 使用map存储表达式及其对应的指令
            std::map<std::pair<uintptr_t, uintptr_t>, Instruction*> exprMap;
            
            for (auto& inst : bb) {
                if (auto* binOp = dyn_cast<BinaryOperator>(&inst)) {
                    if (binOp->getOpcode() == Instruction::Add) {
                        Value* lhs = binOp->getOperand(0);
                        Value* rhs = binOp->getOperand(1);
                        
                        // 创建唯一表达式键
                        auto key = std::make_pair(
                            reinterpret_cast<uintptr_t>(lhs),
                            reinterpret_cast<uintptr_t>(rhs)
                        );
                        
                        // 检查是否已存在相同表达式
                        if (exprMap.find(key) != exprMap.end()) {
                            // 替换为已存在的表达式
                            binOp->replaceAllUsesWith(exprMap[key]);
                            toRemove.push_back(binOp);
                            eliminatedCount++;
                        } else {
                            // 记录新表达式
                            exprMap[key] = binOp;
                        }
                    }
                }
            }
        }
    }
    
    // 删除所有被替换的指令
    for (auto* inst : toRemove) {
        inst->eraseFromParent();
    }

    mOut << "Common Subexpression Elimination\n";
    mOut << "================================\n";
    mOut << "Eliminated expressions: " << eliminatedCount << "\n";
    
    return eliminatedCount > 0 ? PreservedAnalyses::none() : PreservedAnalyses::all();
}