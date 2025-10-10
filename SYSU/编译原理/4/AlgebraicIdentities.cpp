#include "AlgebraicIdentities.hpp"

using namespace llvm;

// 检查值是否为零
static bool isZeroValue(Value *V) {
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
        return CI->isZero();
    }
    return false;
}

// 检查值是否为1
static bool isOneValue(Value *V) {
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
        return CI->isOne();
    }
    return false;
}

PreservedAnalyses AlgebraicIdentities::run(Module& mod, ModuleAnalysisManager& mam) {
    int optimizedCount = 0;

    for (auto& func : mod) {
        for (auto& bb : func) {
            std::vector<Instruction*> toRemove;

            for (auto& inst : bb) {
                if (auto binOp = dyn_cast<BinaryOperator>(&inst)) {
                    Value* lhs = binOp->getOperand(0);
                    Value* rhs = binOp->getOperand(1);
                    
                    // 根据操作类型应用恒等优化
                    switch (binOp->getOpcode()) {
                        case Instruction::Mul:  // 乘法优化
                            if (isZeroValue(lhs) || isZeroValue(rhs)) {
                                binOp->replaceAllUsesWith(ConstantInt::get(binOp->getType(), 0));
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            } 
                            else if (isOneValue(lhs)) {
                                binOp->replaceAllUsesWith(rhs);
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            }
                            else if (isOneValue(rhs)) {
                                binOp->replaceAllUsesWith(lhs);
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            }
                            break;
                            
                        case Instruction::SDiv:  // 有符号除法优化
                            if (isOneValue(rhs)) {
                                binOp->replaceAllUsesWith(lhs);
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            }
                            else if (isZeroValue(lhs)) {
                                binOp->replaceAllUsesWith(ConstantInt::get(binOp->getType(), 0));
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            }
                            break;
                            
                        case Instruction::Add:  // 加法优化
                            if (isZeroValue(lhs)) {
                                binOp->replaceAllUsesWith(rhs);
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            }
                            else if (isZeroValue(rhs)) {
                                binOp->replaceAllUsesWith(lhs);
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            }
                            break;
                            
                        case Instruction::Sub:  // 减法优化
                            if (isZeroValue(rhs)) {
                                binOp->replaceAllUsesWith(lhs);
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            }
                            break;
                            
                        case Instruction::SRem:  // 取模优化
                            if (isOneValue(rhs)) {
                                binOp->replaceAllUsesWith(ConstantInt::get(binOp->getType(), 0));
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            }
                            else if (isZeroValue(lhs)) {
                                binOp->replaceAllUsesWith(ConstantInt::get(binOp->getType(), 0));
                                toRemove.push_back(binOp);
                                optimizedCount++;
                            }
                            break;
                            
                        default:
                            break;
                    }
                }
            }
            
            // 删除优化后的指令
            for (auto inst : toRemove) {
                inst->eraseFromParent();
            }
        }
    }

    mOut << "Algebraic Identities Optimization\n";
    mOut << "================================\n";
    mOut << "Total optimizations: " << optimizedCount << "\n";
    
    return optimizedCount > 0 ? PreservedAnalyses::none() : PreservedAnalyses::all();
}