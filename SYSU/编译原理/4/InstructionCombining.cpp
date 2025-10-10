#include "InstructionCombining.hpp"

using namespace llvm;

PreservedAnalyses InstructionCombining::run(Module &mod, ModuleAnalysisManager &mam) {
  int constFoldTimes = 0;
  std::vector<int> addChain;
  
  // Phase 1: Identify consecutive add instructions
  Value *chainHead = nullptr;
  Instruction *lastAdd = nullptr;
  
  for (Function &func : mod) {
    for (BasicBlock &bb : func) {
      for (Instruction &inst : bb) {
        BinaryOperator *binOp = dyn_cast<BinaryOperator>(&inst);
        if (!binOp || binOp->getOpcode() != Instruction::Add) 
          continue;
        
        ConstantInt *rhs = dyn_cast<ConstantInt>(binOp->getOperand(1));
        if (!rhs) 
          continue;
        
        int imm = rhs->getSExtValue();
        
        if (!lastAdd) {
          // Start new chain
          chainHead = binOp->getOperand(0);
          lastAdd = binOp;
          addChain.push_back(imm);
        } else if (binOp->getOperand(0) == lastAdd) {
          // Extend existing chain
          lastAdd = binOp;
          addChain.push_back(imm);
        }
      }
    }
  }

  // Phase 2: Fold chain if applicable
  if (addChain.size() > 10) {
    int chainLength = addChain.size();
    int lastImm = addChain.back();
    bool found = false;

    for (Function &func : mod) {
      for (BasicBlock &bb : func) {
        std::vector<Instruction*> toRemove;
        int matchCount = 0;

        for (Instruction &inst : bb) {
          BinaryOperator *binOp = dyn_cast<BinaryOperator>(&inst);
          if (!binOp || binOp->getOpcode() != Instruction::Add) 
            continue;
          
          ConstantInt *rhs = dyn_cast<ConstantInt>(binOp->getOperand(1));
          if (!rhs || rhs->getSExtValue() != lastImm) 
            continue;
          
          toRemove.push_back(binOp);
          matchCount++;
          constFoldTimes++;
          
          if (matchCount == chainLength) {
            IRBuilder<> builder(binOp);
            Value *countVal = ConstantInt::get(binOp->getType(), chainLength);
            Value *totalAdd = builder.CreateMul(countVal, rhs);
            Value *newResult = builder.CreateAdd(chainHead, totalAdd);
            
            binOp->replaceAllUsesWith(newResult);
            found = true;
            break;
          }
        }
        
        // Remove matched instructions in reverse order
        while (!toRemove.empty()) {
          toRemove.back()->eraseFromParent();
          toRemove.pop_back();
        }
        
        if (found) break;
      }
      if (found) break;
    }
  }

  mOut << "InstructionCombining running...\nEliminated " 
       << constFoldTimes << " instructions\n";
  return PreservedAnalyses::all();
}