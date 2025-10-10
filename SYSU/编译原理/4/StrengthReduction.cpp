#include "StrengthReduction.hpp"
#include <llvm/IR/IRBuilder.h>

using namespace llvm;

PreservedAnalyses StrengthReduction::run(Module &mod, ModuleAnalysisManager &mam) {
  int replaceCount = 0;

  for (Function &func : mod) {
    for (BasicBlock &bb : func) {
      std::vector<Instruction*> toErase;
      IRBuilder<> builder(bb.getContext());

      for (Instruction &inst : bb) {
        auto *binOp = dyn_cast<BinaryOperator>(&inst);
        if (!binOp) continue;

        Value *lhs = binOp->getOperand(0);
        Value *rhs = binOp->getOperand(1);
        ConstantInt *constLhs = dyn_cast<ConstantInt>(lhs);
        ConstantInt *constRhs = dyn_cast<ConstantInt>(rhs);

        builder.SetInsertPoint(binOp);
        switch (binOp->getOpcode()) {
          case Instruction::Mul: { // 乘法优化
            if (constRhs && constRhs->getValue().isPowerOf2()) {
              APInt rhsVal = constRhs->getValue();
              unsigned shiftAmount = rhsVal.exactLogBase2();
              Value *newInst = builder.CreateShl(lhs, shiftAmount);
              binOp->replaceAllUsesWith(newInst);
              toErase.push_back(binOp);
              replaceCount++;
            }
            else if (constLhs && constLhs->getValue().isPowerOf2()) {
              APInt lhsVal = constLhs->getValue();
              unsigned shiftAmount = lhsVal.exactLogBase2();
              Value *newInst = builder.CreateShl(rhs, shiftAmount);
              binOp->replaceAllUsesWith(newInst);
              toErase.push_back(binOp);
              replaceCount++;
            }
            break;
          }
          case Instruction::SDiv: { // 除法优化
            if (constRhs && constRhs->getSExtValue() == 2) {
              Value *newInst = builder.CreateAShr(lhs, 1);
              binOp->replaceAllUsesWith(newInst);
              toErase.push_back(binOp);
              replaceCount++;
            }
            break;
          }
          /* 模运算优化保持原样
          case Instruction::SRem: {
            if (constRhs && constRhs->getValue().isPowerOf2()) {
              APInt rhsVal = constRhs->getValue();
              Value *mask = ConstantInt::get(binOp->getType(), rhsVal - 1);
              Value *newInst = builder.CreateAnd(lhs, mask);
              binOp->replaceAllUsesWith(newInst);
              toErase.push_back(binOp);
              replaceCount++;
            }
            break;
          }*/
          default:
            break;
        }
      }

      for (Instruction *i : toErase) {
        i->eraseFromParent();
      }
    }
  }

  mOut << "StrengthReduction running...\nReplaced " 
       << replaceCount << " instructions\n";
  return PreservedAnalyses::all();
}