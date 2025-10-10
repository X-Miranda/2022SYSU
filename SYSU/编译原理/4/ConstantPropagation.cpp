#include "ConstantPropagation.hpp"

using namespace llvm;

PreservedAnalyses
ConstantPropagation::run(Module& mod, ModuleAnalysisManager& mam)
{
  int constFoldTimes = 0;
  std::vector<Value*> storeList; // 负责存储修改过的int，包含的变量不可以被当作常量
  std::unordered_map<Value*, ConstantInt*> intList; // 负责存储可以被当作常量的int

  // 遍历所有指令，如果是存储指令，则相关的变量不可以被传播
  for (auto& func : mod) 
    for (auto& bb : func) 
      for (auto& inst : bb) 
        if(auto storeInst = dyn_cast<StoreInst>(&inst)){
          auto pointer = storeInst->getPointerOperand();
          storeList.push_back(pointer);
        }

  // 遍历全局变量，将没有被store过的变量加入intlist
  for (auto& global : mod.globals()) { 
    if (global.hasInitializer()) 
      if (auto constVal = dyn_cast<ConstantInt>(global.getInitializer())){
        auto it = std::find(storeList.begin(), storeList.end(), &global);
        if(it == storeList.end())
          intList[&global] = constVal;
      }
  }

  // 遍历所有指令，消除地址在intlist中的load指令
  for (auto& func : mod) {
    for (auto& bb : func) {
      std::vector<Instruction*> instToErase;
      for (auto& inst : bb) {
        if(auto loadInst = dyn_cast<LoadInst>(&inst)){
          auto pointer = loadInst->getPointerOperand();
          if(intList.find(pointer) != intList.end()){
            loadInst->replaceAllUsesWith(ConstantInt::getSigned(loadInst->getType(),intList[pointer]->getSExtValue()));
            instToErase.push_back(loadInst);
            ++constFoldTimes;           
          }
        }
      }
      for (auto& i : instToErase)
        i->eraseFromParent();
    }
  }

  mOut << "ConstantPropagation running...\nTo eliminate " << constFoldTimes
       << " instructions\n";
  return PreservedAnalyses::all();
}