#include "EmitIR.hpp"
#include <llvm/Transforms/Utils/ModuleUtils.h>

#define self (*this)

using namespace asg;

EmitIR::EmitIR(Obj::Mgr& mgr, llvm::LLVMContext& ctx, llvm::StringRef mid)
  : mMgr(mgr)
  , mMod(mid, ctx)
  , mCtx(ctx)
  , mIntTy(llvm::Type::getInt32Ty(ctx))
  , mCurIrb(std::make_unique<llvm::IRBuilder<>>(ctx))
  , mCtorTy(llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false))
{
}

llvm::Module&
EmitIR::operator()(asg::TranslationUnit* tu)
{
  for (auto&& i : tu->decls)
    self(i);
  return mMod;
}

//==============================================================================
// 类型
//==============================================================================

llvm::Type*
EmitIR::operator()(const Type* type)
{
  if (type->texp == nullptr) {
    switch (type->spec) {
      case Type::Spec::kInt:
        return llvm::Type::getInt32Ty(mCtx);
      // TODO: 在此添加对更多基础类型的处理
      case Type::Spec::kVoid:
        return llvm::Type::getVoidTy(mCtx);
      default:
        ABORT();
    }
  }

  Type subt;
  subt.spec = type->spec;
  subt.qual = type->qual;
  subt.texp = type->texp->sub;

  // TODO: 在此添加对指针类型、数组类型和函数类型的处理
  llvm::ArrayType* arrType;
  std::vector<int> dimention;
  if (auto p = type->texp->dcst<ArrayType>()) { // 数组类型处理
    dimention.push_back(p->len);
    while(p->sub != nullptr){
      if (auto q = p->sub->dcst<ArrayType>()){
         p = q;
         dimention.push_back(p->len);
       }
    }
    int len = dimention.back();
    dimention.pop_back();
    arrType = llvm::ArrayType::get(llvm::Type::getInt32Ty(mCtx), len); 
    if (!dimention.empty()){
      len = dimention.back();
      dimention.pop_back();
      arrType = llvm::ArrayType::get(arrType, len);
    }
    return arrType;
  }
  
  if (auto p = type->texp->dcst<PointerType>()){ // 指针类型处理
    llvm::Type *pointee = llvm::Type::getInt32Ty(mCtx);
    llvm::Type *pointer = llvm::PointerType::get(pointee, 0);
    return pointer;
  }

  if (auto p = type->texp->dcst<FunctionType>()) { // 函数类型处理
    std::vector<llvm::Type*> pty;
    // TODO: 在此添加对函数参数类型的处理
    for (int i = 0; i < p->params.size();i++){
      pty.push_back(self(p->params[i]));
    }
    return llvm::FunctionType::get(self(&subt), std::move(pty), false);
  }

  ABORT();
}

//==============================================================================
// 表达式
//==============================================================================

llvm::Value*
EmitIR::operator()(Expr* obj)
{
  // TODO: 在此添加对更多表达式处理的跳转
  if (auto p = obj->dcst<IntegerLiteral>())
    return self(p);

  if (auto p = obj->dcst<DeclRefExpr>())
    return self(p);

  if (auto p = obj->dcst<ImplicitCastExpr>())
    return self(p);

  if (auto p = obj->dcst<BinaryExpr>())
    return self(p);

  if (auto p = obj->dcst<UnaryExpr>())
    return self(p);

  if (auto p = obj->dcst<ParenExpr>())
    return self(p);

  if (auto p = obj->dcst<InitListExpr>())
    return self(p);

  if (auto p = obj->dcst<CallExpr>())
    return self(p);

  ABORT();
}

llvm::Constant*
EmitIR::operator()(IntegerLiteral* obj) //常量表达式
{
  return llvm::ConstantInt::get(self(obj->type), obj->val);
}

// TODO: 在此添加对更多表达式类型的处理
llvm::Value*
EmitIR::operator()(ImplicitCastExpr* obj)
{
  auto sub = self(obj->sub); //处理子表达式（如DeclRefExpr类型，得到指针）
  auto& irb = *mCurIrb;     
  switch (obj->kind) {       
    case ImplicitCastExpr::kLValueToRValue: { // 左值到右值,由指针获得值
      auto ty = self(obj->sub->type); //获取类型
      auto loadVal = irb.CreateLoad(ty, sub); //load
      return loadVal;
    }

    case ImplicitCastExpr::kFunctionToPointerDecay:{
      return sub;
    }

    case ImplicitCastExpr::kArrayToPointerDecay:{
      return sub;
    }
    
    default:
      ABORT();
  }
}

llvm::Value*
EmitIR::operator()(DeclRefExpr* obj)
{
  // 在LLVM IR层面，左值体现为返回指向值的指针
  return reinterpret_cast<llvm::Value*>(obj->decl->any);
}

llvm::Value*
EmitIR::operator()(BinaryExpr* obj)
{
  llvm::Value *lftVal, *rhtVal;
  if(obj->op!=BinaryExpr::kAnd && obj->op!=BinaryExpr::kOr){
    lftVal = self(obj->lft);
    rhtVal = self(obj->rht);
  }
  auto& irb = *mCurIrb;
  //static std::vector<llvm::Value*> idxList;
  
  switch (obj->op) {
    case BinaryExpr::kAdd:
      return irb.CreateAdd(lftVal, rhtVal); 
      
    case BinaryExpr::kAssign:
      return irb.CreateStore(rhtVal, lftVal);
    
    case BinaryExpr::kSub:
      return irb.CreateSub(lftVal, rhtVal);
        
    case BinaryExpr::kMul:
      return irb.CreateMul(lftVal, rhtVal);  
      
    case BinaryExpr::kDiv:
      return irb.CreateSDiv(lftVal, rhtVal);
    
    case BinaryExpr::kMod:
      return irb.CreateSRem(lftVal, rhtVal);

    case BinaryExpr::kGt:
      return irb.CreateICmpSGT(lftVal, rhtVal);

    case BinaryExpr::kLt:
      return irb.CreateICmpSLT(lftVal, rhtVal);

    case BinaryExpr::kGe:
      return irb.CreateICmpSGE(lftVal, rhtVal);

    case BinaryExpr::kLe:
      return irb.CreateICmpSLE(lftVal, rhtVal);
    
    case BinaryExpr::kEq:
      return irb.CreateICmpEQ(lftVal, rhtVal);

    case BinaryExpr::kNe:
      return irb.CreateICmpNE(lftVal, rhtVal);   

    case BinaryExpr::kAnd:{
      llvm::BasicBlock *curBlock = irb.GetInsertBlock(); //获得当前基本块
      llvm::Function *func = curBlock->getParent();      //获得当前函数
      llvm::BasicBlock *lhsTrueBlock = llvm::BasicBlock::Create(mCtx, "land.rhs", func); //创建右值块
      llvm::BasicBlock *landEndBlock = llvm::BasicBlock::Create(mCtx, "land.end", func); //创建结束块

      llvm::Value* left_result_ = self(obj->lft);         //处理左边式子
      llvm::Value* left_result;
      if (left_result_->getType()->isIntegerTy(32))
        left_result = irb.CreateICmpNE(left_result_, irb.getInt32(0), "isNonZero");
      else
        left_result = left_result_;
      irb.CreateCondBr(left_result, lhsTrueBlock, landEndBlock); // 左边为真，则跳转到 land.rhs，否则跳转到 land.end
      llvm::BasicBlock *leftend = irb.GetInsertBlock(); //获得处理左边后的基本块

      irb.SetInsertPoint(lhsTrueBlock); // 将当前 IR 插入点设置为 land.rhs
      llvm::Value* right_result_ = self(obj->rht);         //处理右边式子
      llvm::Value* right_result;
      if (right_result_->getType()->isIntegerTy(32))
        right_result = irb.CreateICmpNE(right_result_, irb.getInt32(0), "isNonZero");
      else
        right_result = right_result_;
      irb.CreateBr(landEndBlock); // 无条件跳转到 land.end
      llvm::BasicBlock *rightend = irb.GetInsertBlock(); //获得处理右边后的基本块

      irb.SetInsertPoint(landEndBlock); // 将当前 IR 插入点设置为 land.end
      llvm::PHINode *phi = irb.CreatePHI(llvm::Type::getInt1Ty(mCtx), 2, "merge");
      phi->addIncoming(irb.getInt1(false), leftend); // 如果来自初始左值块，则一定为假
      phi->addIncoming(right_result, rightend);   // 如果来自右值块，则为右值结果
      return phi;
    }

    case BinaryExpr::kOr:{
      llvm::BasicBlock *curBlock = irb.GetInsertBlock(); //获得当前基本块
      llvm::Function *func = curBlock->getParent();      //获得当前函数
      llvm::BasicBlock *lhsFalseBlock = llvm::BasicBlock::Create(mCtx, "lor.rhs", func); //创建右值块
      llvm::BasicBlock *lorEndBlock = llvm::BasicBlock::Create(mCtx, "lor.end", func); //创建结束块   

      llvm::Value* left_result_ = self(obj->lft);         //处理左边式子
      llvm::Value* left_result;
      if (left_result_->getType()->isIntegerTy(32))
        left_result = irb.CreateICmpNE(left_result_, irb.getInt32(0), "isNonZero");
      else
        left_result = left_result_;
      irb.CreateCondBr(left_result, lorEndBlock, lhsFalseBlock); //左边为真，则跳转到 lor.end，否则跳转到 lor.rhs
      llvm::BasicBlock *leftend = irb.GetInsertBlock(); //获得处理左边后的基本块

      irb.SetInsertPoint(lhsFalseBlock);  // 将当前 IR 插入点设置为 lor.rhs
      llvm::Value* right_result_ = self(obj->rht);         //处理右边式子
      llvm::Value* right_result;
      if (right_result_->getType()->isIntegerTy(32))
        right_result = irb.CreateICmpNE(right_result_, irb.getInt32(0), "isNonZero");
      else
        right_result = right_result_;
      irb.CreateBr(lorEndBlock);// 无条件跳转到 lor.end
      llvm::BasicBlock *rightend = irb.GetInsertBlock(); //获得处理左边后的基本块

      irb.SetInsertPoint(lorEndBlock);   // 将当前 IR 插入点设置为 lor.end
      llvm::PHINode *phi = irb.CreatePHI(llvm::Type::getInt1Ty(mCtx), 2, "merge");
      phi->addIncoming(irb.getInt1(true), leftend); // 如果来自初始左值块，则一定为真
      phi->addIncoming(right_result, rightend); // 如果来自右值块，则为右值结果
      return phi;
    }

    case BinaryExpr::kIndex:{
      std::vector<llvm::Value*> idxList;
      idxList.push_back(rhtVal);
      return irb.CreateInBoundsGEP(self(obj->type), lftVal, idxList);

      // idxList.push_back(rhtVal);
      // auto q = obj->lft->dcst<ImplicitCastExpr>();
      // if (auto p = q->sub->dcst<BinaryExpr>()) {
      //   return self(q->sub);
      // }
      // if(auto p = q->sub->dcst<DeclRefExpr>()){
      //   idxList.push_back(irb.getInt64(0));
      //   std::reverse(idxList.begin(), idxList.end());
      //   llvm::Value *element = irb.CreateInBoundsGEP(self(p->type), self(p), idxList);
      //   idxList.clear();
      //   return element;
      // }
      // ABORT();
    }

    default:
      ABORT();
  }
}

llvm::Value*
EmitIR::operator()(UnaryExpr* obj)
{
  llvm::Value *sub = self(obj->sub);
  auto& irb = *mCurIrb;
  
  switch (obj->op) {
    case UnaryExpr::kNeg:
      return irb.CreateNeg(sub);
    case UnaryExpr::kNot:{
      llvm::Value* result_= sub; 
      llvm::Value* result;
      if (result_->getType()->isIntegerTy(32))
        result = irb.CreateICmpNE(result_, irb.getInt32(0), "isNonZero");
      else
        result = result_;
      return irb.CreateNot(result);
    }
    case UnaryExpr::kPos:
      return sub;   
    default:
      ABORT();
  }
}

llvm::Value*
EmitIR::operator()(ParenExpr* obj)
{
  return self(obj->sub);
}

llvm::Value* Array;

llvm::Value*
EmitIR::operator()(InitListExpr* obj)
{
  llvm::ArrayType* ty = reinterpret_cast<llvm::ArrayType*>(self(obj->type));
  auto& irb = *mCurIrb;
  llvm::Value* pointer = irb.CreateAlloca(ty, nullptr);
  for (size_t i = 0; i < obj->list.size(); ++i) {
    if(obj->list[i]->dcst<ImplicitInitExpr>()){
      return llvm::Constant::getNullValue(ty);
    } 
    if (self(obj->list[i]->type)->isIntegerTy()) { // 第一层
      auto val = self(obj->list[i]);
      std::vector<llvm::Value*> idxList{ irb.getInt64(0), irb.getInt64((int)i) };
      llvm::Value *element = irb.CreateInBoundsGEP(ty, pointer, idxList);
      irb.CreateStore(val, element);
    } else {                               // 第二层
      auto pointer1D = self(obj->list[i]); //获得第一层指针
      llvm::ArrayType* innerArrayType = llvm::dyn_cast<llvm::ArrayType>(ty->getElementType()); //获得第一层类型
      for (size_t j = 0; j < innerArrayType->getNumElements();j++){ //遍历第一层元素
        std::vector<llvm::Value*> idxList1D{ irb.getInt64(0), irb.getInt64((int)j) };
        llvm::Value *element1D = irb.CreateInBoundsGEP(innerArrayType, pointer1D, idxList1D);
        auto val = irb.CreateLoad(llvm::Type::getInt32Ty(mCtx), element1D);  //取出值
        std::vector<llvm::Value*> idxList{ irb.getInt64(0), irb.getInt64((int)i), irb.getInt64((int)j) };
        llvm::Value *element = irb.CreateInBoundsGEP(ty, pointer, idxList);
        irb.CreateStore(val, element);       //存入值 
      }
    }
  }
  return pointer;
}


llvm::Value*
EmitIR::operator()(CallExpr* obj)
{
  llvm::Function *func = reinterpret_cast<llvm::Function*>(self(obj->head));
  std::vector<llvm::Value*> params;
  for (int i = 0; i < obj->args.size();i++){
    params.push_back(self(obj->args[i]));
  }
  auto& irb = *mCurIrb;
  return irb.CreateCall(func,std::move(params));
}

//==============================================================================
// 语句
//==============================================================================

void
EmitIR::operator()(Stmt* obj)
{
  // TODO: 在此添加对更多Stmt类型的处理的跳转

  if (auto p = obj->dcst<CompoundStmt>())
    return self(p);

  if (auto p = obj->dcst<ReturnStmt>())
    return self(p);

  if (auto p = obj->dcst<DeclStmt>())
    return self(p);

  if (auto p = obj->dcst<ExprStmt>())
    return self(p);

  if (auto p = obj->dcst<IfStmt>())
    return self(p);

  if (auto p = obj->dcst<WhileStmt>())
    return self(p);

  if (auto p = obj->dcst<BreakStmt>())
    return self(p);
  
  if (auto p = obj->dcst<ContinueStmt>())
    return self(p);

  if (auto p = obj->dcst<NullStmt>())
    return;
    
  ABORT();
}

// TODO: 在此添加对更多Stmt类型的处理
llvm::BasicBlock *latest_condition_begin, *latest_while_end;

void
EmitIR::operator()(BreakStmt* obj){
  auto& irb = *mCurIrb;
  irb.CreateBr(latest_while_end);
}

void
EmitIR::operator()(ContinueStmt* obj){
  auto& irb = *mCurIrb;
  irb.CreateBr(latest_condition_begin);
}

void
EmitIR::operator()(WhileStmt* obj){
  auto& irb = *mCurIrb;
  llvm::BasicBlock *curBlock = irb.GetInsertBlock(); //获得当前基本块
  llvm::Function *func = curBlock->getParent();      //获得当前函数
  llvm::BasicBlock *condition_begin = llvm::BasicBlock::Create(mCtx, "condition_begin", func);
  latest_condition_begin = condition_begin;
  irb.CreateBr(condition_begin); // 当前块无条件跳转到condition_begin
  // condition---------------
  irb.SetInsertPoint(condition_begin);
  auto result_ = self(obj->cond); //处理condition
  llvm::Value* result;
  if (result_->getType()->isIntegerTy(32))
    result = irb.CreateICmpNE(result_, irb.getInt32(0), "isNonZero");
  else
    result = result_;
  llvm::BasicBlock *condition_end = irb.GetInsertBlock(); //重新获取处理con后的基本块
  //build end---------------
  llvm::BasicBlock *while_end = llvm::BasicBlock::Create(mCtx, "while_end", func);
  latest_while_end = while_end;
  //body---------------------
  llvm::BasicBlock *body_begin = llvm::BasicBlock::Create(mCtx, "body_begin", func); 
  irb.SetInsertPoint(body_begin);
  self(obj->body);
  llvm::BasicBlock *body_end = irb.GetInsertBlock();      //重新获取处理body后的基本块
  // branch: condition -> body, end
  irb.SetInsertPoint(condition_end);
  irb.CreateCondBr(result, body_begin, while_end);
  //branch: body -> condition
  if(body_end->getTerminator() == nullptr){
      irb.SetInsertPoint(body_end);
      irb.CreateBr(condition_begin);
    }
  //set end
  irb.SetInsertPoint(while_end);
}

void
EmitIR::operator()(IfStmt* obj)
{
  auto& irb = *mCurIrb;
  llvm::BasicBlock *curBlock = irb.GetInsertBlock(); //获得当前基本块
  llvm::Function *func = curBlock->getParent();      //获得当前函数
  llvm::BasicBlock *if_then = llvm::BasicBlock::Create(mCtx, "if.then", func); 

  auto result_ = self(obj->cond);
  llvm::Value* result;
  if (result_->getType()->isIntegerTy(32))
    result = irb.CreateICmpNE(result_, irb.getInt32(0), "isNonZero");
  else
    result = result_;
  llvm::BasicBlock *end0 = irb.GetInsertBlock(); //重新获取处理con后的基本块

  if(obj->else_ != nullptr){
    //处理curblock->then、else间的跳转
    llvm::BasicBlock *if_else = llvm::BasicBlock::Create(mCtx, "if.else", func); 
    irb.SetInsertPoint(end0);
    irb.CreateCondBr(result, if_then, if_else);
    // 处理then部分
    irb.SetInsertPoint(if_then);
    self(obj->then);
    llvm::BasicBlock *end1 = irb.GetInsertBlock(); 
    // 处理else部分
    irb.SetInsertPoint(if_else);
    self(obj->else_);
    llvm::BasicBlock *end2 = irb.GetInsertBlock(); 
    //创建end
    llvm::BasicBlock *if_end = llvm::BasicBlock::Create(mCtx, "if.end", func); 
    //处理end1->end间的跳转（如果没有ret终结指令）
    if(end1->getTerminator() == nullptr){
      irb.SetInsertPoint(end1);
      irb.CreateBr(if_end);
    }
    //处理end2->end间的跳转（如果没有ret终结指令）
    if(end2->getTerminator() == nullptr){
      irb.SetInsertPoint(end2);
      irb.CreateBr(if_end);
    }

    irb.SetInsertPoint(if_end);
  }else{
    //处理then部分
    irb.SetInsertPoint(if_then);
    self(obj->then);
    llvm::BasicBlock *end1 = irb.GetInsertBlock(); 
    //创建end
    llvm::BasicBlock *if_end = llvm::BasicBlock::Create(mCtx, "if.end", func); 
    //处理curblock->end间的跳转
    irb.SetInsertPoint(end0);
    irb.CreateCondBr(result, if_then, if_end);
    //处理end1->end间的跳转（如果没有ret终结指令）
    if(end1->getTerminator() == nullptr){
      irb.SetInsertPoint(end1);
      irb.CreateBr(if_end);
    }

    irb.SetInsertPoint(if_end);
  }
}

void
EmitIR::operator()(DeclStmt* obj)
{
  for (auto&& dec : obj->decls)
    self(dec);
}

void
EmitIR::operator()(ExprStmt* obj)
{
    self(obj->expr);
}

void
EmitIR::operator()(CompoundStmt* obj)
{ // TODO: 可以在此添加对符号重名的处理
  //auto& irb = *mCurIrb;
  //auto sp = irb.CreateIntrinsic(llvm::Intrinsic::stacksave, {}, {}, nullptr, "sp");

  for (auto&& stmt : obj->subs)
    self(stmt);

  //irb.CreateIntrinsic(llvm::Intrinsic::stackrestore, {}, {sp});
}


void
EmitIR::operator()(ReturnStmt* obj)
{
  auto& irb = *mCurIrb;

  llvm::Value* retVal;
  if (!obj->expr)  //如果没有返回值
    retVal = nullptr;
  else             //如果有返回值 
    retVal = self(obj->expr);

  mCurIrb->CreateRet(retVal);

  //auto exitBb = llvm::BasicBlock::Create(mCtx, "return_exit", mCurFunc);
  //mCurIrb = std::make_unique<llvm::IRBuilder<>>(exitBb);
}

//==============================================================================
// 声明
//==============================================================================

void
EmitIR::operator()(Decl* obj)
{
  // TODO: 添加变量声明处理的跳转
  if (auto p = obj->dcst<VarDecl>())
    return self(p);

  if (auto p = obj->dcst<FunctionDecl>())
    return self(p);

  ABORT();
}

// TODO: 添加变量声明的处理 -----

void //变量的初始化
EmitIR::trans_init(llvm::Type* ty, llvm::Value* val, asg::Expr* obj)
{
  auto& irb = *mCurIrb;
  //常量--------------------------
  if (auto p = obj->dcst<IntegerLiteral>()) { 
    auto initVal = self(p);
    irb.CreateStore(initVal, val);
    return;
  }
  //空数组------------------------
  if (auto p = obj->dcst<ImplicitInitExpr>()){ 
    return;
  }
  //数组--------------------------
  if (auto p = obj->dcst<InitListExpr>()) {   
    if(p->list[0]->dcst<ImplicitInitExpr>()) return;
    llvm::Value* pointer = self(p);
    llvm::ArrayType* tyy = reinterpret_cast<llvm::ArrayType*>(ty);
    int outer = tyy->getNumElements(); //外层长度
    llvm::Type* innerArrayType = tyy->getElementType();
    if (auto q = llvm::dyn_cast<llvm::ArrayType>(innerArrayType)) { //两层
      int inner = q->getNumElements(); // 内层长度
      for (int i = 0; i < outer; i++)
        for (int j = 0; j < inner; j++){
          std::vector<llvm::Value*> idxList{ irb.getInt64(0), irb.getInt64(i), irb.getInt64(j) };
          llvm::Value *element1 = irb.CreateInBoundsGEP(tyy, pointer, idxList);
          auto initval = irb.CreateLoad(llvm::Type::getInt32Ty(mCtx), element1); 
          llvm::Value *element2 = irb.CreateInBoundsGEP(tyy, val, idxList);
          irb.CreateStore(initval, element2);        
        }
    } else { //一层
      for (int i = 0; i < outer; i++){
        std::vector<llvm::Value*> idxList{ irb.getInt64(0), irb.getInt64(i)};
        llvm::Value *element1 = irb.CreateInBoundsGEP(tyy, pointer, idxList);
        auto initval = irb.CreateLoad(llvm::Type::getInt32Ty(mCtx), element1); 
        llvm::Value *element2 = irb.CreateInBoundsGEP(tyy, val, idxList);
        irb.CreateStore(initval, element2);        
      }      
    }
    return;
  }
  //函数--------------------------------
  if (auto p = obj->dcst<CallExpr>()){
    auto initVal = self(p);
    irb.CreateStore(initVal, val);
    return;
  }
  //变量--------------------------------
  if (auto p = obj->dcst<ImplicitCastExpr>()){
    auto initVal = self(p);
    irb.CreateStore(initVal, val);
    return;
  }
  //二元--------------------------------
  if (auto p = obj->dcst<BinaryExpr>()){
    auto initVal = self(p);
    irb.CreateStore(initVal, val);
    return;
  }
  //一元----------------------------------
  if (auto p = obj->dcst<UnaryExpr>()){
    auto initVal = self(p);
    irb.CreateStore(initVal, val);
    return;
  }
  ABORT();
}

void //变量的声明与初始化
EmitIR::operator()(VarDecl* obj)
{ 
  auto& irb = *mCurIrb;
  llvm::BasicBlock *curBlock = irb.GetInsertBlock();
  if(curBlock){ // 在写一个块，局部变量----------------------
    auto ty = self(obj->type); 
    auto lvar = irb.CreateAlloca(ty, nullptr, obj->name);
    obj->any = lvar;
    auto initval = llvm::Constant::getNullValue(ty);
    irb.CreateStore(initval, lvar);
    if (obj->init == nullptr)
      return;
    trans_init(ty, lvar, obj->init); 
  } else {      // 没有在写块，全局变量---------------------
    auto ty = self(obj->type); 
    auto gvar = new llvm::GlobalVariable(
      mMod, ty, false, llvm::GlobalVariable::ExternalLinkage, nullptr, obj->name);
    obj->any = gvar; //把llvm的处理声明结果存到any中，在引用的时候可以用到
    gvar->setInitializer(llvm::Constant::getNullValue(ty));  // 默认初始化为 0
    if (obj->init == nullptr)  // 如果声明没有进行初始化，则结束
      return;
    // 创建构造函数用于初始化
    mCurFunc = llvm::Function::Create(
      mCtorTy, llvm::GlobalVariable::PrivateLinkage, "ctor_" + obj->name, mMod);
    llvm::appendToGlobalCtors(mMod, mCurFunc, 65535);
    // 为函数创建 entry 基本块
    auto entryBb = llvm::BasicBlock::Create(mCtx, "entry", mCurFunc);
    mCurIrb = std::make_unique<llvm::IRBuilder<>>(entryBb);
    //调用初始化函数进行初始化
    trans_init(ty, gvar, obj->init); 
    mCurIrb->CreateRet(nullptr);
    //恢复到初始状态
    mCurIrb->ClearInsertionPoint();
  }

}

//函数声明
void
EmitIR::operator()(FunctionDecl* obj)
{
  // 创建函数
  auto fty = llvm::dyn_cast<llvm::FunctionType>(self(obj->type));
  auto func = llvm::Function::Create(
    fty, llvm::GlobalVariable::ExternalLinkage, obj->name, mMod);

  obj->any = func;

  if (obj->body == nullptr)
    return;
  auto entryBb = llvm::BasicBlock::Create(mCtx, "entry", func);
  mCurIrb = std::make_unique<llvm::IRBuilder<>>(entryBb);
  auto& entryIrb = *mCurIrb;

  // TODO: 添加对函数参数的处理
  auto argIter = func->arg_begin();
  for (int i = 0; i < obj->params.size();i++){
    argIter->setName(obj->params[i]->name);
    auto ty = self(obj->params[i]->type); 
    auto lvar = entryIrb.CreateAlloca(ty, nullptr, obj->params[i]->name);
    obj->params[i]->any = lvar;
    entryIrb.CreateStore(&*argIter, lvar);
    argIter++;
  }
  // 翻译函数体
  mCurFunc = func;
  self(obj->body);
  auto& exitIrb = *mCurIrb;

  if (fty->getReturnType()->isVoidTy())
    exitIrb.CreateRetVoid();
  else{
    llvm::BasicBlock* block = exitIrb.GetInsertBlock();
    if(block->getTerminator()==nullptr)
      exitIrb.CreateUnreachable();
  }
  mCurIrb->ClearInsertionPoint();
}