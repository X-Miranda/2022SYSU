/* 生成.output文件 */
%verbose

/* 用于调试 (yydebug) */
%define parse.trace

%code top {
int yylex (void);             // 该函数由 Flex 生成
void yyerror (char const *);	// 该函数定义在 par.cpp 中


}

%code requires {
#include "par.hpp"
#include <iostream>
}

%union {
  std::string* RawStr;
  par::Decls* Decls;
  par::Exprs* Exprs;

  asg::TranslationUnit* TranslationUnit;
  asg::Type* Type;
  asg::Expr* Expr;
  asg::Decl* Decl;
  asg::FunctionDecl* FunctionDecl;
  asg::Stmt* Stmt;
  asg::CompoundStmt* CompoundStmt;
  asg::ExprStmt* ExprStmt;
  asg::ReturnStmt* ReturnStmt;
  asg::IfStmt* IfStmt;
  asg::WhileStmt* WhileStmt;
  asg::ContinueStmt* ContinueStmt;
  asg::BreakStmt* BreakStmt;
  asg::NullStmt* NullStmt;
}

/* 在下面说明每个非终结符对应的 union 成员，以便进行编译期类型检查 */
%type <Type> declaration_specifiers type_specifier

%type <Expr> additive_expression multiplicative_expression unary_expression postfix_expression
%type <Expr> expression primary_expression assignment_expression initializer initializer_list
%type <Expr> logical_or_expression logical_and_expression equality_expression relational_expression

%type <Stmt> block_item statement
%type <CompoundStmt> compound_statement block_item_list
%type <ExprStmt> expression_statement
%type <ReturnStmt> jump_statement

%type <Decls> external_declaration declaration init_declarator_list parameter_list
%type <Exprs> argument_expression_list
%type <FunctionDecl> function_definition
%type <Decl> declarator init_declarator parameter_declaration

%type <TranslationUnit> translation_unit

%type <Stmt> selection_statement iteration_statement

%token <RawStr> IDENTIFIER CONSTANT
%token INT VOID RETURN CONST WHILE DO SWITCH BREAK CONTINUE
%token CHAR IF ELSE
%token EQ_EQ NE LE GE AND_AND OR_OR '>' '<' ';' '='

%start start

%%

// 起始符号
start
  :	{
      par::Symtbl::g = new par::Symtbl();
    }
    translation_unit
    {
      par::gTranslationUnit = $2;
      delete par::Symtbl::g;
    }
  ;

translation_unit
  : external_declaration
    {
      $$ = par::gMgr.make<asg::TranslationUnit>();
      for (auto&& decl: *$1)
        $$->decls.push_back(decl);
      delete $1;
    }
  | translation_unit external_declaration
    {
      $$ = $1;
      for (auto&& decl: *$2)
        $$->decls.push_back(decl);
      delete $2;
    }
  ;

external_declaration
  : function_definition
    {
      $$ = new par::Decls();
      $$->push_back($1);
    }
  | declaration { $$ = $1; }
  ;

function_definition
  : declaration_specifiers declarator
    {
      auto funcDecl = $2->dcst<asg::FunctionDecl>();
      ASSERT(funcDecl);
      // 设置当前全局的函数作用变量
      par::gCurrentFunction = funcDecl; 
      auto ty = par::gMgr.make<asg::Type>();
      if (funcDecl->type != nullptr)
        ty->texp = funcDecl->type->texp; 
      ty->spec = $1->spec, ty->qual = $1->qual;
      funcDecl->type = ty;

    }
    compound_statement
    {	
      $$ = par::gCurrentFunction;
      $$->name = $2->name;
      $$->body = $4;
    }
  ;

declaration
  : declaration_specifiers init_declarator_list ';'
    {
      for (auto decl: *$2)
      {
        auto ty = par::gMgr.make<asg::Type>();
        if (decl->type != nullptr)
          ty->texp = decl->type->texp; // 保留前面 ArrayType 的texp
        ty->spec = $1->spec, ty->qual = $1->qual;
        decl->type = ty;
        auto varDecl = dynamic_cast<asg::VarDecl*>(decl);
        if (varDecl != nullptr)
        {
          if (varDecl->init != nullptr)
            varDecl->init->type = decl->type;
        }
      }
      $$ = $2;
    }
  ;

declaration_specifiers
  : type_specifier { $$ = $1; }
  | CONST type_specifier
    {
      $$ = $2;
      $$->qual.const_ = true;
    }
  | type_specifier declaration_specifiers
    {
      $$ = $2;
      $$->spec = $1->spec;
    }
  | CONST declaration_specifiers
    {
      $$ = $2;
      $$->qual.const_ = true;
    }
  ;

type_specifier
  : VOID
    {
      $$ = par::gMgr.make<asg::Type>();
      $$->spec = asg::Type::Spec::kVoid;
    }
  | INT
    {
      $$ = par::gMgr.make<asg::Type>();
      $$->spec = asg::Type::Spec::kInt;
    }
  | CHAR
    {
      $$ = par::gMgr.make<asg::Type>();
      $$->spec = asg::Type::Spec::kChar;
    }
  /*| FLOAT
    {
      $$ = par::gMgr.make<asg::Type>();
      $$->spec = asg::Type::Spec::kFloat;
    }
  | DOUBLE
    {
      $$ = par::gMgr.make<asg::Type>();
      $$->spec = asg::Type::Spec::kDouble;
    }*/
  ;

declarator
  : IDENTIFIER
    {
      $$ = par::gMgr.make<asg::VarDecl>();
      $$->name = std::move(*$1);
      delete $1;

      // 插入符号表
      par::Symtbl::g->insert_or_assign($$->name, $$);
    }
  | declarator '[' ']' // 未知长度数组
    {
      $$ = $1; 
      // 填充Type
      auto ty = par::gMgr.make<asg::Type>();
      if ($$->type != nullptr)
        ty->texp=$$->type->texp;
      auto p = par::gMgr.make<asg::ArrayType>();
      p->len = asg::ArrayType::kUnLen;
      if (ty->texp == nullptr)
      {
        ty->texp = p;
      }
      else
      {
        ty->texp->sub = p;
      }
      $$->type = ty;

      // 插入符号表
      par::Symtbl::g->insert_or_assign($$->name, $$);
    }
  | declarator '[' assignment_expression ']' // 数组定义
    {
      $$ = $1; 
      // 填充Type
      auto ty = par::gMgr.make<asg::Type>();
      if ($$->type != nullptr)
        ty->texp=$$->type->texp;
      auto p = par::gMgr.make<asg::ArrayType>();
      auto integerLiteral = $3->dcst<asg::IntegerLiteral>();
      ASSERT(integerLiteral);
      p->len = integerLiteral->val;
      if (ty->texp == nullptr)
      {
        ty->texp = p;
      }
      else
      {
        ty->texp->sub = p;
      }
      $$->type = ty;

      // 插入符号表
      par::Symtbl::g->insert_or_assign($$->name, $$);
    }
  | declarator '(' ')'
    {
      $$ = par::gMgr.make<asg::FunctionDecl>();
      $$->name = $1->name;
      auto ty = par::gMgr.make<asg::Type>();
      auto p = par::gMgr.make<asg::FunctionType>();
      ty->texp = p;
      $$->type = ty;

      // 插入符号表
      par::Symtbl::g->insert_or_assign($$->name, $$);
    }
  // 函数列表的定义
  | declarator '(' parameter_list ')'
    {
      auto p = par::gMgr.make<asg::FunctionDecl>();
      p->name = $1->name;
      p->params = *$3;
      auto ty = par::gMgr.make<asg::Type>();
      auto functionType = par::gMgr.make<asg::FunctionType>();
      for (auto decl: *$3)
      {
        functionType->params.push_back(decl->type);
      }
      ty->texp = functionType;
      p->type = ty;
      $$ = p;

      // 插入符号表
      par::Symtbl::g->insert_or_assign($$->name, $$);
    }
  ;

parameter_list
  : parameter_declaration
    {
      $$ = new par::Decls();
      $$->push_back($1);
    }
  | parameter_list ',' parameter_declaration
    {
      $$ = $1;
      $$->push_back($3);
    }
  ;

parameter_declaration
  : declaration_specifiers declarator
    {
      // 保留之前定义的 Type
      auto ty = par::gMgr.make<asg::Type>();
      if ($2->type != nullptr)
        ty->texp = $2->type->texp;
      ty->spec = $1->spec, ty->qual = $1->qual;
      $2->type = ty;
      $$ = $2;
    }
  ;

compound_statement
  : {$$ = par::gMgr.make<asg::CompoundStmt>();} // 代码块为空的情况
  |'{' '}' { $$ = par::gMgr.make<asg::CompoundStmt>(); }
  | '{'
    { new par::Symtbl(); } 		// 开启新的符号表作用域
    block_item_list
    '}'
    {
      delete par::Symtbl::g; 	// 结束符号表作用域
      $$ = $block_item_list;
    }
  ;

block_item_list
  : block_item
    {
      $$ = par::gMgr.make<asg::CompoundStmt>();
      $$->subs.push_back($1);
    }
  | block_item_list block_item
    {
      $$ = $1;
      $$->subs.push_back($2);
    }
  ;

block_item
  : declaration
    {
      auto p = par::gMgr.make<asg::DeclStmt>();
      for (auto decl: *$1)
        p->decls.push_back(decl);
      $$ = p;
    }
  | statement { $$ = $1; }
  ;

statement
  : compound_statement { $$ = $1; }
  | expression_statement { $$ = $1; }
  | jump_statement { $$ = $1; }
  | selection_statement { $$ = $1; }
  | iteration_statement { $$ = $1; }
  | ';' { $$ = par::gMgr.make<asg::NullStmt>(); } // 空语句
  | BREAK ';' { $$ = par::gMgr.make<asg::BreakStmt>(); } // 添加break语句
  | CONTINUE ';' { $$ = par::gMgr.make<asg::ContinueStmt>(); } // 添加continue语句
  ;

expression_statement
  : expression ';'
    {
      $$ = par::gMgr.make<asg::ExprStmt>();
      $$->expr = $1;
    }
  ;

jump_statement
  : RETURN ';'
    {
      $$ = par::gMgr.make<asg::ReturnStmt>();
      $$->func = par::gCurrentFunction;
    }
  | RETURN expression ';'
    {
      $$ = par::gMgr.make<asg::ReturnStmt>();
      $$->func = par::gCurrentFunction;
      $$->expr = $2;
    } 
  ;

selection_statement
  : IF '(' expression ')' statement
    {
      auto p = par::gMgr.make<asg::IfStmt>();
      p->cond = $3;
      p->then = $5;
      $$ = p;
    }
  | IF '(' expression ')' statement ELSE statement
    {
      auto p = par::gMgr.make<asg::IfStmt>();
      p->cond = $3;
      p->then = $5;
      p->else_ = $7;
      $$ = p;
    }
  ;

  iteration_statement
  : WHILE '(' expression ')' statement
    {
      auto p = par::gMgr.make<asg::WhileStmt>();
      p->cond = $3;
      p->body = $5;
      $$ = p;
    }
  | DO statement WHILE '(' expression ')' ';'
    {
      auto p = par::gMgr.make<asg::DoStmt>();
      p->body = $2;
      p->cond = $5;
      $$ = p;
    }
  ;

expression
  : assignment_expression { $$ = $1; }
  | expression ',' assignment_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kComma;
      p->lft = $1, p->rht = $3;
      $$ = p;
    }
  ;

assignment_expression
  : logical_or_expression { $$ = $1; }
  | unary_expression '=' assignment_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kAssign;;
      p->lft = $1, p->rht = $3;
      $$ = p;
    }
  ;


logical_and_expression
  : equality_expression { $$ = $1; }
  | logical_and_expression AND_AND equality_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kAnd;
      p->lft = $1;
      p->rht = $3;
      $$ = p;
    }
  ;

logical_or_expression
  : logical_and_expression { $$ = $1; }
  | logical_or_expression OR_OR logical_and_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kOr;
      p->lft = $1;
      p->rht = $3;
      $$ = p;
    }
  ;

equality_expression
  : relational_expression { $$ = $1; }
  | equality_expression EQ_EQ relational_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kEq;
      p->lft = $1;
      p->rht = $3;
      $$ = p;
    }
  | equality_expression NE relational_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kNe;
      p->lft = $1;
      p->rht = $3;
      $$ = p;
    }
  ;

relational_expression
  : additive_expression { $$ = $1; }
  | relational_expression '<' additive_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kLt;
      p->lft = $1;
      p->rht = $3;
      $$ = p;
    }
  | relational_expression '>' additive_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kGt;
      p->lft = $1;
      p->rht = $3;
      $$ = p;
    }
  | relational_expression LE additive_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kLe;
      p->lft = $1;
      p->rht = $3;
      $$ = p;
    }
  | relational_expression GE additive_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kGe;
      p->lft = $1;
      p->rht = $3;
      $$ = p;
    }
  ;

additive_expression
  : multiplicative_expression { $$ = $1;}
  | additive_expression '+' multiplicative_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kAdd;
      p->lft = $1, p->rht = $3;
      $$ = p;
    }
  | additive_expression '-' multiplicative_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kSub;
      p->lft = $1, p->rht = $3;
      $$ = p;
    }
  ;

multiplicative_expression
  : unary_expression  { $$ = $1;}
  | multiplicative_expression '*' unary_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kMul;
      p->lft = $1, p->rht = $3;
      $$ = p;
    }
  | multiplicative_expression '/' unary_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kDiv;
      p->lft = $1, p->rht = $3;
      $$ = p;
    }
  | multiplicative_expression '%' unary_expression
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kMod;
      p->lft = $1, p->rht = $3;
      $$ = p;
    }
  ;

unary_expression
  : postfix_expression { $$ = $1; }
  | '+' unary_expression
    {
      auto p = par::gMgr.make<asg::UnaryExpr>();
      p->op = asg::UnaryExpr::Op::kPos;
      p->sub = $2;
      $$ = p;
    }
  | '-' unary_expression
    {
      auto p = par::gMgr.make<asg::UnaryExpr>();
      p->op = asg::UnaryExpr::Op::kNeg;
      p->sub = $2;
      $$ = p;
    }
  | '!' unary_expression
    {
      auto p = par::gMgr.make<asg::UnaryExpr>();
      p->op = asg::UnaryExpr::Op::kNot;
      p->sub = $2;
      $$ = p;
    }
  ;

postfix_expression
  : primary_expression { $$ = $1; }
  | postfix_expression '(' ')'  // 函数调用(无参数)
    {
      auto p = par::gMgr.make<asg::CallExpr>();
      p->head = $1;
      // 设置返回类型
      if (auto declRef = $1->dcst<asg::DeclRefExpr>()) {
        if (declRef->decl && declRef->decl->type) {
          p->type = declRef->decl->type;
        }
      }
      $$ = p;
    }
  | postfix_expression '(' argument_expression_list ')'  // 函数调用(有参数)
    {
      auto p = par::gMgr.make<asg::CallExpr>();
      p->head = $1;
      // 设置返回类型
      if (auto declRef = $1->dcst<asg::DeclRefExpr>()) {
        if (declRef->decl && declRef->decl->type) {
          p->type = declRef->decl->type;
        }
      }
      for (auto&& expr : *$3)
        p->args.push_back(expr);
      delete $3;
      $$ = p;
    }
  | postfix_expression '[' expression ']'  // 数组访问
    {
      auto p = par::gMgr.make<asg::BinaryExpr>();
      p->op = asg::BinaryExpr::Op::kIndex;
      p->lft = $1;
      p->rht = $3;
      $$ = p;
    }
  ;

primary_expression
  : IDENTIFIER
    {
      // 查找符号表, 找到对应的Decl
      auto decl = par::Symtbl::resolve(*$1);
      ASSERT(decl);
      delete $1;
      auto p = par::gMgr.make<asg::DeclRefExpr>();
      p->decl = decl;
      $$ = p;
    }
  | CONSTANT
    {
      auto p = par::gMgr.make<asg::IntegerLiteral>();
      std::string numStr = *$1;
      if (numStr.size() > 1 && numStr[0] == '0') {
        if (numStr[1] == 'x' || numStr[1] == 'X') {
          // 十六进制处理
          p->val = std::stoull(numStr.substr(2), nullptr, 16);
        } else {
          // 八进制处理
          p->val = std::stoull(numStr, nullptr, 8);
        }
      } else {
        // 十进制处理
        p->val = std::stoull(numStr, nullptr, 10);
      }
      delete $1;
      $$ = p;
    }
  | '(' expression ')'  // 括号表达式
    {
      auto p = par::gMgr.make<asg::ParenExpr>();
      p->sub = $2;
      $$ = p;
    }
  ;

argument_expression_list
  : assignment_expression
    {
      $$ = new par::Exprs();
      $$->push_back($1);
    }
  | argument_expression_list ',' assignment_expression
    {
      $$ = $1;
      $$->push_back($3);
    }
  ;

init_declarator_list
  : init_declarator
    {
      $$ = new par::Decls();
      $$->push_back($1);
    }
  | init_declarator_list ',' init_declarator
    {
      $$ = $1;
      $$->push_back($3);
    }
  ;

init_declarator
  : declarator { $$ = $1; }
  | declarator '=' initializer
    {
      auto varDecl = $1->dcst<asg::VarDecl>();
      ASSERT(varDecl);
      $3->type = varDecl->type;
      varDecl->init = $3;
      $$ = varDecl;
    }
  ;

// 初始化右值
initializer
  : assignment_expression
    {
      // 处理函数调用初始化
      if (auto callExpr = $1->dcst<asg::CallExpr>()) {
        $$ = callExpr;
        // 确保类型正确
        if ($$->type == nullptr && callExpr->head) {
          if (auto declRef = callExpr->head->dcst<asg::DeclRefExpr>()) {
            if (declRef->decl && declRef->decl->type) {
              $$->type = declRef->decl->type;
            }
          }
        }
      } 
      // 处理普通表达式初始化
      else {
        auto p = par::gMgr.make<asg::InitListExpr>();
        p->list.push_back($1);
        p->type = $1->type;  // 保留原始类型
        $$ = p;
      }
    }
  | '{' initializer_list '}' 
    { 
      $$ = $2; 
    }
  | '{' '}'
    {
      $$ = par::gMgr.make<asg::InitListExpr>();
    }
  ;

// 初始化列表
initializer_list
  : initializer
    {
      // 单个初始化项直接使用
      $$ = $1;
    }
  | initializer_list ',' initializer
    {
      // 合并初始化列表
      auto initList = $1->dcst<asg::InitListExpr>();
      auto newItem = $3;
      
      if (!initList) {
        // 如果第一个元素不是InitListExpr，创建新列表
        initList = par::gMgr.make<asg::InitListExpr>();
        initList->list.push_back($1);
      }
      
      // 如果新项是InitListExpr，展开它
      if (auto subList = newItem->dcst<asg::InitListExpr>()) {
        for (auto& item : subList->list) {
          initList->list.push_back(item);
        }
      } else {
        initList->list.push_back(newItem);
      }
      
      $$ = initList;
    }
  ;

%%
