lexer grammar SYsULexer;

Int : 'int';
Return : 'return';
Const : 'const';

LeftParen : '(';
RightParen : ')';
LeftBracket : '[';
RightBracket : ']';
LeftBrace : '{';
RightBrace : '}';

Plus : '+';
Minus : '-';
Star : '*';
Div : '/';
Mod : '%';

Semi : ';';
Comma : ',';

Equal : '=';

For : 'for';          // 新增 for 关键字
While : 'while';      // 新增 while 关键字
Do : 'do';            // 新增 do 关键字
Switch : 'switch';    // 新增 switch 关键字
Case : 'case';        // 新增 case 关键字
Break : 'break';      // 新增 break 关键字
Continue : 'continue';// 新增 continue 关键字
Default : 'default';  // 新增 default 关键字
Float : 'float';      // 新增 float 关键字
Double : 'double';    // 新增 double 关键字
Char : 'char';        // 新增 char 关键字
Void : 'void';        // 新增 void 关键字

// 比较运算符
EqualEqual : '==';    // 等于
NotEqual : '!=';      // 不等于
Less : '<';           // 小于
LessEqual : '<=';     // 小于等于
Greater : '>';        // 大于
GreaterEqual : '>=';  // 大于等于

// 逻辑运算符
AndAnd : '&&';        // 逻辑与
OrOr : '||';          // 逻辑或
Not : '!';            // 逻辑非

Auto : 'auto';
Else : 'else';        // 新增 else 关键字
If : 'if';


Identifier
    :   IdentifierNondigit
        (   IdentifierNondigit
        |   Digit
        )*
    ;

fragment
IdentifierNondigit
    :   Nondigit
    ;

fragment
Nondigit
    :   [a-zA-Z_]
    ;

fragment
Digit
    :   [0-9]
    ;

Constant
    :   IntegerConstant
    |   HexadecimalConstant  // 新增十六进制常量
    //|   NegativeIntegerConstant  // 新增负数常量
    ;

fragment
IntegerConstant
    :   DecimalConstant
    |   OctalConstant
    ;

fragment
DecimalConstant
    :   NonzeroDigit Digit*
    ;

fragment
OctalConstant
    :   '0' OctalDigit*
    ;

fragment
HexadecimalConstant
    :   '0' [xX] HexadecimalDigit+  // 匹配 0x 或 0X 开头的十六进制数
    ;

/*fragment
NegativeIntegerConstant
    :   '-' DecimalConstant  // 匹配负的十进制常量
    |   '-' OctalConstant    // 匹配负的八进制常量
    |   '-' HexadecimalConstant  // 匹配负的十六进制常量
    ;*/

fragment
NonzeroDigit
    :   [1-9]
    ;

fragment
OctalDigit
    :   [0-7]
    ;

fragment
HexadecimalDigit
    :   [0-9a-fA-F]
    ;

// 预处理信息处理，可以从预处理信息中获得文件名以及行号
// 预处理信息前面的数组即行号
LineAfterPreprocessing
    :   '#' Whitespace* ~[\r\n]*
        //-> skip
    ;

Whitespace
    :   [ \t]+
        -> skip
    ;

// 换行符号，可以利用这个信息来更新行号
Newline
    :   (   '\r' '\n'?
        |   '\n'
        )
        -> skip
    ;

