#include "SYsULexer.h"
#include <antlr4-runtime.h>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

std::unordered_map<std::string, std::string> tokenTypeMapping = {
    {"Int", "int"},
    {"Return", "return"},
    {"Const", "const"},
    {"For", "for"},
    {"While", "while"},
    {"Do", "do"},
    {"Switch", "switch"},
    {"Case", "case"},
    {"Break", "break"},
    {"Continue", "continue"},
    {"Default", "default"},
    {"Float", "float"},
    {"Double", "double"},
    {"Char", "char"},
    {"Void", "void"},
    {"EqualEqual", "equalequal"},
    {"NotEqual", "exclaimequal"},
    {"Less", "less"},
    {"LessEqual", "lessequal"},
    {"Greater", "greater"},
    {"GreaterEqual", "greaterequal"},
    {"AndAnd", "ampamp"},
    {"OrOr", "pipepipe"},
    {"Not", "exclaim"},
    {"Plus", "plus"},
    {"Minus", "minus"},
    {"Star", "star"},
    {"Div", "slash"},
    {"Mod", "percent"},
    {"LeftParen", "l_paren"},
    {"RightParen", "r_paren"},
    {"LeftBracket", "l_square"},
    {"RightBracket", "r_square"},
    {"LeftBrace", "l_brace"},
    {"RightBrace", "r_brace"},
    {"Semi", "semi"},
    {"Comma", "comma"},
    {"Equal", "equal"},
    {"Identifier", "identifier"},
    {"Else", "else"},
    {"If", "if"},
    {"Constant", "numeric_constant"},
    {"HexadecimalConstant", "hexadecimal constant"},
    {"EOF", "eof"}
};

struct SourcePosition {
    std::string filename;
    int line;
    int column;
    int lineOffset;  // 新增：跟踪行号偏移
};

void handleLineDirective(const antlr4::Token* token, const std::string& directive, SourcePosition& pos) {
    size_t lineNumEnd = directive.find('"');
    if (lineNumEnd == std::string::npos) return;
    
    // 提取行号
    size_t lineNumStart = directive.find_first_not_of(" \t", 1); // 跳过#和空格
    if (lineNumStart == std::string::npos || lineNumStart >= lineNumEnd) return;
    
    int newLineNum = std::stoi(directive.substr(lineNumStart, lineNumEnd - lineNumStart));
    
    // 提取文件名
    size_t firstQuote = directive.find('"', lineNumEnd);
    size_t lastQuote = directive.rfind('"');
    if (firstQuote == std::string::npos || lastQuote == std::string::npos || firstQuote >= lastQuote)
        return;
    
    std::string newFileName = directive.substr(firstQuote + 1, lastQuote - firstQuote - 1);
    
    // 更新位置信息
    if (newFileName != "<built-in>" && newFileName != "<command line>") {
        pos.filename = newFileName;
    }
    
    // 关键修改：根据预处理指令中的行号重置行号偏移
    pos.lineOffset = token->getLine() - newLineNum;
}

void print_token(const antlr4::Token* token,
                 const antlr4::CommonTokenStream& tokens,
                 std::ofstream& outFile,
                 antlr4::Lexer& lexer,
                 SourcePosition& pos,
                 int lastLine) {
    auto& vocabulary = lexer.getVocabulary();
    std::string tokenTypeName = (token->getType() == antlr4::Token::EOF) ? 
                               "EOF" : 
                               std::string(vocabulary.getSymbolicName(token->getType()));
    
    if (tokenTypeName.empty()) tokenTypeName = "<UNKNOWN>";
    if (tokenTypeMapping.find(tokenTypeName) != tokenTypeMapping.end()) {
        tokenTypeName = tokenTypeMapping[tokenTypeName];
    }

    // 计算实际行号（考虑行号偏移）
    pos.line = token->getLine() - pos.lineOffset - 1;
    pos.column = token->getCharPositionInLine();

    bool startOfLine = (pos.line != lastLine);
    
    bool leadingSpace = false;
    if (pos.column > 0) {
        // 获取token前的文本
        size_t tokenStart = token->getStartIndex();
        if (tokenStart > 0) {
            char prevChar = lexer.getInputStream()->getText(antlr4::misc::Interval(tokenStart-1, tokenStart-1))[0];
            leadingSpace = (prevChar == ' ' || prevChar == '\t');
        }
    }

    if (token->getText() != "<EOF>") {
        outFile << tokenTypeName << " '" << token->getText() << "'";
    } else {
        outFile << tokenTypeName << " ''";
    }

    if (startOfLine) outFile << "\t [StartOfLine]";
    if (leadingSpace) outFile << "\t [LeadingSpace]";
    
    // 使用原始文件路径而不是预处理后的路径
    std::string outputPath = pos.filename;
    size_t build_pos = outputPath.find("/build/");
    if (build_pos != std::string::npos) {
        size_t test_pos = outputPath.find("/test/", build_pos);
        if (test_pos != std::string::npos) {
            outputPath = outputPath.substr(test_pos + 1);
        }
    }
    
    outFile << "\tLoc=<" << outputPath << ":" << pos.line << ":" << pos.column + 1 << ">" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input> <output>\n";
        return -1;
    }

    std::ifstream inFile(argv[1]);
    if (!inFile) {
        std::cout << "Error: unable to open input file: " << argv[1] << '\n';
        return -2;
    }

    std::ofstream outFile(argv[2]);
    if (!outFile) {
        std::cout << "Error: unable to open output file: " << argv[2] << '\n';
        return -3;
    }

    antlr4::ANTLRInputStream input(inFile);
    SYsULexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    tokens.fill();

    SourcePosition pos;
    pos.filename = argv[1];
    pos.line = 1;
    pos.column = 0;
    pos.lineOffset = 0;  // 初始行号偏移为0
    
    int lastLine = 0;

    for (auto&& token : tokens.getTokens()) {
        std::string tokenTypeName = std::string(lexer.getVocabulary().getSymbolicName(token->getType()));
        
        if (tokenTypeName == "LineAfterPreprocessing") {
            handleLineDirective(token, token->getText(), pos);
            continue;  // 预处理行不产生实际token
        }
        
        print_token(token, tokens, outFile, lexer, pos, lastLine);
        lastLine = pos.line;
    }

    return 0;
}