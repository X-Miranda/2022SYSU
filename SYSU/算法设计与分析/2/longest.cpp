#include <iostream>
#include <string>
#include <vector>

using namespace std;

string findLongest(const string &str) {
    int n = str.length();
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
    int maxLength = 0; // 最长重复子字符串的长度
    int endIndex = 0; // 最长重复子字符串的结束索引

    // 构建dp数组
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            // 检查字符是否相同且子字符串不重叠
            if (str[i - 1] == str[j - 1] && dp[i - 1][j - 1] < (j - i)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                // 更新最长长度和结束索引
                if (dp[i][j] > maxLength) {
                    maxLength = dp[i][j];
                    endIndex = i - 1;
                }
            }
        }
    }

    // 提取最长重复非重叠子字符串
    if (maxLength > 0) {
        return str.substr(endIndex - maxLength + 1, maxLength);
    }
    return ""; // 如果没有重复子字符串，返回空字符串
}

int main() {
    string str1 ;
    while(cin >> str1){
    	cout << "Longest repeating non-overlapping substring: " << findLongest(str1) << endl;
	}
    return 0;
}

