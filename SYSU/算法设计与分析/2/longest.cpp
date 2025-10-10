#include <iostream>
#include <string>
#include <vector>

using namespace std;

string findLongest(const string &str) {
    int n = str.length();
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
    int maxLength = 0; // ��ظ����ַ����ĳ���
    int endIndex = 0; // ��ظ����ַ����Ľ�������

    // ����dp����
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            // ����ַ��Ƿ���ͬ�����ַ������ص�
            if (str[i - 1] == str[j - 1] && dp[i - 1][j - 1] < (j - i)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                // ��������Ⱥͽ�������
                if (dp[i][j] > maxLength) {
                    maxLength = dp[i][j];
                    endIndex = i - 1;
                }
            }
        }
    }

    // ��ȡ��ظ����ص����ַ���
    if (maxLength > 0) {
        return str.substr(endIndex - maxLength + 1, maxLength);
    }
    return ""; // ���û���ظ����ַ��������ؿ��ַ���
}

int main() {
    string str1 ;
    while(cin >> str1){
    	cout << "Longest repeating non-overlapping substring: " << findLongest(str1) << endl;
	}
    return 0;
}

