#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 检查是否可以将书分配给学生，使得每个学生阅读的页数不超过x。
bool check(const vector<int>& pages, int x, int M) {
    int students = 1; // 当前学生数量
    int sum = 0; // 当前学生阅读的页数总和
    for (int i = 0; i < pages.size(); i++) {
        if (sum + pages[i] > x) {
            // 如果当前学生阅读的页数加上当前书的页数超过x，则分配给下一个学生
            students++;
            sum = pages[i];
            if (students > M)
                // 如果学生数量超过M，则无法分配
                return false;
        }
        else 
            // 如果当前学生可以阅读当前书，则累加页数
            sum += pages[i];
    	
    }
    return true;
}

int findMinPages(vector<int> pages, int M) {
    int left = 0;
    int right = accumulate(pages.begin(), pages.end(), 0); // 所有书的总页数
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (check(pages, mid, M))
            // 如果可以分配，则尝试减小x
            right = mid;
        else
            // 如果不可以分配，则增加x
            left = mid + 1;
    }
    return left;
}

int main() {
    int N ;
    vector<int> pages = {};
    int M ;
    cin >> N;
    for (int i = 0; i < N; i++) {
        int temp;
        cin >> temp;
        pages.push_back(temp);
    }
    cin >> M;
    cout << "The minimum number of pages is: " << findMinPages(pages, M) << endl; 
    return 0;
}
