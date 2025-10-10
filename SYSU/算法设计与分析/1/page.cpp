#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// ����Ƿ���Խ�������ѧ����ʹ��ÿ��ѧ���Ķ���ҳ��������x��
bool check(const vector<int>& pages, int x, int M) {
    int students = 1; // ��ǰѧ������
    int sum = 0; // ��ǰѧ���Ķ���ҳ���ܺ�
    for (int i = 0; i < pages.size(); i++) {
        if (sum + pages[i] > x) {
            // �����ǰѧ���Ķ���ҳ�����ϵ�ǰ���ҳ������x����������һ��ѧ��
            students++;
            sum = pages[i];
            if (students > M)
                // ���ѧ����������M�����޷�����
                return false;
        }
        else 
            // �����ǰѧ�������Ķ���ǰ�飬���ۼ�ҳ��
            sum += pages[i];
    	
    }
    return true;
}

int findMinPages(vector<int> pages, int M) {
    int left = 0;
    int right = accumulate(pages.begin(), pages.end(), 0); // ���������ҳ��
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (check(pages, mid, M))
            // ������Է��䣬���Լ�Сx
            right = mid;
        else
            // ��������Է��䣬������x
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
