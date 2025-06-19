#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
using namespace std;

#define long_type long long int  // 定义长整型别名

// 全局变量
unsigned long_type randomSeed = (unsigned)time(NULL);  // 用当前时间初始化随机种子
long_type N;  // 皇后数量
long_type *queens, *primaryDiagonal, *secondaryDiagonal, *primaryDiagonal1, *secondaryDiagonal1;
// queens: 存储每列皇后所在的行位置
// primaryDiagonal: 主对角线冲突计数数组(左上到右下)
// secondaryDiagonal: 副对角线冲突计数数组(右上到左下)
// primaryDiagonal1/secondaryDiagonal1: 辅助对角线冲突计数数组

/**
 * 生成指定范围内的随机数
 * @param start 随机数范围下限(包含)
 * @param end 随机数范围上限(不包含)
 * @return 生成的随机数
 */
unsigned long_type generateRandom(long_type start=0, long_type end=100) {
	unsigned long_type maxVal = 0x80000000ULL;  // 2^31
	randomSeed *= ((unsigned long_type)134775813);  // 随机种子更新
	randomSeed += 1;
	randomSeed = randomSeed % maxVal;
	double randomFraction = ((double)randomSeed) / (double)0x7fffffff;
	return end > start ? start + (unsigned long_type)((end - start) * randomFraction) : end;
}

/**
 * 交换两个皇后的位置，并更新相关的对角线冲突计数
 * @param a 第一个皇后的列索引
 * @param b 第二个皇后的列索引
 * @param flag 标志位: 0-减少辅助计数器,1-增加辅助计数器,2-不更新辅助计数器
 */
void swapQueens(long_type a, long_type b, int flag) {
	long_type temp;
	
	// 交换前先减少原位置的对角线计数
	primaryDiagonal[queens[a] - a + N - 1]--;
	secondaryDiagonal[queens[a] + a]--;
	primaryDiagonal[queens[b] - b + N - 1]--;
	secondaryDiagonal[queens[b] + b]--;
	
	// 根据标志位更新辅助计数器
	if (flag == 0) {
		primaryDiagonal1[queens[a] - a + N - 1]--;
		secondaryDiagonal1[queens[a] + a]--;
	}
	
	// 交换皇后位置
	temp = queens[a];
	queens[a] = queens[b];
	queens[b] = temp;
	
	// 交换后增加新位置的对角线计数
	primaryDiagonal[queens[a] - a + N - 1]++;
	secondaryDiagonal[queens[a] + a]++;
	primaryDiagonal[queens[b] - b + N - 1]++;
	secondaryDiagonal[queens[b] + b]++;
	
	// 根据标志位更新辅助计数器
	if (flag == 1) {
		primaryDiagonal1[queens[a] - a + N - 1]++;
		secondaryDiagonal1[queens[a] + a]++;
	}
}

/**
 * 检查指定列的皇后是否存在冲突
 * @param k 列索引
 * @return 1表示有冲突，0表示无冲突
 */
int countConflicts(long_type k) {
	return primaryDiagonal[queens[k] - k + N - 1] > 1 || secondaryDiagonal[queens[k] + k] > 1;
}

/**
 * 重置所有对角线冲突计数器为零
 */
void resetCounters() {
	for (long_type i = 0; i < 2 * N - 1; i++) {
		primaryDiagonal[i] = 0;
		secondaryDiagonal[i] = 0;
		primaryDiagonal1[i] = 0;
		secondaryDiagonal1[i] = 0;
	}
}

/**
 * 计算当前棋盘的总冲突数(启发式代价)
 * @return 总冲突数
 */
long_type calculateHeuristic() {
	resetCounters();
	long_type heuristicValue = 0;
	long_type maxDiagonal = 2 * N - 1;
	
	// 统计所有对角线上的皇后数量
	for (long_type i = 0; i < N; i++) {
		primaryDiagonal[queens[i] - i + N - 1]++;
		secondaryDiagonal[i + queens[i]]++;
	}
	
	// 计算总冲突数(每对角线上n个皇后产生n*(n-1)/2个冲突)
	for (long_type i = 0; i < maxDiagonal; i++) {
		heuristicValue += primaryDiagonal[i] * (primaryDiagonal[i] - 1) / 2;
		heuristicValue += secondaryDiagonal[i] * (secondaryDiagonal[i] - 1) / 2;
	}
	
	return heuristicValue;
}

/**
 * 初始化搜索:随机放置皇后并计算初始冲突数
 * @return 未预放置的皇后数量
 */
long_type initializeSearch() {
	resetCounters();
	long_type i, j, m;
	long_type maxIterations = N * 3.08;  // 最大迭代次数
	
	// 初始摆放:每列皇后放在对应行(对角线排列)
	for (i = 0; i < N; i++) {
		queens[i] = i;
		primaryDiagonal[queens[i] - i + N - 1]++;
		secondaryDiagonal[queens[i] + i]++;
	}
	
	// 随机交换并优化初始冲突数
	for (j = 0, i = 0; i < maxIterations && j < N; i++) {
		m = generateRandom(j, N);  // 随机选择交换位置
		swapQueens(j, m, 1);  // 交换并更新辅助计数器
		
		// 如果交换后仍有冲突，则撤销交换
		if (primaryDiagonal1[queens[j] - j + N - 1] > 1 || secondaryDiagonal1[queens[j] + j] > 1)
			swapQueens(j, m, 0);
		else
			j++;  // 无冲突则处理下一列
	}
	
	// 对剩余未处理的列进行完全随机化
	for (i = j; i < N; i++) {
		m = generateRandom(i, N);
		swapQueens(i, m, 2);  // 交换但不更新辅助计数器
	}
	
	cout << "Initialized: pre-positioned queens = " << j << ", initial conflicts = " << calculateHeuristic() << endl;
	return N - j;  // 返回未预放置的皇后数量
}

/**
 * 最终搜索:尝试消除剩余冲突
 * @param k 需要处理的皇后数量
 */
void finalSearch(long_type k) {
	long_type i, j;
	long_type b, c = 0;
	
	// 从第一个未预放置的皇后开始处理
	for (i = N - k - 1; i < N; i++) {
		c = 0;
		if (countConflicts(i)) {  // 如果当前皇后有冲突
			do {
				j = generateRandom(0, N);  // 随机选择交换位置
				c++;  // 尝试次数计数
				swapQueens(i, j, 2);  // 尝试交换
				b = (countConflicts(i) || countConflicts(j));  // 检查交换后是否有冲突
				
				// 如果交换后仍有冲突，则撤销交换
				if (b) swapQueens(i, j, 2);
				
				// 如果尝试次数过多，则重新初始化搜索
				if (c == 7000) {
					cout << "Restart triggered." << endl;
					k = initializeSearch();
					i = N - k - 1;
					break;
				}
			} while (b);  // 直到找到不产生冲突的交换
		}
	}
}

int main() {
	cout << "Input N: ";
	cin >> N;
	
	// 动态分配内存
	queens = (long_type *)malloc(sizeof(long_type) * N);
	primaryDiagonal = (long_type *)malloc(sizeof(long_type) * (2 * N - 1));
	secondaryDiagonal = (long_type *)malloc(sizeof(long_type) * (2 * N - 1));
	primaryDiagonal1 = (long_type *)malloc(sizeof(long_type) * (2 * N - 1));
	secondaryDiagonal1 = (long_type *)malloc(sizeof(long_type) * (2 * N - 1));
	
	// 初始化随机种子
	srand((unsigned)time(NULL) + rand());
	
	// 记录开始时间
	clock_t start = clock();
	
	// 执行搜索:先初始化，然后处理剩余冲突
	finalSearch(initializeSearch());
	
	// 记录结束时间
	clock_t finish = clock();
	double totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	
	// 输出结果
	cout << "Final conflicts: " << calculateHeuristic() << ", Time used: " << totalTime << " sec." << endl;
	
	// 释放内存
	free(queens);
	free(primaryDiagonal);
	free(secondaryDiagonal);
	free(primaryDiagonal1);
	free(secondaryDiagonal1);
	
	return 0;
}
