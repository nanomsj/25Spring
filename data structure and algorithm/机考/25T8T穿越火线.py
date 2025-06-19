# 考试中完全没时间看，故直接复制助教学长给出的答案

import heapq


def solve():
    n, m, T = map(int, input().split())
    adj = [[] for _ in range(n + 1)]
    max_a = 0

    for _ in range(m):
        u, v, t, a = map(int, input().split())
        adj[u].append((v, t, a))
        adj[v].append((u, t, a))  # 无向图
        if a > max_a:
            max_a = a

    # 二分查找最小防护等级
    left = 0
    right = max_a
    answer = max_a

    def is_possible(M):
        dist = [[float('inf')] * 2 for _ in range(n + 1)]
        dist[1][0] = 0  # 未使用迷彩
        heap = [(0, 1, 0)]  # (time, node, used)

        while heap:
            time, u, used = heapq.heappop(heap)
            if u == n:
                return time <= T
            if time > dist[u][used]:
                continue

            for (v, t, a) in adj[u]:
                # 情况1：不使用迷彩（a <= M）
                if a <= M:
                    if dist[v][used] > time + t:
                        dist[v][used] = time + t
                        heapq.heappush(heap, (dist[v][used], v, used))

                # 情况2：使用迷彩（如果还有机会）
                if not used:
                    if dist[v][1] > time + t:
                        dist[v][1] = time + t
                        heapq.heappush(heap, (dist[v][1], v, 1))

        return False

    # 二分查找
    while left <= right:
        mid = (left + right) // 2
        if is_possible(mid):
            answer = mid
            right = mid - 1
        else:
            left = mid + 1

    print(answer)


solve()