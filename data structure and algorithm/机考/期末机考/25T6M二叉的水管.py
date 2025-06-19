# 考场上根据提示看出来了是拓扑排序，但是树的部分不会写
# 上传助教学长给出的答案

import sys
from collections import defaultdict, deque
def topological_sort(n, edges):
    """正确的拓扑排序实现（节点编号1~n）"""
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    # 构建图：A > B 表示为 A -> B
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # 初始化所有节点的入度（1~n）
    for node in range(1, n + 1):
        if node not in in_degree:
            in_degree[node] = 0

    queue = deque()
    # 检查初始状态
    initial_zero = [node for node in range(1, n + 1) if in_degree[node] == 0]

    if not initial_zero:
        return "Device error."  # 没有入度为0的节点，存在环
    if len(initial_zero) > 1:
        return "Not determined."  # 多个起点，顺序不唯一
    queue.extend(initial_zero)
    topo_order = []
    while queue:
        if len(queue) > 1:
            return "Not determined."  # 处理过程中出现多个选择
        u = queue.popleft()
        topo_order.append(u)

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return topo_order if len(topo_order) == n else "Device error."

def get_inorder(topo_result, n):
    # 生成标准完全二叉树的后序编号序列
    def build_std_postorder(n):
        post_order = []

        def build(pos):
            if pos > n:
                return
            build(2 * pos)  # 左子树
            build(2 * pos + 1)  # 右子树
            post_order.append(pos)

        build(1)
        return post_order

    # 生成标准完全二叉树的中序编号序列
    def build_std_inorder(n):
        in_order = []

        def build(pos):
            if pos > n:
                return
            build(2 * pos)  # 左子树
            in_order.append(pos)
            build(2 * pos + 1)  # 右子树

        build(1)
        return in_order

    # 建立编号到值的映射
    std_post = build_std_postorder(n)
    value_map = {pos: val for pos, val in zip(std_post, topo_result)}

    # 获取中序编号序列并转换为值
    std_in = build_std_inorder(n)
    return [value_map[pos] for pos in std_in]

def main():
    data = sys.stdin.read().split('\n')
    ptr = 0
    while ptr < len(data):
        while ptr < len(data) and data[ptr].strip() == '':
            ptr += 1
        if ptr >= len(data):
            break

        # 读取n和m
        first_line = data[ptr].strip().split()
        if len(first_line) < 2:
            ptr += 1
            continue
        n, m = map(int, first_line[:2])
        ptr += 1

        # 读取所有比较关系
        edges = []
        for _ in range(m):
            while ptr < len(data) and data[ptr].strip() == '':
                ptr += 1
            if ptr >= len(data):
                break
            parts = data[ptr].strip().split()
            ptr += 1
            if len(parts) == 3 and parts[1] == '>':
                u, _, v = parts
                edges.append((int(u), int(v)))
        # 执行拓扑排序
        topo_result = topological_sort(n, edges)

        if isinstance(topo_result, str):
            print(topo_result)
            return
        topo_result=reversed(topo_result)
        # 生成标准中序并映射
        result = get_inorder(topo_result, n)
        print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()