from collections import deque

def Joseph(n, m):
    circle = deque(range(1, n + 1))
    direction = 1  # 1 counter-clockwise; -1 clockwise
    step = m
    multi = 1

    while len(circle) > 1:
        if direction == 1:
            circle.rotate(-(step - 1))
        else:
            circle.rotate(step)
        circle.popleft()
        direction *= -1
        multi *= 2
        step = m * multi
        # print(circle)

    return circle[0]

n, m = map(int, input().split())
print(Joseph(n, m))