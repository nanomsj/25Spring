# 去年机考题目，直接复制粘贴即可。

import re

def infix_to_postfix(tokens):
    """
    中缀（tokens）→ 后缀：
    - 使用栈存运算符，根据优先级（+：1，*：2）和左结合性来出栈与入栈。
    - 数字直接输出到 postfixList，遇到 '(' 入栈，遇到 ')' 则依次弹出运算符直到遇到 '('。
    """
    op_stack = []
    postfix = []
    prec = {"+": 1, "*": 2}

    for tk in tokens:
        if tk.isdigit():
            postfix.append(tk)
        elif tk == "(":
            op_stack.append(tk)
        elif tk == ")":
            while op_stack and op_stack[-1] != "(":
                postfix.append(op_stack.pop())
            op_stack.pop()  # 弹出 '('
        else:
            while (op_stack and op_stack[-1] != "("
                   and prec[tk] <= prec[op_stack[-1]]):
                postfix.append(op_stack.pop())
            op_stack.append(tk)

    while op_stack:
        postfix.append(op_stack.pop())

    return postfix


def postfix_to_infix(postfix):

    stack = []
    for tk in postfix:
        if tk.isdigit():
            # 数字本身优先级设为 3
            stack.append((tk, 3))
        else:
            # tk 是 '+' 或 '*'
            right_str, right_prec = stack.pop()
            left_str, left_prec = stack.pop()
            prec_op = 2 if tk == "*" else 1

            if left_prec < prec_op:
                left_str = f"({left_str})"

            if right_prec < prec_op or right_prec == prec_op:
                right_str = f"({right_str})"

            merged = f"{left_str}{tk}{right_str}"
            stack.append((merged, prec_op))

    return stack.pop()[0]


def simple(expr):
    tokens = re.findall(r'\d+|[()+*]', expr)
    postfix = infix_to_postfix(tokens)
    return postfix_to_infix(postfix)


if __name__ == "__main__":
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(simple(line))