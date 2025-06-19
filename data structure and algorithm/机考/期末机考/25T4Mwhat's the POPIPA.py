def count(s):
    s = s.replace(' ', '')

    po = 0
    pi = 0
    pa = 0

    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i + 2] == "PO":
            po += 1
            i += 2
        elif i + 1 < len(s) and s[i:i + 2] == "PI":
            pi += po
            i += 2
        elif i + 1 < len(s) and s[i:i + 2] == "PA":
            pa += pi
            i += 2

    return pa


while True:
    try:
        s = input()
        print(count(s))
    except EOFError:
        break