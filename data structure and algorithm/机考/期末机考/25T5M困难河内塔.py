def isnext(x,y):
    if (x=='A' and y=='B') or (x=='B' and y=='A') or (x=='B' and y=='C') or (x =='C' and y=='B'):
        return True
    return False

def middle(x,y):
    if x==y:
        return x
    elif (x=='A' and y=='B') or (x=='B' and y=='A'):
        return 'C'
    elif (x=='B' and y=='C') or (x =='C' and y=='B'):
        return 'A'
    elif (x=='A' and y=='C') or (x == 'C' and y=='A'):
        return 'B'

def hanoi(n,start,end,medium,golden):
    if n==1:
        if golden[0]==1:
            if isnext(start,end):
                print(f"moving disk 1 from {start} to {end}")
            else:
                mid=middle(start,end)
                print(f"moving disk 1 from {start} to {mid}")
                print(f"moving disk 1 from {mid} to {end}")
        else:
            print(f"moving disk 1 from {start} to {end}")
    else:
        if golden[n-1]==1 and not isnext(start,end):
            hanoi(n-1,start,end,medium,golden)
            print(f"moving disk {n} from {start} to {medium}")
            hanoi(n-1, end, start, medium, golden)
            print(f"moving disk {n} from {medium} to {end}")
            hanoi(n-1,start,end,medium,golden)
        else:
            hanoi(n-1,start,medium,end,golden)
            print(f"moving disk {n} from {start} to {end}")
            hanoi(n-1,medium,end,start,golden)

def main():
    n = int(input())
    golden = input().split()
    for i in range(n):
        golden[i]=int(golden[i])
    hanoi(n, 'A', 'C', 'B', golden)

main()