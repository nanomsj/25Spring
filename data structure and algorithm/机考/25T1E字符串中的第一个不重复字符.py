s=input()
d=dict()
for i in s:
    if i in d:
        d[i]=d[i]+1
    else:
        d[i]=1
found=False
for k in d:
    if d[k]==1:
        print(s.index(k))
        found=True
        break
    else:
        continue
if found==False:
    print(-1)