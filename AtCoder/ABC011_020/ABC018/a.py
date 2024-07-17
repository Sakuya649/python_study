A = [int(input()) for _ in range(3)]
for i in A:
    print(3-sorted(A).index(i))