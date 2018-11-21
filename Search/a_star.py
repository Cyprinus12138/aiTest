def init_list(length):
    tmp = []
    for i in range(length):
        tmp.append(0)
    return tmp


m = 34
solv = [init_list(4), init_list(4), init_list(4), init_list(4)]
used = init_list(17)
x = init_list(4)
y = init_list(4)
z = init_list(3)
c = 16
track_file = open('a_star_track.txt', 'w')


def a_star(r, c=c):
    k = r
    p = int(k / 4)
    q = k % 4
    if r >= c:
        print(solv)
        c = 0
        exit(0)
    else:
        l = sorted([i for i in range(1, c + 1)], key=lambda it: abs(it - ((m - x[p]) / (4 - p) + (m - y[q]) / (4 - q))), reverse=False)
        for i in l:
            if used[i] == 1:
                continue
            solv[p][q] = i
            used[i] = 1
            track_file.write(str(solv) + "\n")
            x[p] += i
            y[q] += i
            if p == q:
                z[1] += i
            if (p+q) == 3:
                z[2] += i
            if (k in [3, 7, 11]) and (x[p] != m):
                _ = 0
            elif (k in [12, 13, 14]) and (y[q] != m):
                _ = 0
            elif k == 9 and z[2] != m:
                _ = 0
            elif k == 15 and not (m in [y[q], x[p], z[1]]):
                _ = 0
            else:
                a_star(r + 1)
            if p == q:
                z[1] -= i
            if (p+q) == 3:
                z[2] -= i
            used[i] = 0
            x[p] -= i
            y[q] -= i


if __name__ == "__main__":
    a_star(0)
