
def init_list(length):
    tmp = []
    for i in range(length):
        tmp.append(0)
    return tmp


class Statu:
    def __init__(self, statu, k, used, x, y, z):
        self.statu = statu
        self.k = k
        self.used = used
        self.x = x
        self.y = y
        self.z = z

m = 34
solv = [init_list(4), init_list(4), init_list(4), init_list(4)]
used = init_list(17)
x = init_list(4)
y = init_list(4)
z = init_list(3)
c = 16
status = []


def bfs(sta):
    sta = sta
    if sta.statu[3][3] != 0:
        print(sta.statu)
        exit(0)
    del status[0]
    k = sta.k
    for i in range(1, 17):
        p = int(k / 4)
        q = k % 4
        if sta.used[i] == 1:
            continue
        else:
            tmp = sta
            sta.statu[p][q] = i
            tmp.used[i] = 1
            tmp.k += 1
            tmp.x[p] += i
            tmp.y[q] += i
            if p == q:
                tmp.z[1] += i
            if (p+q) == 3:
                tmp.z[2] += i
            if (tmp.k in [3, 7, 11]) and (tmp.x[p] != m):
                _ = 0
            elif (tmp.k in [12, 13, 14]) and (tmp.y[q] != m):
                _ = 0
            elif tmp.k == 9 and tmp.z[2] != m:
                _ = 0
            elif tmp.k == 15 and not (m in [tmp.y[q], tmp.x[p], tmp.z[1]]):
                _ = 0
            else:
                status.append(tmp)
    bfs(status[0])


if __name__ == "__main__":
    sta = Statu(solv, 0, used, x, y, z)
    status.append(sta)
    bfs(status[0])