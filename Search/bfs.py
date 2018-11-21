import copy
import sys
sys.setrecursionlimit(10**9)

def init_list(length):
    tmp = []
    for i in range(length):
        tmp.append(0)
    return tmp


class Statu:
    def __init__(self, statu, k, used, x, y, z):
        self.statu = copy.deepcopy(statu)
        self.k = k
        self.used = copy.deepcopy(used)
        self.x = copy.copy(x)
        self.y = copy.copy(y)
        self.z = copy.copy(z)
        return

m = 34
solv = [init_list(4), init_list(4), init_list(4), init_list(4)]
used = init_list(17)
x = init_list(4)
y = init_list(4)
z = init_list(3)
c = 16
status = []


def bfs(sta):
    print(status[0].statu, status[0].k)
    if sta.k >= 15:
        print(sta.statu)
        exit(0)
    k = sta.k
    for i in range(1, 17):
        p = int(k / 4)
        q = k % 4
        if sta.used[i] == 1:
            continue
        else:
            tmp = Statu(sta.statu, sta.k, sta.used, sta.x, sta.y, sta.z)
            tmp.statu[p][q] = i
            tmp.used[i] = 1
            tmp.k += 1
            tmp.x[p] += i
            tmp.y[q] += i
            if p == q:
                tmp.z[1] += i
            if (p+q) == 3:
                tmp.z[2] += i
            if (tmp.k in [4, 8, 12]) and (tmp.x[p] != m):
                del tmp
            elif (tmp.k in [13, 14, 15]) and (tmp.y[q] != m):
                del tmp
            elif tmp.k == 10 and tmp.z[2] != m:
                del tmp
            elif tmp.k == 16 and not (m in [tmp.y[q], tmp.x[p], tmp.z[1]]):
                del tmp
            else:
                status.append(tmp)
    del status[0]
    bfs(status[0])


if __name__ == "__main__":
    sta = Statu(solv, 0, used, x, y, z)
    status.append(sta)
    bfs(status[0])
