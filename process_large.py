# TODO remember the mutation and the SBX is [-1,1]
import math
import random
from random import choice
import pickle
import matplotlib.pyplot as plt
import numpy as np

obj_num = 3


def ob1(pop):
    return pop[0] ** 4 - 10 * pop[0] ** 2 + pop[0] * pop[1] + pop[1] ** 4 - (pop[0] ** 2) * (pop[1] ** 2)


def ob2(pop):
    return pop[1] ** 2 - (pop[0] ** 2) * (pop[1] ** 2) + pop[0] ** 4 + pop[0] * pop[1]


def init_pop():
    return ((-1 + 2 * np.random.random((1000, 2))) * 5).tolist()


def SBX(pop):
    '''
    in this algorithm the pop has become a list from np.array
    :param pop:
    :return:
    '''
    ui = np.random.random()
    rj = 0
    if ui > 0.5:
        rj = (1 / (2 * (1 - ui))) ** (1 / 16)
    if ui <= 0.5:
        rj = (2 * ui) ** (1 / 16)

    random.shuffle(pop)

    pop2 = []
    for i in list(range(0, len(pop), 2)):
        x1 = pop[i]
        x2 = pop[i + 1]
        X1 = []
        X2 = []
        for num in range(2):
            X1.append(0.5 * ((1 + rj) * x1[num] + (1 - rj) * x2[num]))
            X2.append(0.5 * ((1 - rj) * x1[num] + (1 + rj) * x2[num]))
        pop2.append(X1)
        pop2.append(X2)

    return pop2


def mutation(pop2):
    ui = np.random.random()
    pi = 0
    if ui < 0.5:
        pi = (2 * ui) ** (1 / 21) - 1
    if ui >= 0.5:
        pi = 1 - (2 * (1 - ui)) ** (1 / 21)

    num = random.choice(range(0, 1000))
    x = pop2[num]
    pop2.pop(num)
    x = [a + pi for a in x]
    pop2.append(x)

    return pop2


def non_dominated_sort(popR):
    '''

    :param popR:
    :return:
    '''
    popv = []
    h = []
    d = {}
    f = []
    all = []
    temp = []
    for index in popR:
        popv.append([ob1(index), ob2(index)])

    while sum([len(x) for x in all]) < 1000:
        if h == []:
            h = range(1000)

        for n in h:
            num = 0
            s = []
            for m in h:
                if m == n:
                    continue
                if popv[n][0] > popv[m][0] and popv[n][1] > popv[m][1]:
                    num += 1
                if popv[n][0] < popv[m][0] and popv[n][1] < popv[m][1]:
                    s.append(m)

            d[n] = [num, s]

        for k, v in d.items():
            if v[0] == 0:
                f.append(k)
                temp = temp + v[1]

        all.append(f)
        f = []
        h = list(set(temp))
        temp = []
        d = {}

    return all, popv


def normalize(pop5, v):
    temp = []
    for index in pop5:
        temp += index
    min_x1 = 100000
    min_x2 = 100000
    for index in temp:
        if v[index][0] < min_x1:
            min_x1 = v[index][0]
        if v[index][1] < min_x2:
            min_x2 = v[index][1]
    v2 = [[[x[0] - min_x1, x[1] - min_x2] for x in v][x] for x in temp]
    num_min_1 = max(v2[0][0], v2[0][1] * 10 ** 6)
    aim_1 = 0
    for n in range(len(v2)):
        t_1 = max(v2[n][0], v2[n][1] * 10 ** 6)
        if t_1 < num_min_1:
            num_min_1 = t_1
            aim_1 = n
    num_min_2 = max(v2[0][0] * 10 ** 6, v2[0][1])
    aim_2 = 0
    for n in range(len(v2)):
        t_2 = max(v2[n][0] * 10 ** 6, v2[n][1])
        if t_2 < num_min_2:
            num_min_2 = t_2
            aim_2 = n
    a1 = v2[aim_1][0]
    a2 = v2[aim_2][1]
    v3 = [[x[0] / (a1 - min_x1), x[1] / (a2 - min_x2)] for x in v2]

    return v3, temp, len(pop5[-1])


def associate(pop, t, number):
    line = [
        [1, 0, 0],
        [3, -1, 0],
        [1, -1, 0],
        [1, -3, 0],
        [0, 1, 0]
    ]
    d1 = {}
    d2 = {}

    for num in range(5):
        d1[num] = []
        d2[num] = []

    pop1 = pop[:len(pop) - number]
    pop2 = pop[len(pop) - number:]
    t1 = t[:len(t) - number]
    t2 = t[len(t) - number:]

    for n in range(len(pop1)):
        small = 0
        dis = abs(pop1[n][0] * line[0][0] + pop1[n][1] * line[0][1] + line[0][2]) / (
            math.sqrt(line[0][0] ** 2 + line[0][1] ** 2))
        for m in range(1, 5):
            now = abs(pop1[n][0] * line[m][0] + pop1[n][1] * line[m][1] + line[m][2]) / (
                math.sqrt(line[m][0] ** 2 + line[m][1] ** 2))
            if now < dis:
                small = m
                dis = now

        d1[small].append(t1[n])

    for n in range(len(pop2)):
        small = 0
        dis = abs(pop2[n][0] * line[0][0] + pop2[n][1] * line[0][1] + line[0][2]) / (
            math.sqrt(line[0][0] ** 2 + line[0][1] ** 2))
        for m in range(1, 5):
            now = abs(pop2[n][0] * line[m][0] + pop2[n][1] * line[m][1] + line[m][2]) / (
                math.sqrt(line[m][0] ** 2 + line[m][1] ** 2))
            if now < dis:
                small = m
                dis = now

        d2[small].append(t2[n])

    return d1, d2


def nich(d1, d2, t, number, vv):
    line = [
        [1, 0, 0],
        [3, -1, 0],
        [1, -1, 0],
        [1, -3, 0],
        [0, 1, 0]
    ]
    q = t[:len(t) - number]
    while len(q) < 1000:
        l = [len(x) for x in d1.values()]
        ll = []
        for k,v in d1.items():
            if len(v) == min(l):
                ll.append(k)
        little = choice(ll)

        if len(d1[little]) == 0:
            print(d1)
            print(d2)

            if len(d2[little]) == 0:
                del d1[little]
                del d2[little]

            if len(d2[little]) > 0:
                small = d2[little][0]
                dis = 1000
                for n in d2[little]:
                    pos = vv[temp.index(n)]
                    dis_now = abs(pos[0] * line[0][0] + pos[1] * line[0][1] + line[0][2]) / (
                        math.sqrt(line[0][0] ** 2 + line[0][1] ** 2))
                    if dis_now < dis:
                        dis = dis_now
                        small = n

                q.append(small)
                d1[little].append(small)
                d2[little].remove(small)

        if len(d1[little]) > 0:
            if d2[little] !=[]:
                c = choice(d2[little])
                q.append(c)
                d1[little].append(c)
                d2[little].remove(c)
            else:
                del d1[little]
                del d2[little]

    return q


if __name__ == '__main__':
    pop = init_pop()  # x1 x2
    # file = open('data.txt', 'w')
    # file.write(str(pop))
    # file.close()
    #
    # f = open('data.txt')
    # ss = f.read()[2:-2]
    # pop = [[float(x.split(',')[0]),float(x.split(',')[1])] for x in ss.split('], [')]

    iteration = 0
    while iteration < 100:
        pop2 = SBX(pop)
        pop3 = mutation(pop2)
        pop4 = pop3 + pop

        pop5, popv = non_dominated_sort(pop4)

        q = []
        if sum([len(x) for x in pop5]) > 1000:
            pop6, temp, num = normalize(pop5, popv)

            d1, d2 = associate(pop6, temp, num)

            q = nich(d1, d2, temp, num, pop6)

        if sum([len(x) for x in pop5]) == 1000:
            q= []
            for index in pop5:
                q +=index

        pt2 = []
        for index in q:
            pt2.append(pop4[index])

        iteration +=1
        print('this is the '+str(iteration)+' generation')

    result = []
    pos_r = []
    for index in pop5[0]:
        result.append(popv[index])
        pos_r.append(pop4[index])


    plt.scatter([x[0] for x in result], [x[1] for x in result])
    plt.show()




