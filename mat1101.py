import numpy as np
import decimal
from decimal import Decimal


class trans:
    def __init__(self, mat: np.ndarray):
        self.mat = mat
        self.n = mat.shape[0]
        self.type = type(mat[0, 0])
        self.trans2 = False  # 是否进行变换2
        self.a = None

    def __trans1(self):
        n = self.n
        row_sum = self.mat.sum(axis=1)

        if self.type == decimal.Decimal:
            delta = Decimal(0.5)
        else:
            delta = 0.5

        for i in range(n):
            for j in range(n):
                self.mat[i, j] = delta + (row_sum[i] - row_sum[j]) / n

    def __trans2(self):
        self.mat = self.mat + self.a

    def transform(self):
        self.__trans1()
        a = self.mat.min()
        if a > 0:
            return
        self.a = abs(a)
        self.trans2 = True
        self.__trans2()

    def validate(self) -> bool:
        n = self.n

        if self.type == decimal.Decimal:
            delta = Decimal(0.5)
            mat = np.array(list((Decimal(0) for i in range(n))
                           for j in range(n)))
        else:
            delta = 0.5
            mat = np.zeros(self.mat.shape)
        if self.trans2:
            delta *= (1 + 2 * self.a)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    mat[i, j] = self.mat[i, k] + \
                        self.mat[k, j] - self.mat[i, j]
        print(mat)
        print(delta)

    def _validate(self):
        n = self.n

        zero = np.zeros((n, n, n))

        if self.type == decimal.Decimal:
            delta = Decimal(0.5)
            mat = np.array([[Decimal(0) for _ in range(n)] for _ in range(n)])
        else:
            delta = 0.5
            mat = np.zeros((n, n, n))
        if self.trans2:
            delta *= (1 + 2 * self.a)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    mat[i, j, k] = self.mat[i, k] + \
                        self.mat[k, j] - self.mat[i, j] - delta
        # print(mat)
        print(sum(mat))
        return np.isclose(zero, mat)

    def show(self):
        print(self.mat)


R1 = np.array([
    [Decimal('0.5'), Decimal('1.0'), Decimal('1.0'), Decimal('1.0')],
    [Decimal('0.0'), Decimal('0.5'), Decimal('0.6'), Decimal('0.4')],
    [Decimal('0.0'), Decimal('0.4'), Decimal('0.5'), Decimal('1.0')],
    [Decimal('0.0'), Decimal('0.6'), Decimal('0.0'), Decimal('0.5')],
])

R2 = np.array([
    [Decimal('0.5'), Decimal('1.0'), Decimal('1.0')],
    [Decimal('0.0'), Decimal('0.5'), Decimal('0.4')],
    [Decimal('0.0'), Decimal('0.6'), Decimal('0.5')],
])

R3 = np.array([
    [Decimal('0.5'), Decimal('0.1'), Decimal('0.6'), Decimal('0.7')],
    [Decimal('0.9'), Decimal('0.5'), Decimal('0.8'), Decimal('0.4')],
    [Decimal('0.4'), Decimal('0.2'), Decimal('0.5'), Decimal('0.9')],
    [Decimal('0.3'), Decimal('0.6'), Decimal('0.1'), Decimal('0.5')],
])

R = (R1, R2, R3)

r1 = np.array([
    [0.5, 1.0, 1.0, 1.0],
    [0.0, 0.5, 0.6, 0.4],
    [0.0, 0.4, 0.5, 1.0],
    [0.0, 0.6, 0.0, 0.5],
])

r2 = np.array([
    [0.5, 1.0, 1.0],
    [0.0, 0.5, 0.4],
    [0.0, 0.6, 0.5],
])

r3 = np.array([
    [0.5, 0.1, 0.6, 0.7],
    [0.9, 0.5, 0.8, 0.4],
    [0.4, 0.2, 0.5, 0.9],
    [0.3, 0.6, 0.1, 0.5],
])

r = (r1, r2, r3)

for mat in r:
    m = trans(mat)
    m.transform()
    print(m._validate())
