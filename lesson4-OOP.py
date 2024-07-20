class Polynomial:
    f: list
    __deg: int

    def __init__(self, f_deg):
        self.f = list()
        self.__deg = f_deg + 1
        for i1 in range(0, self.__deg):
            self.f.append(0)

    def out(self):
        print(self.f)

    def __getitem__(self, index):
        return self.f[index]

    def __setitem__(self, index, k):
        self.f[index] = k

    def __add__(self, other):
        if self.__deg == other.deg:
            a = Polynomial(self.__deg - 1)
            for i2 in range(0, self.__deg):
                a[i2] = self.f[i2] + other.f[i2]
            return a

    def value(self, x):
        p = self.f[self.__deg - 1]
        for i3 in range(self.__deg - 2, -1, -1):
            p *= x
            p += self.f[i3]
        return p

    def get_deg(self):
        return self.__deg

    def set_deg(self, f_deg: int):
        if type(f_deg) == int:
            self.__deg = f_deg + 1
        #for i4 in range ()


class Quadratic(Polynomial):
    def __init__(self, f_deg=2):
        super().__init__(f_deg=2)

    def value(self, x):
        return super().value(x)



f = Polynomial(3)
g = Quadratic()
f.set_deg()
print(f.get_deg())
f.out()
