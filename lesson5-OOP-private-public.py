class Human:
    __name: str
    __surname: str
    __sex: bool
    __age: int

    def __init__(self, name='Ivan', surname='Ivanov', sex=bool(), age=int()):
        self.__name = name
        self.__sex = sex
        self.__age = age
        self.__surname = surname

    def get_age(self):
        return self.__age

    def get_sex(self):
        return self.__sex

    def get_name(self):
        return self.__name

    def get_surname(self):
        return self.__name

    def set_name(self, name):
        if type(name) == str:
            self.__name = name


class Child(Human):
    pass


class Bus:
    __passengers: list
    __seats: int
    __x: int
    __y: int

    def __init__(self, seats, x, passengers: list):
        self.__seats = seats
        self.__x = x
        self.__passengers = passengers
        for i in range(seats):
            self.__passengers.append(0)

    def go(self, x0):
        if self.__x >= x0:
            self.__x = self.__x - x0
        else:
            self.__x = self.__x + x0

    def __setitem__(self, index, value):
        self.__passengers[index] = value

    def __getitem__(self, index):
        return self.__passengers[index]

    def printed(self):
        print(self.__passengers)

    def enter(self, children):
        available_seat = 0
        for i in self.__passengers:
            if i == 0:
                available_seat += 1
        if available_seat < children:
            for i in range(available_seat):
                self.__passengers[i] =
        else:


    def exit(self, children):
        for i in range(self.__seats):
            if

