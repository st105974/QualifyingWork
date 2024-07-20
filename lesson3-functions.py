def printed(a: list):
    for i in range(0, len(a)):
        print(a[i])


b = [2, 1, 3, 6, 5, 4]


def bubble_sort(a: list):
    k = 0
    for i in range(0, len(a)):
        for j in range(i + 1, len(a)):
            if a[i] >= a[j]:
                k = a[i]
                a[i] = a[j]
                a[j] = k
    return a


printed(bubble_sort(b))


def FIO(name, otch, surname='Ivanov'):
    print(name, otch,  surname)


FIO(name='Ivan', otch="Ivanovich")
