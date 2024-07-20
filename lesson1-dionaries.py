import random
a = [1, 2 , 3, 4 ,5]
a[0] = 6
print(a[0])
a1 = list()
a2 = list()
a3 = list()
for i in range(1,6):
    a1.append(random.randint(1,5))
    a2.append(random.randint(1,5))
    a3.append(random.randint(1, 5))
print(a1, a2, a3)
a4 = a1 + a2 + a3
print(a4)
a4 = list(set(a4))
print(a4)
b1 = {'agility': a1,
      'intelect': a2,
      'strong': a3}
b2 = {1: 'dick'}
b1['new'] = b2
b1['new']['new'] = {'universe': 'spbu'}
print(b1)
govnoFROMslovar = b1.get('new')
print(govnoFROMslovar)
listD = list(b1.values())
print(listD)
b1['new'] = None
print(b1)
