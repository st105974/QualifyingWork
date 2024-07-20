a = [1, 2, 3, 4, 'govno']
b = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
n = 1
d1 = {'first key': 2456,
      'second key': 17890083}
d2 = {'third key' : 20040806,
      'fourth key' : 20041611}
for key in d1.keys():
    d2[key] = d1[key]
print(type(a))
print(d2)
#for element in b:
#    for number in element:
 #       print(number)
#for key in d1.keys():
 #   print(key, d1[key])
#while 1 != 0:
#    print(n)
#    n += 1
#    if (n > 10000):
#        break
for one_element in range(1, 4):
    for j in range (0, 10):
        print('one_element =', str(one_element))
        print("hello", j)