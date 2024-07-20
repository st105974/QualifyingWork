# def alg196(n):
#     N = n
#     numbers = 0
#     transform = 0
#     while n != 0:
#         n = n // 10
#         numbers += 1
#
#     for i in range(numbers):
#         k = N % 10
#         N = N // 10
#         transform += k * 10 ** (numbers - 1)
#         numbers-=1
#     return transform
#
# print(alg196(115))
def alg196(n):
    if 0 < n < 10:
        return True
    else:
        n = str(n)
        sgn = 0
        for i in range(1, len(n)):
            if n[i] == n[len(n) - i]:
                sgn = 1
            else:
                sgn = 0
                return False

            if sgn == 1:
                return True
print(alg196(1))