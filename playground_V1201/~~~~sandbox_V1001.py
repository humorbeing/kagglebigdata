# a = [i for i in range(100)]
#
# b = a[80::3]
# print(a)
# print(b)
import shelve
d = shelve.open('f')
import dbm
print(dbm.whichdb('f'))