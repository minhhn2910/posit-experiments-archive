import numpy as np
a = np.loadtxt('/tmp/bitcount_log.txt', delimiter=',')
perc1 = sum(a.T[0])/float(sum(a.T[1]))
perc0  = 1- perc1
print ("perc 1 ", perc1)
print ("perc 0 ", perc0)
