full_bit =[]
flipped_bit= []
total_bit =[]
import numpy as np

f = open("/tmp/asymmetric_log.txt", "r")
for x in f:
  arr = x.split(",")
  if (len(arr)!= 3) :
      continue
  full_bit.append(int(arr[0]))
  flipped_bit.append(int(arr[1]))
  total_bit.append(32*int(arr[2]))

print ("full_bit ", sum(full_bit))
print ("flipped_bit ", sum(flipped_bit))
print ("total_bit ", sum(total_bit))
print ("BER ", float(sum(flipped_bit))/sum(total_bit))
