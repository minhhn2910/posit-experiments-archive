full_bit =[]
flipped_bit= []
total_bit =[]
import numpy as np
import sys
def main (weight_layers = 0, batches = 0):
    print ("float 32 data")
    f = open("/tmp/asymmetric_log.txt", "r")
    for x in f:
        arr = x.split(",")
        if (len(arr)!= 3) :
            continue
        full_bit.append(int(arr[0]))
        flipped_bit.append(int(arr[1]))
        total_bit.append(32*int(arr[2]))
    total_len = len(full_bit)
    weight_layers = int(weight_layers)
    batches = int(batches)
    print ("total len ",total_len)
    print ("weight layers ", weight_layers)
    print (" batches ", batches)

    print ("full_bit ", sum(full_bit))
    print ("flipped_bit ", sum(flipped_bit))
    print ("total_bit ", sum(total_bit))
    print ("BER ", float(sum(flipped_bit))/sum(total_bit))
    print ("BER total (total bits)", float(sum(full_bit))/sum(total_bit))
    batch_act = int ((total_len - weight_layers)/batches)

    full_bit_1batch = weight_layers+batch_act
    print ("------------")
    print ("full_bit 1batch ", sum(full_bit[:full_bit_1batch]))
    print ("flipped_bit 1batch", sum(flipped_bit[:full_bit_1batch]))
    print ("total_bit 1batch", sum(total_bit[:full_bit_1batch]))
    print ("BER 1batch", float(sum(flipped_bit[:full_bit_1batch]))/sum(total_bit[:full_bit_1batch]))
    print ("BER 1batch (total bits)", float(sum(full_bit[:full_bit_1batch]))/sum(total_bit[:full_bit_1batch]))
    weight_only = weight_layers
    print ("------------")
    print ("full_bit weight ", sum(full_bit[:weight_layers]))
    print ("flipped_bit weight", sum(flipped_bit[:weight_layers]))
    print ("total_bit weight", sum(total_bit[:weight_layers]))
    print ("BER weight", float(sum(flipped_bit[:weight_layers]))/sum(total_bit[:weight_layers]))
    print ("BER weight(total bits)", float(sum(full_bit[:weight_layers]))/sum(total_bit[:weight_layers]))

if __name__ == '__main__':
    arguments = sys.argv[1:]
    if (len(arguments)!=2):
        print ("input number of layers and number of batches ")
    main(arguments[0], arguments[1])
