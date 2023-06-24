import math

loss_in_nats = 1.28
loss_in_bits = loss_in_nats / math.log(2)
BPB = loss_in_bits / 8
print("BPB:", BPB)