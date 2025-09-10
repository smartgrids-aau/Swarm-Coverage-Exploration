message = bytearray(4)
idd = 254
message[0] = 0b00000000
message[2] = idd
print(message)