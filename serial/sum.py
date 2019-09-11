import numpy

data = np.fromfile('../Data_noise_free.bin', dtype=np.float32)
data = data.reshape(2000, 20000)

receivers = np.fromfile('../Receivers_Array.bin', dtype=np.float32)
receivers = receivers.reshape(2000, 3)

dt = 0.002
vv = 3000

i = 0
j = 0
k = 0

area_coord = [[0 for i in xrange(1000)] for j in xrange(3)]

x0 = -1000
x1 = 1000
y0 = -1000
y1 = 1000
z0 = 500
z1 = 2500

d = (x1-x0)/9

while i < 10:
	while j < 10:
		while k < 10:
			area_coord[i*100+j*10+k][0] = x0+d*i
			area_coord[i*100+j*10+k][1] = y0+d*j
			area_coord[i*100+j*10+k][2] = z0+d*k
			k+=1
		j+=1
	i+=1