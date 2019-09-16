import CoherentSumModule
import numpy as np
import math
import plotly.graph_objects as go

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def calc_radius(x, y, z):
	return math.sqrt(x*x+y*y+z*z)

data = np.fromfile('../Data_noise_free.bin', dtype=np.float32)
data = data.reshape(2000, 10000)

receivers = np.fromfile('../Receivers_Array.bin', dtype=np.float32)

receivers = receivers.reshape(2000, 3)

dt = 0.002
vv = 3000

sources = np.arange(1000*3, dtype=np.float32)
sources = sources.reshape(1000, 3)

x0 = -1000
x1 = 1000
y0 = -1000
y1 = 1000
z0 = 500
z1 = 2500

d = (x1-x0)/9
print(d)

i = 0
while i < 10:
	j = 0
	while j < 10:
		k = 0
		while k < 10:
			sources[i*100+j*10+k][0] = x0+d*j
			sources[i*100+j*10+k][1] = y0+d*k
			sources[i*100+j*10+k][2] = z0+d*i
			k+=1
		j+=1
	i+=1

sources_times = np.arange(1000*2000, dtype=np.int32)
sources_times = sources_times.reshape(1000, 2000)

i = 0
while i < 1000:
	sources_times[i][0] = int(round_half_up(calc_radius(sources[i][0]-receivers[0][0], 
														sources[i][1]-receivers[0][1],
														sources[i][2]-receivers[0][2])
														/(vv*dt))+1)
	min_ind = int(sources_times[i][0])
	m = 1
	while m < 2000:
		sources_times[i][m] = int(round_half_up(calc_radius(sources[i][0]-receivers[m][0], 
															sources[i][1]-receivers[m][1],
															sources[i][2]-receivers[m][2])
															/(vv*dt))+1)
		min_ind = min(min_ind, sources_times[i][m])
		m+=1
	m = 0
	while m < 2000:
		sources_times[i][m] -= min_ind
		m+=1
	i+=1
	
result = CoherentSumModule.computeCoherentSummation(data, receivers, sources_times)
result = result.reshape(1000*10000)

another_results = np.fromfile('../Summation_Results2.bin', dtype=np.float32)

x = np.arange(1000*10000, dtype=np.int32)
i=0
while i < 10000000:
	another_results[i] = another_results[i]-result[i]
	x[i] = i
	i+=1

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=another_results, name='difference', connectgaps=True))
fig.show()