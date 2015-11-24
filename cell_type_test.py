# -*- coding=utf-8 -*-
# __author__ = 'xxq'

import matplotlib.pyplot as plt
import numpy as np
import pyNN.nest as sim

parameters = {
 u'E_L': 0.0,
 u'I_e': 0.9, # 用这个参数来表示leaky
 u'V_reset': 0.0,
 u'V_th': 0.5,
 u't_ref': .0,
}

sim.setup(timestep=01.0)
nt = sim.native_cell_type('iaf_psc_delta_xxq')
n = sim.Population(1, nt(**parameters))
s = sim.Population(1, sim.SpikeSourceArray())
s[0].spike_times = [10, 15, 20, 30, 40]
p = sim.Projection(s, n, sim.FromListConnector([(0, 0, 0.00025, 0.01)]))
# p1 = sim.Projection(n, n, sim.FromListConnector([(0, 0, 0.00025, 1.0)]))
n.record('V_m')
n.record('V_m')
sim.initialize(n, V_m=0.)
sim.run(128.0)

vtrace = n.get_data(clear=True).segments[0].filter(name='V_m')[0]
print p.get(['weight'], format='array')

plt.figure()
plt.plot(vtrace.times, vtrace, 'o')
plt.ylim([0, 0.6])
plt.show()

sim.end()