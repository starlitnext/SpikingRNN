# -*- coding=utf-8 -*-
# __author__ = 'xxq'

import pyNN.nest as sim
import matplotlib.pyplot as plt

if __name__ == '__main__':
    params = {
        'VBits': 32,
        'VdecBits': 32,
        'Vdec_sig': 2**30,
        'V_th': 10000000,
        'wBits': 16,
        'wAccBits': 20,
        'wAccShift': 10,
        't_ref': 0.
    }
    sim.setup(timestep=0.1)
    cell_type = sim.native_cell_type('iaf_psc_delta_int')
    neuron = cell_type(**params)
    n = sim.Population(1, cell_type(**params))
    s = sim.Population(1, sim.SpikeSourceArray())
    o = sim.Population(1, cell_type(**params))
    s[0].spike_times = [10, 15, 20, 30, 40]
    p = sim.Projection(s, n, sim.FromListConnector([(0, 0, 25, 0.01)]))
    p1 = sim.Projection(n, o, sim.FromListConnector([(0, 0, 25, 0.01)]))
    o.record('V_m')
    sim.initialize(n, V_m=0)
    sim.run(128.0)

    vtrace = o.get_data(clear=True).segments[0].filter(name='V_m')[0]
    print p.get(['weight'], format='array')

    plt.figure()
    plt.plot(vtrace.times, vtrace, 'o')
    # plt.ylim([0, 0.6])
    plt.show()

    sim.end()


