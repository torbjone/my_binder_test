#!/usr/bin/env python
'''
Cortical pyramidal cell model from Almog and Korngreen (2014) J Neurosci 34:1 182-196
Fast Spiking Basket cell from http://neuromorpho.org/neuroMorpho/neuron_info.jsp?neuron_name=FS-basket

This file is only supplimentary to the Ipython Notebook file with the same name. 
'''

import sys, os
import numpy as np
import pylab as plt
import scipy.fftpack as ff
import neuron
import LFPy

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def exercise1(soma_clamp_params, apic_clamp_params):

    cell_parameters = {
        'morphology': 'A140612.hoc',  # File with cell morphology
        'v_init': -62,
        'passive': False,
        'nsegs_method': None,
        'timeres_NEURON': 2**-3,  # [ms] Should be a power of 2
        'timeres_python': 2**-3,
        'tstartms': -50,  # [ms] Simulation start time
        'tstopms': 50,  # [ms] Simulation end time
        'custom_code': ['cell_model.hoc'] # Loads model specific code
    }

    ### MAKING THE CELL
    cell = LFPy.Cell(**cell_parameters)

    ### MAKING THE INPUT
    apic_clamp_params['idx'] = cell.get_closest_idx(x=-150., y=750., z=0.)
    cl_soma = LFPy.StimIntElectrode(cell, **soma_clamp_params)
    cl_apic = LFPy.StimIntElectrode(cell, **apic_clamp_params)

    cell.simulate(rec_imem=True, rec_vmem=True, rec_istim=True)

    ### PLOTTING THE RESULTS
    cell_plot_idxs = [soma_clamp_params['idx'], apic_clamp_params['idx']]    
    cell_plot_colors = {cell_plot_idxs[idx]: plt.cm.Greens_r(1./(len(cell_plot_idxs) + 1) * idx + 0.1) for idx in range(len(cell_plot_idxs))}


    # Plotting the morphology
    plt.figure(figsize=(16,9))
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(121, aspect='equal', xlabel='x [$\mu m$]', ylabel='y [$\mu m$]', xlim=[-400, 400], xticks=[-400, 0, 400], title='Green dots: Inputs')
    [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], c='k') for idx in xrange(cell.totnsegs)]
    [plt.plot(cell.xmid[idx], cell.ymid[idx], 'o', c=cell_plot_colors[idx], ms=12) for idx in cell_plot_idxs]

    # Plotting the membrane potentials
    plt.subplot(222, title='Membrane potential', xlabel='Time [ms]', ylabel='mV', ylim=[-80, 20])
    [plt.plot(cell.tvec, cell.vmem[idx, :], c=cell_plot_colors[idx], lw=2) for idx in cell_plot_idxs]

    # Plotting the input currents
    stim_lim = [2*np.min([cl_soma.i, cl_apic.i]), 2*np.max([cl_soma.i, cl_apic.i])]
    plt.subplot(224, title='Input currents', xlabel='Time [ms]', ylabel='nA', ylim=stim_lim)
    plt.plot(cell.tvec, cl_soma.i, c=cell_plot_colors[soma_clamp_params['idx']], lw=2)
    plt.plot(cell.tvec, cl_apic.i, '--', c=cell_plot_colors[apic_clamp_params['idx']], lw=2)




