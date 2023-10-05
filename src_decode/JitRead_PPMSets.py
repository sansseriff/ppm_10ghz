

from TimeTagger import createTimeTagger, FileWriter, FileReader
import os
import numpy as np
from numba import njit
from numba.typed import List
import yaml
import time
import json
import timeit
from bokeh.plotting import figure, output_file, show
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import math
from scipy.stats import norm
# import phd
# import matplotlib
from datetime import datetime
from nan_seperation import seperate_by_nans
from ClockTools_PPMSets import clockScan, histScan
import matplotlib
import matplotlib.pyplot as plt

plt.ion()

from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.plotting import figure, curdoc
import orjson

from bokeh.io import export_svg
from bokeh.layouts import gridplot
from bokeh.models import Span
# import phd.viz as viz
import snsphd.viz as viz
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis
from bokeh.models import HoverTool
from bokeh.models import Circle

from bokeh.models import ColumnDataSource

from scipy import ndimage

from sklearn.mixture import GaussianMixture
from dataclasses import dataclass
from enum import Enum

# Colors, palette = phd.viz.phd_style( data_width = 0.3, grid = True, axese_width=0.3, text = 2)
matplotlib.rcParams['figure.dpi'] = 150
doc = curdoc()
Colors, pallet = viz.bokeh_theme(return_color_list=True)
colors = Colors
output_file("layout.html")

Colors2, pallet2 = viz.phd_style(grid=True)




def checkLocking(Clocks, RecoveredClocks,mpl = False):

    Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"
    basis = np.linspace(Clocks[0], Clocks[-1], len(Clocks))
    diffs = Clocks - basis
    diffsRecovered = RecoveredClocks - basis
    s = 12800000/1e12
    # print("s is this: ", s)
    # print(len(diffs))
    x = np.arange(0, len(diffs))*s

    # print(np.max(x))

    if mpl:
        plt.figure()
        plt.plot(x,diffs)
        plt.plot(x,diffsRecovered)
        plt.title("check locking")
        return


    source = ColumnDataSource(data=dict(
        x=x[1:-1:8],
        y1=diffs[1:-1:8],
        y2=diffsRecovered[1:-1:8]))


    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)")
    ]
    s1 = figure(plot_width=800,
                plot_height=400,
                title="Clock Locking",
                output_backend="webgl",
                tools=Tools,
                active_scroll='xwheel_zoom', tooltips=TOOLTIPS)

    s1.line('x', 'y1', source = source, color = Colors['light_purple'])
    s1.line('x', 'y2', source = source, color = Colors['black'])
    s1.xaxis.axis_label = "time"
    s1.yaxis.axis_label = "Amplitude"


    s2 = figure(plot_width=800, plot_height=400,title="Clock Jitter", output_backend = "svg")
    hist,bins = np.histogram(diffs[9000:12000], bins=100, density=True)
    s2.line(bins[:-1],hist,color = colors['orange'])
    # ax[1].hist(diffs[9000:12000], bins=100, density=True)
    mean = np.mean(diffs)
    variance = np.var(diffs)
    sigma = np.sqrt(variance)
    x2 = np.linspace(min(diffs), max(diffs), 100)
    lbl = "fit, sigma = " + str(int(sigma))
    # ax[1].plot(x2, norm.pdf(x2, mean, sigma), label=lbl)
    # ax[1].legend()
    s2.line(x2, norm.pdf(x2, mean, sigma), color = Colors['black'],legend_label = lbl)
    # print("this is the length of the plotted array: ", len(diffs))

    return s1,s2




def checkLockingEffect(dataTags, dataTagsR, xlim=-1, ylim=-1):
    L = len(dataTags) // 2
    if xlim == -1:
        xlim = 0
    if ylim == -1:
        ylim = np.max(dataTags[L:])

    setBins = np.arange(xlim, ylim, 1)
    hist, bin_edges = np.histogram(dataTags[L:], setBins)
    histRough, bins_edges = np.histogram(dataTagsR[L:], setBins)
    # fig,ax = plt.subplots(1,1,figsize = (6,4))
    s3 = figure(plot_width=800, plot_height=400,title="Data with Phased Locked vs Regular Clock")
    s3.line(bin_edges[:-1], hist, legend_label="Phased Locked Clock", color = colors['light_purple'])
    s3.line(bin_edges[:-1], histRough, legend_label="Regular Clock", color = colors['purple'])

    # ax.plot(bin_edges[:-1], hist, label="Phased Locked Clock")
    # ax.plot(bin_edges[:-1], histRough, label="Regular Clock")
    # ax.legend()
    return s3


@njit
def countRateMonitor_b(timetaggs, reduc_factor):
    # print("legnth of passed array: ", len(timetaggs))
    # print("first in array: ", timetaggs[0])
    for i in range(len(timetaggs)):
        if timetaggs[i] > 0:
            break
    for j in range(1,len(timetaggs)):
        if timetaggs[-j] > 0:
            break
    delta_time = (timetaggs[-j] - timetaggs[i])//reduc_factor
    counts = []
    index = []  # for slicing the data array in another parent function
    timetaggs = timetaggs #- timetaggs[i] + 1
    # basic binning in chunks of time
    # print(timetaggs[10000:10010])
    times = []
    q = 0
    zero_counter = 0
    bla = 0
    rm_region = np.zeros(reduc_factor)
    for u in range(0, reduc_factor):
        current_time = delta_time*u + timetaggs[i]
        current_sum = 0
        while timetaggs[q] < current_time:
            q = q + 1
            if timetaggs[q] != 0:
                current_sum = current_sum + 1
                zero_counter = 0
            else:
                bla = bla + 1
                zero_counter = zero_counter + 1
        if zero_counter > 3:
            rm_region[u] = 1
        times.append(current_time)
        counts.append(current_sum/delta_time)
        index.append(q)
    # print("this is bla: ", bla)
    return counts, times, index, rm_region


@njit
def countRateMonitor(timetaggs,channels, channel_check,reduc_factor):
    delta_time = (timetaggs[-1] - timetaggs[0])//reduc_factor
    # delta_time is the bin size
    told = timetaggs[0]
    current = 0
    idx = 0
    counts = []
    index = []  # for slicing the data array in another parent function
    timeR = timetaggs[0] + delta_time
    timetaggs = timetaggs - timetaggs[0]
    # basic binning in chunks of time

    for i in range(len(timetaggs)):
        if channels[i] == channel_check:
            current = current + 1
            idx = idx + 1
        if timetaggs[i] > timeR:
            timeR = timeR + delta_time
            t_elapsed = timetaggs[i] - told
            told = timetaggs[i]
            counts.append(current/t_elapsed)
            index.append(idx)
            current = 0
    return counts, delta_time, index


def find_roots(x,y):  # from some stack overflow answer
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)


def analyze_count_rate_b(timetags, reduc):
    counts, times, index, rm_region = countRateMonitor_b(timetags, reduc)
    counts = np.array(counts)
    times = np.array(times)
    index = np.array(index)

    # cut out the long regions of zeros for easier plotting
    mask = np.invert(rm_region.astype(bool))
    times = times[mask]
    counts = counts[mask]
    index = index[mask]
    plt.figure()
    plt.plot(np.arange(len(rm_region)),rm_region)



    Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"

    X = np.array(times)  # (np.arange(len(counts))[1:]/len(counts))
    Y = np.array(counts) * 1e6
    IDX = np.array(index)
    source = ColumnDataSource(data = dict(x=X, y=Y, idx=IDX))

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("idx", "@idx")
        ]
    s3 = figure(plot_width=800, plot_height=400, title="count rate monitor",
                #output_backend="webgl",
                tools=Tools,
                active_scroll='xwheel_zoom', tooltips=TOOLTIPS)
    s3.xaxis.axis_label = "time"
    s3.yaxis.axis_label = "count rate (MCounts/s)"
    s3.line('x','y',source = source)

    mean = 3000000
    Y2 = find_roots(X,Y - mean)

    marker_source = ColumnDataSource(data = dict(x = Y2,y = np.zeros(len(Y2)) + mean))
    #print(Y2*len(X))
    glyph = Circle(x="x", y="y", size=3, line_color='red', fill_color="white", line_width=3)
    s3.add_glyph(marker_source, glyph)

    section_list = []

    return s3, section_list


def analyze_count_rate(timetags, channels, checkChan, reduc):
    counts, delta_time, index = countRateMonitor(timetags, channels, checkChan, reduc)
    Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"

    X = (np.arange(len(counts))[1:]/len(counts))
    Y = np.array(counts)[1:] * 1e6
    IDX = np.array(index[:-1])
    #print(IDX[:40])
    source = ColumnDataSource(data = dict(x=X, y=Y, idx=IDX))

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("idx", "@idx")
        ]
    s3 = figure(plot_width=800, plot_height=400, title="count rate monitor",
                output_backend="webgl", tools=Tools,
                active_scroll='xwheel_zoom', tooltips=TOOLTIPS)
    s3.xaxis.axis_label = "time"
    s3.yaxis.axis_label = "count rate (MCounts/s)"
    s3.line('x','y',source = source)

    Y2 = find_roots(X,Y - 0.5)

    marker_source = ColumnDataSource(data = dict(x = Y2,y = np.zeros(len(Y2)) + 2))
    glyph = Circle(x="x", y="y", size=3, line_color='red', fill_color="white", line_width=3)
    s3.add_glyph(marker_source, glyph)


    return s3, X, Y


def fft_convolve(array1, array2):
    array1 = np.flip(array1)
    tiled_array2 = np.tile(array2, 2)
    q = fftconvolve(array1, tiled_array2, mode='valid')
    print("length of fft convolve: ", len(q))
    return np.arange(len(q)), np.flip(q)


@njit
def jit_convolve(array1,array2):
    q = np.zeros(len(array1))
    for i in range(len(array1)):
        q[i] = np.dot(array1,array2)
        array1 = np.roll(array1,-1)
    return np.arange(len(q)), q

@njit
def jit_convolve_limited(array1,array2, start,end):
    array1 = np.roll(array1,start)
    r = end - start
    q = np.zeros(r)
    for i in range(r):
        q[i] = np.dot(array1,array2)
        array1 = np.roll(array1,1)
    return np.arange(len(q)) - start, q

def import_ground_truth(path, cycle_number):
    files = os.listdir(path)
    date_time = files[0].split('_')[1]
    file_sequence_data = str(cycle_number) + '_' + date_time + '_CH1.yml'
    with open(os.path.join(path,file_sequence_data), 'r') as f:
        sequence_data = yaml.load(f, Loader=yaml.FullLoader)


    file_set_data = '0_' + date_time + '_params.yml'
    with open(os.path.join(path,file_set_data), 'r') as f:
        set_data = yaml.load(f, Loader=yaml.FullLoader)

    return sequence_data, set_data


def make_ground_truth_hist(ground_truth_path,clock_period, sequence,resolution = 1000):
    sequence_data, set_data = import_ground_truth(ground_truth_path, sequence)
    dead_pulses = set_data["pulses_per_cycle"] - set_data["ppm"]["m_value"]
    dead_time_ps = dead_pulses * set_data["laser_time"] * 1e12
    times = np.array(
        sequence_data["times"]) * 1e12 + dead_time_ps
    bins = np.linspace(0, clock_period, resolution)
    ground_truth_hist, bins = np.histogram(times, bins=bins)
    return ground_truth_hist, bins


def make_ground_truth_regions(ground_truth_path,clock_period, sequence):
    sequence_data, set_data = import_ground_truth(ground_truth_path, sequence)
    dead_pulses = set_data["pulses_per_cycle"] - set_data["ppm"]["m_value"]
    dead_time_ps = dead_pulses * set_data["laser_time"] * 1e12
    times = np.array(
        sequence_data["times"]) * 1e12 + dead_time_ps
    regions = np.zeros((len(times),2))
    laser_time = set_data["laser_time"]*1e12
    regions[:,0] = times - (laser_time/2)
    regions[:,1] = times + (laser_time/2)

    return regions

def find_rough_offset(data,sequence,ground_truth_path,resolution = 1000):
    sequence_data, set_data = import_ground_truth(ground_truth_path, sequence)
    clock_period = int(set_data["total_samples"] / (0.001 * set_data["sample_rate"]))

    real_data_bins = np.linspace(0, clock_period, resolution)
    real_data_hist, bins = np.histogram(data, bins=real_data_bins)




    dead_pulses = set_data["pulses_per_cycle"] - set_data["ppm"]["m_value"]
    dead_time_ps = dead_pulses*set_data["laser_time"]*1e12
    #print("dead time in ps: ", dead_time_ps)

    # !!!! I am adding dead_time_ps because I want to move the clock tag to:
    # the FIRST laser pulse of the LAST deadtime in the awg sequences.
    times = np.array(
        sequence_data["times"]) * 1e12 + dead_time_ps  # adding on 10ns so that redefined clock is 10ns before first data
    # the extra 1000 cancels out a 1ns delay added in the sequenceGenerator script
    ground_truth_hist, bins = np.histogram(times, bins=bins)
    # x,y = jit_convolve(ground_truth_hist.astype(float),real_data_hist.astype(float))
    x,y = fft_convolve(ground_truth_hist.astype(float),real_data_hist.astype(float))
    plt.figure()
    plt.plot(x,y)
    title = f"convolution max found at {y.argmax()}"

    plt.title(title)
    return -(y.argmax()/resolution)*clock_period


def plot_hists(data, sequence, ground_truth_path, addition, resolution = 1000):
    sequence_data, set_data = import_ground_truth(ground_truth_path, sequence)
    clock_period = int(set_data["total_samples"] / (0.001 * set_data["sample_rate"]))

    real_data_bins = np.linspace(0, clock_period, resolution)
    real_data_hist, real_data_bins = np.histogram(data + addition, bins=real_data_bins)
    bins = real_data_bins
    dead_pulses = set_data["pulses_per_cycle"] - set_data["ppm"]["m_value"]
    dead_time_ps = dead_pulses*set_data["laser_time"]*1e12
    # print("dead time in ps: ", dead_time_ps)

    # !!!! I am adding dead_time_ps because I want to move the clock tag to:
    # the FIRST laser pulse of the LAST deadtime in the awg sequences (plus the extra bit at the end of the awg seq).
    times = np.array(
        sequence_data["times"]) * 1e12 + dead_time_ps   # adding on 10ns so that redefined clock is 10ns before
    # the extra 1000 cancels out a 1ns delay added in the sequenceGenerator script
    ground_truth_hist, bins = np.histogram(times + addition, bins=bins)
    # x,y = jit_convolve(ground_truth_hist.astype(float),real_data_hist.astype(float))
    plt.figure()
    plt.plot(bins[:-1],ground_truth_hist*1000, label = "ground truth")
    plt.plot(real_data_bins[:-1], real_data_hist, label = "real data")
    plt.title("this shows the offset that's used to reposition refChan is correct")
    plt.legend()


def offset_tags(dual_data,offset,clock_period):
    """
    Change the clock reference from which the data in dual_data is measured from. When the time offset sends
    some timetaggs outside of the range from 0 to [clock_period in ps], then those tags are 'rolled over'
    to the previous or successive group of tags with with a single clock reference.
    """
    dual_data = dual_data - offset
    greater_than_mask = (dual_data[:,0] > clock_period) & ~np.isnan(dual_data[:,0])
    less_than_mask = (dual_data[:,0] < 0) & ~np.isnan(dual_data[:,0])
    dual_data[greater_than_mask] = dual_data[greater_than_mask] - clock_period
    dual_data[less_than_mask] = dual_data[less_than_mask] + clock_period
    return dual_data



def offset_tags_single(data, offset, clock_period):
    data = data - offset
    greater_than_mask = (data > clock_period + 50/2)
    less_than_mask = (data < -50/2)
    data[greater_than_mask] = data[greater_than_mask] - clock_period
    data[less_than_mask] = data[less_than_mask] + clock_period

    return data


def generate_section_list(x,y,x_intercepts,idx):
    """
    Takes in a list of x_intercepts and the x and y axese of the count rate vs time plot on which they were found.
    A list is generated where reach row has an index that is near the beginning of a high count rate region, and has
    an index that is near the end of that region.
    """
    right_old = 0
    section_list = np.zeros((len(x_intercepts),2),dtype='int64') - 1
    q = 0
    threshold = np.mean(y)
    for intercept in x_intercepts:
        left = np.argmax(x > intercept) - 3
        right = np.argmax(x > intercept) + 3
        # print("value at left side: ", y[left])
        # print("value at right side: ", y[right])
        if y[right_old] > threshold and y[left] > threshold:
            # add that pair to the section list
            section_list[q,0] = right_old
            section_list[q, 1] = left
            q = q + 1
        right_old = right

    section_list = section_list[section_list[:,0] >= 0]
    print("identified ", len(section_list), " sections of high count rate.")

    # convert to dualData scaling from plot scaling
    for row in section_list:
        row[0] = idx[row[0]]
        row[1] = idx[row[1]]

    return section_list


def generate_PNR_analysis_regions(dual_data, cycle_number,clock_period, gt_path,region_radius = 3000):

    """
    finds regions of time about 2ns wide where detector counts are known to be. Adds the counts from multiple regions
    in one awg sequence to a single list of tags that can be used together for PNR effect cancellation.
    """
    bins = np.linspace(0, clock_period, 5000000)
    final_hist, final_bins = np.histogram(dual_data[:, 1], bins)
    gthist, bins = make_ground_truth_hist(gt_path, clock_period, cycle_number,
                                          resolution=5000000)
    plt.figure()
    plt.plot(bins[1:], gthist * 200)
    plt.plot(final_bins[1:], final_hist)
    plt.title("for generating PNR analysis regions")

    sequence_data, set_data  = import_ground_truth(gt_path, cycle_number)
    dead_pulses = set_data["pulses_per_cycle"] - set_data["ppm"]["m_value"]
    dead_time_ps = dead_pulses * set_data["laser_time"] * 1e12

    times = np.array(
        sequence_data["times"]) * 1e12 + dead_time_ps

    times_left = times - region_radius
    times_right = times + region_radius  # 400ps window over which to do the calibration
    for i, (left, right, current_time) in enumerate(zip(times_left,times_right,times)):
        mask = (dual_data[:,0] > left) & (dual_data[:,0] < right)
        counts = dual_data[mask] - current_time  # bring the counts to to near-zero
        l_bound = left - current_time
        r_bound = right - current_time
        #print(l_bound)
        if i == 0:
            sequence_counts = counts
        else:
            sequence_counts = np.concatenate((sequence_counts,counts))

    bins = np.arange(int(l_bound),int(r_bound))

    hist, bins = np.histogram(sequence_counts[:,0],bins = bins)
    hist2, bins = np.histogram(sequence_counts[:, 1], bins=bins)

    plt.figure()
    plt.plot(bins[1:],hist)
    plt.plot(bins[1:], hist2)

    return sequence_counts


def find_pnr_correction(counts):
    # I might have to do some data cleaning here?
    slices = np.arange(0, 2000, 4)
    corr1 = []
    corr2 = []
    for i in range(len(slices) - 1):
        left_bound = slices[i]
        right_bound = slices[i + 1]
        delta_t = counts[:, 1] - counts[:, 0]
        mask = (delta_t > left_bound) & (delta_t <= right_bound)
        corr1.append(np.mean(counts[:, 0][mask]))
        corr2.append(np.mean(counts[:, 1][mask]))

    corr1 = np.array(corr1)
    corr2 = np.array(corr2)

    # fill in NaNs
    mask = np.isnan(corr1)
    corr1[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), corr1[~mask])

    mask = np.isnan(corr2)
    corr2[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), corr2[~mask])

    return slices, corr1, corr2


def viz_counts_and_correction(counts,slices,corr1,corr2):

    bins = np.arange(-1000,1000)
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    ax[0].hist2d(counts[:, 1] - counts[:, 0], counts[:, 0], bins=(bins, bins),
                 norm=matplotlib.colors.LogNorm())
    ax[0].plot(slices[:-1], corr1, color='k')

    ax[1].hist2d(counts[:, 1] - counts[:, 0], counts[:, 1], bins=(bins, bins),
                 norm=matplotlib.colors.LogNorm())
    ax[1].plot(slices[:-1], corr2, color='k')


def apply_pnr_correction(dual_data_nan, slices, corr1, corr2,seperated_arrays = False):
    print("length of nans in the input array: ", np.sum(np.isnan(dual_data_nan)))

    nan_mask = np.isnan(dual_data_nan[:,0])
    not_nan_mask = ~nan_mask  # make it 1D
    dual_data = dual_data_nan[not_nan_mask]
    corrected1 = np.zeros(len(dual_data))
    corrected2 = np.zeros(len(dual_data))
    array_list1 = []
    array_list2 = []
    for i in range(len(slices) - 1):
        left_bound = slices[i]
        right_bound = slices[i + 1]
        delta_t = dual_data[:,1] - dual_data[:,0]
        mask = (delta_t > left_bound) & (delta_t <= right_bound)
        a = dual_data[:,0][mask] - corr1[i]
        b = dual_data[:, 1][mask] - corr2[i]
        corrected1[mask] = a
        corrected2[mask] = b
        if seperated_arrays:
            array_list1.append(a)
            array_list2.append(b)

    corrected1_nan = np.zeros(len(dual_data_nan))
    corrected2_nan = np.zeros(len(dual_data_nan))
    corrected1_nan[not_nan_mask] = corrected1  # left and right side should be same length
    corrected2_nan[not_nan_mask] = corrected2
    corrected1_nan[nan_mask] = np.nan
    corrected2_nan[nan_mask] = np.nan

    if seperated_arrays:
        return corrected1_nan, corrected2_nan, array_list1, array_list2
    else:
        return corrected1_nan, corrected2_nan
    

@dataclass
class GMData:
    num_components: int
    log_likelihood: float
    covariances: np.ndarray
    means: np.ndarray
    weights: np.ndarray


@dataclass
class GMTotalData:
    gm_list: list[GMData]
    counts: np.ndarray


def viz_correction_effect(sequence_counts, slices, corr1, corr2, gt_path, hist_set = False):
    if hist_set:
        corrected1, corrected2, array_list1, array_list2 = apply_pnr_correction(sequence_counts, slices, corr1, corr2,
                                                                            seperated_arrays=True)
    else:
        corrected1, corrected2 = apply_pnr_correction(sequence_counts, slices, corr1, corr2,
                                                                            seperated_arrays=False)
    sequence_data, set_data = import_ground_truth(gt_path, 0)
    laser_rate = set_data['system']['laser_rate']
    tbin = (1/laser_rate)*1000  # ps per time bin
    print("time bin width is: ", tbin)
    plt.figure()
    bins = np.arange(sequence_counts.min(), sequence_counts.max())
    hist, bins = np.histogram(corrected1, bins=bins)
    hist2, bins = np.histogram(corrected2, bins=bins)
    plt.plot(bins[1:], hist)
    plt.plot(bins[1:], hist2)
    plt.title("viz correction effect")

    m = (corrected1 <= tbin/2) & (corrected1 > -tbin/2)
    print("ratio: ", m.sum() / len(corrected1))

    if hist_set:
        plt.figure()
        i = 0
        for item in array_list1:
            if len(item) > 100:
                i = i + 1
                bins = np.arange(item.min(), item.max())
                hist, bins = np.histogram(item, bins=bins, density = True)
                plt.plot(bins[1:], hist)

        print("iterations: ", i)

@njit
def group_list_generator(tags):
    """
    this takes in a set of tags that all correspond to the same awg sequence. However, the set of tags is from many
    repetitions of that sequence. Objective is to seperate out groups of tags that were recorded from one repetition
    of the awg. For minimal loss, this can be done by seperating the tags at locations where the time differnce bewteen
    tags is negative (because they are in histogram-space). But when there's high loss, I need to identify when
    some reps of the awg result in zero detections. This fucntion adds emtpy numpy arrays to the output list to signify
    those events.
    """
    group_list = []
    old_tag = tags[0]//2 # just ensure old_tag is larger to start
    i = -1
    chunk = np.zeros(100)
    diffs = [] ###
    for tag in tags:
        diff = tag - old_tag
        if diff < 0:
            diffs.append(diff) ###
            group_list.append(chunk[:i])
            i = 0
            chunk = np.zeros(100)
            chunk[0] = tag
            i = i + 1
        else:
            if i < 100:
                chunk[i] = tag
                i = i + 1
            else:
                continue
        old_tag = tag

    return group_list, diffs ###


def using_clump(a):
    list_of_arrays = [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]
    for arr in list_of_arrays:
        arr.sort()
    return list_of_arrays

class Result(Enum):
    CORRECT = "A"
    INCORRECT = "B"
    MISSING = "D"
    INCORRECT_EXTRA = "C"
    DEADTIME_ERROR = "E"


@dataclass
class Event:
    result: Result
    measured: int = -1
    gaussian_measured: int = -1

    true: int = -1
    tag: int = -1
    tag_x: int = -1
    tag_y: int = -1


def decode_ppm(m_data_corrected, gt_path , sequence, clock_period, res_idx=[1]):
    sequence_data, set_data = import_ground_truth(gt_path, sequence)
    dead_pulses = set_data["pulses_per_cycle"] - set_data["ppm"]["m_value"]
    dead_time_ps = dead_pulses * set_data["laser_time"] * 1e12

    times = np.array(sequence_data["times"])
    laser_time = set_data['laser_time']*1e12  # in ps

    pulses_list = sequence_data["times_sequence"]
    pulses_per_cycle = set_data['pulses_per_cycle']
    # initial_time = dead_time_ps * 1e-12  # 250ns (for 20GHz)
    # initial_time = 0
    initial_time = 1000 # the 1000 is 1ns and matches an offset between times and times_sequence

    start_symbol_time = []
    start_data_time = []
    end_time = []
    true_pulse = []

    for pulse in pulses_list:
        time = initial_time + pulse*laser_time

        start_symbol_time.append(initial_time)
        start_data_time.append(initial_time + dead_time_ps)
        end_time.append(initial_time + (pulses_per_cycle - 1)*laser_time)
        true_pulse.append(pulse)

        initial_time = initial_time + pulses_per_cycle*laser_time

    #tag_group_list = np.split(m_data_corrected, np.where(np.diff(m_data_corrected) <= 0)[0] + 1)
    #tag_group_list, diffs = group_list_generator(m_data_corrected)
    #tag_group_list, diffs = group_list_generator(m_data_corrected)
    #print("m data corrected: ", m_data_corrected[:100])
    #print(m_data_corrected[:20])
    # tag_group_list = using_clump(m_data_corrected)
    tag_group_list = seperate_by_nans(m_data_corrected, 200)

    symbol_start = start_symbol_time[0]
    symbol_end = end_time[0]
    data_start = start_data_time[0]
    true_p = true_pulse[0]
    stage = []
    q = 0
    _results = []

    #####
    Results = [0]*len(res_idx)
    for x, idx in enumerate(res_idx):
        current_list = tag_group_list[idx]
        _results = []
        ######
        # print(current_list)
        # has a tendency to process a tag too early...
        for i, tag in enumerate(current_list): # generalize later
            if tag > end_time[-1]:
                # tag is in extra region at the end of the sequence that doesn't corresponds to any data
                break

            if tag > symbol_end + (laser_time/2):  # last slot (like 2047 or 1023) is at time symbol_end
                # incude the len(stage) part so that we don't try to decode empty stage if 1st tag of sequence not found
                # current stage is full. Process it.
                if len(stage) > 0:
                    _results.append(decode_symbol(stage, symbol_start, symbol_end, data_start, true_p, laser_time))
                    stage = []
                else:
                    _results.append([-1, 'D'])

                # if sequence == 37:
                #     print("symbol end 37: ", symbol_end + (laser_time/2))
                #     print("tag: ", tag)
                # if sequence == 46:
                #     print("symbol end 46: ", symbol_end + (laser_time / 2))
                #     print("tag: ", tag)
                    # stage is still []
                # prepare the next stage.
                # does current tag fit in next stage or a later stage?
                # look through remaining symbol regions
                while 1:
                    q = q + 1
                    if q < len(pulses_list):
                        symbol_start = start_symbol_time[q]
                        symbol_end = end_time[q]
                        data_start = start_data_time[q]
                        true_p = true_pulse[q]
                        if (tag > symbol_start - (laser_time/2)) and (tag < symbol_end + (laser_time/2)):
                            # found new region to fill
                            stage.append(tag)

                            # if last tag, process it now before checking for more
                            if i == len(current_list) - 1:
                                _results.append(
                                    decode_symbol(stage, symbol_start, symbol_end, data_start, true_p, laser_time))
                            break
                        else:
                            # symbol passed with no data. append the vacuume identifier to results.
                            _results.append([-1,'D'])
                    else:
                        # no more data found in this cycle.
                        _results.append([-1, 'D'])
                        break

            else:
                stage.append(tag)

                # if last tag, process it now before checking for more
                if i == len(current_list) - 1:
                    _results.append(
                        decode_symbol(stage, symbol_start, symbol_end, data_start, true_p, laser_time))

        still_missing = len(pulses_list) - len(_results)
        _results.extend([[-1,'D']]*still_missing)
        ###################

        # e = 0
        # for item in results:
        #     if item[1] == 'E':
        #         e = e + 1

        # if len(current_list) < e and e > 6:
        #     print(results)
        #     print(current_list)
        #     viz_current_decoding(current_list, gt_path, 3200000, sequence, start = start_data_time, end = end_time)
        #     print("##################################")

        # if sequence == 37:
        #     viz_current_decoding(current_list, gt_path, clock_period, sequence, start=start_data_time, end=end_time)
        #     print("########## sequence: ", sequence)
        #     print(current_list)
        #     print(_results)
        #
        # if sequence == 46:
        #     viz_current_decoding(current_list, gt_path, clock_period, sequence, start=start_data_time, end=end_time)
        #     print("########## sequence: ", sequence)
        #     print(current_list)
        #     print(_results)
        #
        # if sequence == 118:
        #     viz_current_decoding(current_list, gt_path, clock_period, sequence, start=start_data_time, end=end_time)
        #     print("########## sequence: ", sequence)
        #     print(current_list)
        #     print(_results)

        # if len(results) != 9:
        #     print(current_list)
        #     viz_current_decoding(current_list, gt_path, 3200000, sequence)
        #     print("symbol end: ", symbol_end)
        Results[x] = _results


    # print("ROBUST RESULT: ", results)
            # move onto next PPM region
    # print("tag group list 0: ", tag_group_list[0])
    # print("tag group list 1: ", tag_group_list[1])
    # print("tag group list 2: ", tag_group_list[2])

    # print(len(tag_group_list))
    # print(tag_group_list[1]) # don't use the 1st section, it may be partially filled.
    # print("LENGTH OF TAG GROUP LIST 1: ", len(tag_group_list[1]))

    # print("TAG GROUP LIST: ", tag_group_list)

    # print("TAG GROUP LIST 2: ", tag_group_list[2])
    # for i, tag in enumerate(tag_group_list[2]):
    #     if len(tag_group_list[2]) == len(start_data_time):
    #         tag = tag - start_data_time[i]
    #         tag_group_list[2][i] = round(tag/50)
    #
    # if len(tag_group_list[2]) == len(start_data_time):
    #     print("DATA TEST: ", tag_group_list[2])
    #     print("GT: ", true_pulse)

    tt = 0
    for item in _results:
        if item[1] == 'A':
            tt = tt + 1

    return Results, tt #  , diffs


def decode_symbol(stage, symbol_start, symbol_end, data_start, true_p, laser_time):
    err = []
    for tag in stage:

        if (tag > data_start - (laser_time/2)) and (tag < symbol_end + (laser_time/2)):
            # options A or B
            solved = round((tag - data_start)/laser_time)
            # if its correct and there haven't been
            if solved == true_p: # and len(err) == 0:
                if len(stage) > 1:
                    for i,tag in enumerate(stage):
                        stage[i] = round((tag - data_start)/laser_time)
                    return [solved,'A']
                else:
                    return [solved, 'A']
            else:
                err.append(solved)

        else:
            # must be a tag in the deadtime
            # A, C, or E
            err.append(-1)

    for error in err:
        if (error != -1) and len(err) > 1:
            return [error,'C']
        if (error != -1) and len(err) == 1:
            return [error,'B']

    # if here, all errors are in the deadtime
    return [-1,'E']


def find_first_dirtyClock(array, num):
    q = 0
    for item in array:
        if item != 0:
            q = q + 1
        if q == num:
            return item


def find_trailing_dirtyClock(array, num):
    q = 0
    for i in range(len(array)):
        if array[-i] != 0:
            q = q + 1
        if q == num:
            return array[-i]


def find_diff_regions(tags,extra = 3):
    # assumed there are a lot of zeros mixed in the list of tags.
    # these are placeholders so the length of the tags array matches other arrays elsewhere
    mask = tags > 0
    idx_ref = np.arange(len(tags))[mask]
    diffs = np.diff(tags[mask])

    plt.figure()
    plt.title("clock locking analysis")
    x = np.arange(len(diffs))
    plt.plot(x, diffs)
    ind = np.argpartition(diffs, -100)[-100:]  # the indexes of the 100 largest numbers in diffs (from stack overflow)
    ind1000 = np.sort(np.argpartition(diffs, -1000)[-1000:])[:30]
    #print("30 smallest of 1000 largest: ", diffs[ind1000])
    general_max = np.mean(diffs[ind])
    #print(np.mean(diffs[ind]))
    ints = find_roots(x,diffs - general_max/2)
    intersection_viz_y = np.zeros(len(ints)) + general_max/2
    plt.plot(ints, intersection_viz_y,'o',markersize=2)

    sections = []
    sections.append([0,int(math.floor(ints[0]))]) # the first big calibrate section
    for i in range(1,len(ints)-1):
        if ints[i] - ints[i-1] < 4 and ints[i+1] - ints[i] > 30:
            right = ints[i]
        if ints[i] - ints[i-1] > 30 and ints[i+1] - ints[i] < 4:
            left = ints[i]
            if ints[i - 1] == right:
                sections.append([int(math.ceil(right)) + extra,int(math.floor(left)) - extra])

    sections.append([int(math.ceil(left+1)),int(math.floor(ints[-1]))])

    plt.plot(sections[35],[general_max/2,general_max/2], 'o', color = 'red')
    plt.plot(sections[100], [general_max / 2, general_max / 2], 'o', color='orange')
    plt.plot(sections[307], [general_max / 2, general_max / 2], 'o', color='orange')

    # convert sections from the compressed array index to the expanded array index (the format with many zeros)
    for i, section in enumerate(sections):
        section[0] = idx_ref[section[0]]
        section[1] = idx_ref[section[1]]
    return np.array(sections)


def viz_current_decoding(data,gt_path, clock_period, cycle_number, start = None, end = None):

    #print("viz of cycle number: ", cycle_number)
    bins = np.linspace(0, clock_period, 500000)
    final_hist, final_bins = np.histogram(data, bins)
    # gthist, bins = make_ground_truth_hist(gt_path, clock_period, cycle_number,
    #                                       resolution=500000)
    regions = make_ground_truth_regions(gt_path, clock_period, cycle_number)
    fig,ax = plt.subplots(1,1)
    #plt.plot(bins[1:], gthist * 200)
    ax.plot(final_bins[1:], final_hist, linewidth=3)
    title = "viz_current_decoding of cycle number: " + str(cycle_number)
    for i in range(len(regions)):
        ax.axvspan(regions[i,0],regions[i,1], facecolor = 'g', alpha = 0.3)
        ax.axvline(x = regions[i, 0], color='g', alpha=0.8)
        ax.axvline(x=regions[i, 1], color='g', alpha=0.8)
        if end is not None and start is not None:
            ax.axvspan(start[i], end[i], facecolor = 'r', alpha = 0.3)

    plt.title(title)


def adjust_ref_channel(_channels, _timetags, _offset, refChan):
        mask = (_channels == refChan)
        # timetags_old = np.copy(_timetags)
        _timetags[mask] = _timetags[mask] - _offset
        sort = _timetags.argsort()
        shifted = np.count_nonzero((np.diff(sort)<0))
        print("shifted ", shifted, " refChan tags before full decode")
        return _channels[sort], _timetags[sort]

def runAnalysisJit(path_, file_, gt_path):
    full_path = os.path.join(path_, file_)
    file_reader = FileReader(full_path)

    while file_reader.hasData():
        n_events = 2000000000
        data = file_reader.getData(n_events)
        channels = data.getChannels()
        timetags = data.getTimestamps()
        R = 2400000
        hist_tags = histScan(channels[R:R + 60000], timetags[R:R + 60000], -5, -14, 9)
        bins = np.arange(np.max(hist_tags))
        hist, bins = np.histogram(hist_tags,bins)
        plt.figure()
        plt.plot(bins[:-1],hist)
        plt.title("for alignment")

        t1 = time.time()
        print("start finding ref offset")
        ref_offset = find_rough_offset(hist_tags, 0, gt_path, resolution=800000)
        print("stop finding ref offset")
        print("find rough offset time is: ", time.time() - t1)
        print("offset is: ", ref_offset)
        _, set_data = import_ground_truth(gt_path, 0)
        CLOCK_PERIOD = int(set_data["total_samples"] / (0.001 * set_data["sample_rate"]))
        print("CLOCK PERIOD: ", CLOCK_PERIOD)

        hist_tags = offset_tags_single(hist_tags,ref_offset,CLOCK_PERIOD)
        plot_hists(hist_tags, 0, gt_path, 0, resolution=50000)
        #plot_hists(hist_tags - offset, 0, gt_path, 0, resolution=50000)
        # adjust_ref_channel(channels, timetags, offset, 9)
        channels, timetags = adjust_ref_channel(channels, timetags, ref_offset, 9)
        #print(channels[R:R + 100])

        Clocks, RecoveredClocks, dataTags, dataTagsR, dualData, countM, dirtyClock, histClock = clockScan(
            channels[R:-1], timetags[R:-1], 18, -5, -14, 9, clock_mult=4, deriv = 1000, prop=64e-13)

        

        # s1, s2 = checkLocking(Clocks[0:-1:20], RecoveredClocks[0:-1:20])
        checkLocking(Clocks[0:-1], RecoveredClocks[0:-1],mpl = True)

        

        # return [s1,s2]

        section_list = find_diff_regions(dirtyClock,extra = 5)
        print("LENGTH OF SECTION LIST: ", len(section_list))

        calibrate_number = 0
        sequence_offset = 0
        SEQ = calibrate_number + sequence_offset
        calibrate_section = section_list[calibrate_number]  # should always be the first section
        not_nan_mask = ~np.isnan(dualData[:,0])
        not_nan_dualData = dualData[not_nan_mask]

        # don't want start to be negative
        if calibrate_section[1] - 500000 > 0:
            start = calibrate_section[1] - 500000
        else:
            start = 0

        m_data = dualData[start:calibrate_section[1]-1000]  # only use a small portion for cal.

        # grab the last little bit of the tags from the 1st calibration awg sequence
        offset_analysis_region = histClock[calibrate_section[1]-10000:calibrate_section[1]-1000]

        print("variance of offset_analysis_region: ", np.var(offset_analysis_region[offset_analysis_region != 0]))

        dirty_clock_offset_1 = np.mean(offset_analysis_region[offset_analysis_region != 0])

        bins = np.arange(np.max(m_data[~np.isnan(m_data)]))
        hist,bins = np.histogram(m_data[~np.isnan(m_data)],bins = bins)
        plt.figure()
        plt.plot(bins[:-1],hist)
        plt.title("m data")


        # will input datatags that correspond only to individual sequences
        offset = find_rough_offset(m_data, SEQ, gt_path, resolution=800000)
        m_data = offset_tags(m_data, offset, CLOCK_PERIOD)
        sequence_counts = generate_PNR_analysis_regions(m_data, SEQ, CLOCK_PERIOD,
                                                        gt_path,region_radius = 5000)
        # start = time.time()
        slices, corr1, corr2 = find_pnr_correction(sequence_counts)
        # end = time.time()

        viz_counts_and_correction(sequence_counts, slices, corr1, corr2)
        viz_correction_effect(sequence_counts, slices, corr1, corr2, gt_path)
        m_data_corrected, _ = apply_pnr_correction(m_data, slices, corr1, corr2)


        t1 = time.time()

        #q = dualData[section_list[3,0]:section_list[3,1]]

        #print(q[:100])


        #print(dualData[calibrate_section[1]:])

        imgData = offset_tags(dualData[calibrate_section[1]:], offset, CLOCK_PERIOD)
        #print(imgData[100000:100100])



        section_list = section_list - calibrate_section[1]

        # imgData is now shorter than dualData because it does not include the calibrate region.
        # new versions of these arrays that don't include the calibrate section
        histClock = histClock[calibrate_section[1]:]
        dirtyClock = dirtyClock[calibrate_section[1]:]

        print("number of nans in one axis of imgData: ", np.sum(np.isnan(imgData[:,0])))
        imgData_corrected, _ = apply_pnr_correction(imgData, slices, corr1, corr2)
        print("number of nan in imgData_corrected: ", np.sum(np.isnan(imgData_corrected)))

        # loop over the whole image
        t1 = time.time()
        TTS = []
        results = [[]]  # inside should match res_idx
        Diffs = []
        offs = []

        input('to continue press enter')


        for i, slice in enumerate(section_list[:-1]):
            if i == 0:  # fist section used only for calibration
                continue

            left = slice[0]
            right = slice[1] - 200 # weird that I need the 1000. see log for 8/25/21. some delay issue??

            current_data_corrected = imgData_corrected[left:right]
            #print("current data corrected: ", current_data_corrected[:100])
            dirtyClock_offset = histClock[left:right]
            dirtyClock_b = dirtyClock[left:right]


            dirty_clock_of1 = np.mean(dirtyClock_offset[(dirtyClock_offset != 0) & (~np.isnan(dirtyClock_offset))])


            _, set_data = import_ground_truth(gt_path, 0)
            mult = 16000/set_data['system']['laser_rate']
            # print(mult)

            offset_adjustment_1 = round((dirty_clock_of1 - dirty_clock_offset_1)/mult)*mult
            # print("offs: ", dirty_clock_of1 - dirty_clock_offset_1)

            offs.append(dirty_clock_of1 - dirty_clock_offset_1)

            #offset_adjustment_2 = round((dirty_clock_of2 - dirty_clock_offset_1)/200)*200

            current_data_corrected_1 = offset_tags_single(current_data_corrected,offset_adjustment_1,CLOCK_PERIOD)
            #current_data_corrected_2 = offset_tags_single(current_data_corrected, offset_adjustment_2, CLOCK_PERIOD)

            results1, TT1 = decode_ppm(current_data_corrected_1, gt_path, i, CLOCK_PERIOD, res_idx = [1])
            #results2, TT2, _ = decode_ppm(current_data_corrected_2, gt_path, i, res_idx = [3])
            print(results1)


            for res, master_res in zip(results1, results):
                master_res.extend(res)
            TTS.append(TT1)

        print("loop time: ", time.time() - t1)
        TTS = np.array(TTS)
        print("ACCURACY: ", np.mean(TTS)/8)


        # graphs = [s1, s2]
        return results #, graphs

@dataclass
class Out:
    results: list[list[Event]]
    gm_data: GMTotalData
    hist_data: PNRHistCorrectionData
    correction_data: CorrectionData

if __name__ == "__main__":
    full_decode = False
    path = "..//data//measured//"
    gt_path = "..//data//ground_truth//"
    if full_decode:
        file_list = ["430s_.002_.050_Aug8_picScan_18.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_20.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_22.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_24.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_26.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_28.0.1.ttbin"
                     "430s_.002_.050_Aug8_picScan_30.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_32.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_34.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_36.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_38.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_40.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_42.0.1.ttbin",
                     "430s_.002_.050_Aug8_picScan_44.0.1.ttbin"]

        DEBUG = False
        for file in file_list:
            results = runAnalysisJit(path, file, gt_path)
            results_bytes = orjson.dumps(results)

            dB_stub = file.split('_')[-1][:-10]
            with open("..//inter//decode_10GHz" + dB_stub + ".json", 'wb') as file:
                file.write(results_bytes)
            time.sleep(4)

    else:
        DEBUG = True

        # file = "430s_.002_.050_Aug8_picScan_18.0.1.ttbin"   # works
        # file = "430s_.002_.050_Aug8_picScan_20.0.1.ttbin"   # works
        # file = "430s_.002_.050_Aug8_picScan_22.0.1.ttbin"   # works
        # file = "430s_.002_.050_Aug8_picScan_24.0.1.ttbin"   # works median: .990 mean: 0.989
        # file = "430s_.002_.050_Aug8_picScan_26.0.1.ttbin"
        # file = "430s_.002_.050_Aug8_picScan_28.0.1.ttbin"   # works median: 0.946   mean: 0.949
        # file = "430s_.002_.050_Aug8_picScan_30.0.1.ttbin"
        file = "430s_.002_.050_Aug8_picScan_32.0.1.ttbin"
        # file = "430s_.002_.050_Aug8_picScan_34.0.1.ttbin"
        # file = "430s_.002_.050_Aug8_picScan_36.0.1.ttbin"
        # file = "430s_.002_.050_Aug8_picScan_38.0.1.ttbin"
        # file = "430s_.002_.050_Aug8_picScan_40.0.1.ttbin"
        # file = "430s_.002_.050_Aug8_picScan_42.0.1.ttbin"
        # file = "430s_.002_.050_Aug8_picScan_44.0.1.ttbin"

        results = runAnalysisJit(path, file, gt_path)
        #show(column(graphs[0],graphs[1],graphs[2]))
        # show(column(graphs[0], graphs[1]))
        #print(results)
        input("Press Enter to exit...")

