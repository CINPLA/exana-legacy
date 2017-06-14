import os
import os.path as op
import numpy as np
from datetime import datetime
import quantities as pq


def auto_denoise(anas, thresh=None, copy_signal=True):
    """Clean neural data from EMG, chewing, and moving artifact noise
    Rectified signals are smoothed and thresholded to find and remove 
    noisy portion of the signals.

    Parameters
    ----------
    anas : np.array 
           2d array of analog signals
    thresh : float
             (optional) threshold in number of SD on high-pass data
    copy_signal : bool
                  copy signals or not

    Returns
    -------
    anas_copy :  cleaned analog signals
    """
    from scipy import signal
    from copy import copy
    if thresh:
        thresh = thresh
    else:
        thresh = 2.5
    
    if copy_anas:
        anas_copy = copy(anas)
    else:
        anas_copy = anas

    env = signal.hilbert(anas_copy)
    env = np.abs(env)

    # Smooth
    smooth = 0.01
    b_smooth, a_smooth = signal.butter(4, smooth)
    env = signal.filtfilt(b_smooth, a_smooth, env)

    if len(anas.shape) == 1:
        sd = np.std(anas)
    else:
        sd = np.std(anas, axis=1)
        noisy_idx = []
        # Find time points in which all channels are noisy
        for i in np.arange(len(anas[0])):
            if np.all([env[ch, i] >= thresh*sd[ch] for ch in np.arange(env.shape[0])]):
                noisy_idx.append(i)
        anas_copy[:, noisy_idx] = 0


    return anas_copy

def manual_denoise(anas, thresh=None, copy_signal=True):
    """Clean neural data from EMG, chewing, and moving artifact noise
    User can select the points to cut out for the denoised signal.

    Parameters
    ----------
    anas : np.array 
           2d array of analog signals
    thresh : float
             (optional) threshold in number of SD on high-pass data
    copy_anas : bool
                copy signals or not

    Returns
    -------
    anas_copy :  cleaned analog signals
    """
    from copy import copy
    import matplotlib.pyplot as plt

    if copy_signal:
        anas_copy = copy(anas)
    else:
        anas_copy = anas
    plt.figure()
    plt.plot(np.transpose(anas))
    seg = plt.ginput(0, timeout=15)
    seg_x = [seg[i][0] for i in range(len(seg))]

    if divmod(len(seg_x), 2)[1] != 0:
        seg_x = seg_x[:-2]

    seg_x = [int(s) for s in seg_x]

    # Clip selected
    interv = np.reshape(seg_x, (int(len(seg)/2), 2))
    for i, s in enumerate(interv):
        anas_copy[:, s[0]:s[1]] = 0

    return anas_copy

def ica_denoise(anas, channels=None, n_comp=10, correlation_thresh=0.1):
    """Removes noise by ICA. Indepentend components highly correlated to the 
    grand average of the signals are removed.
    Signals are then back projected to the channel space.

    Parameters
    ----------
    anas : np.array 
           2d array of analog signals
    channels : list
               list of good channels to perform ICA with
    n_comp : int
             number of ICA components 
    correlation_tresh : float
                        correlation threshold between average signal
                        and source above which source contibution gets
                        removed

    Returns
    -------
    anas_sig : cleaned analog signals
    anas_noise : noise projection on signals
    sources : ICA sources
    mixing : ICA mixing matrix
    corr : correlation between ICA sources and average signal
    """
    from sklearn.decomposition import FastICA
    from scipy import stats
    if channels is None:
        channels = np.arange(anas.shape[0])

    print('Applying ICA...')
    ica = FastICA(n_components=n_comp)
    sources = np.transpose(ica.fit_transform(np.transpose(anas[channels])))
    mixing = ica.mixing_

    ref = np.mean(anas[channels], axis=0)

    # find correlations
    corr = np.zeros(n_comp)
    print('Computing correlations...')
    for i in np.arange(n_comp):
        corr[i] = stats.stats.pearsonr(ref, sources[i])[0]

    anas_sig = np.zeros(anas.shape)
    anas_noise = np.zeros(anas.shape)

    idx_sig = np.where(corr <= correlation_thresh)
    idx_noise = np.where(corr > correlation_thresh)
    anas_sig[channels] = np.squeeze(np.dot(mixing[:, idx_sig], sources[idx_sig]))
    anas_noise[channels] = np.squeeze(np.dot(mixing[:, idx_noise], sources[idx_noise]))

    return anas_sig, anas_noise, sources, mixing, corr


def apply_CAR(anas, channels=None, car_type='mean', split_probe=None, copy_signal=True):
    """Removes noise by Common Average or Median Reference.

    Parameters
    ----------
    anas : np.array 
           2d array of analog signals
    channels : list
               list of good channels to perform CAR/CMR with
    car_type : string
               'mean' or 'median'
    split_probe : int
                  splits anas into different probes to apply
                  car/cmr to each probe separately

    Returns
    -------
    anas_car : cleaned analog signals
    avg_ref : reference removed from signals
    """
    from copy import copy
    if channels is None:
        channels = np.arange(anas.shape[0])
    if copy_signal:
        anas_car = copy(anas)
    else:
        anas_car = anas
    anas_car = np.array(anas_car, dtype=np.float32)

    if car_type is 'mean':
        print('Applying CAR')
        if split_probe is not None:
            avg_ref = np.mean(anas_car[:split_probe], axis=0)
            anas_car[:split_probe] -= avg_ref
            avg_ref = np.mean(anas_car[split_probe:], axis=0)
            anas_car[split_probe:] -= avg_ref
        else:
            avg_ref = np.mean(anas_car[channels], axis=0)
            anas_car[channels] -= avg_ref
    elif car_type is 'median':
        print('Applying CMR')
        if split_probe is not None:
            avg_ref_1 = np.median(anas_car[:split_probe], axis=0)
            anas_car[:split_probe] -= avg_ref_1
            avg_ref_2 = np.median(anas_car[split_probe:], axis=0)
            anas_car[split_probe:] -= avg_ref_2
            avg_ref = np.array([avg_ref_1, avg_ref_2])
        else:
            avg_ref = np.median(anas_car[channels], axis=0)
            anas_car[channels] -= avg_ref
    else:
        raise AttributeError("'type must be 'mean' or 'median'")

    return anas_car, avg_ref

def extract_rising_edges(adc_signal, times, thresh=1.65):
    """Extract rising times from analog signal used as TTL.

    Parameters
    ----------
    adc_signal : np.array 
                 1d array of analog TTL signal
    times : np.array
            timestamps array
    thresh: float
            threshold to detect 'high' value

    Returns
    -------
    rising_times : np.array with rising times
    """
    print('im here')
    idx_high = np.where(adc_signal>1.65)[0]

    rising = []

    if len(idx_high) != 0:
        for i, idx in enumerate(idx_high[:-1]):
            if i==0:
                # first idx is rising
                rising.append(idx)
            elif idx - 1 != idx_high[i-1]:
                rising.append(idx)
    rising_times = times[rising]

    return rising_times

def filter_analog_signals(anas, freq, fs, filter_type='bandpass', order=3):
    """Filters analog signals with zero-phase Butterworth filter.
    The function raises an Exception if the required filter is not stable.

    Parameters
    ----------
    anas : np.array 
           2d array of analog signals
    freq : list or float
           cutoff frequency-ies in Hz
    fs : float
         sampling frequency
    filter_type : string
                  'lowpass', 'highpass', 'bandpass', 'bandstop'
    order : int
            filter order

    Returns
    -------
    anas_filt : filtered signals
    """
    from scipy.signal import butter, filtfilt
    fn = fs / 2.
    band = np.array(freq) / fn

    b, a = butter(order, band, btype=filter_type)

    if np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1):
        print('Filtering signals with ', filter_type, ' filter at ' , freq ,'...')
        anas_filt = []
        if len(anas.shape) == 2:
            anas_filt = filtfilt(b, a, anas, axis=1)
        elif len(anas.shape) == 1:
            anas_filt = filtfilt(b, a, anas)
        return anas_filt
    else:
        raise ValueError('Filter is not stable')

def ground_bad_channels(anas, bad_channels, copy_signal=True):
    """Grounds selected noisy channels.

    Parameters
    ----------
    anas : np.array 
           2d array of analog signals
    bad_channels : list
                   list of channels to be grounded
    copy_signal : bool
                  copy signals or not

    Returns
    -------
    anas_zeros : analog signals with grounded channels
    """
    print('Grounding channels: ', bad_channels, '...')

    from copy import copy
    nsamples = anas.shape[1]
    if copy_signal:
        anas_zeros = copy(anas)
    else:
        anas_zero = anas
    if type(bad_channels) is not list:
        bad_channels = [bad_channels]

    for i, ana in enumerate(anas_zeros):
        if i in bad_channels:
            anas_zeros[i] = np.zeros(nsamples)

    return anas_zeros


def duplicate_bad_channels(anas, bad_channels, probefile, copy_signal=True):
    """Duplicate selected noisy channels with channels in 
    the same channel group.

    Parameters
    ----------
    anas : np.array 
           2d array of analog signals
    bad_channels : list
                   list of channels to be grounded
    probefile : string
                absolute path to klusta-like probe file
    copy_signal : bool
                  copy signals or not

    Returns
    -------
    anas_dup : analog signals with duplicated channels
    """
    print('Duplicating good channels on channels: ', bad_channels, '...')

    def _select_rnd_chan_in_group(channel_map, ch_idx):
        for group_idx, group in channel_map.items():
            if ch_idx in group['channels']:
                gr = np.array(group['channels'])
                rnd_idx = np.random.choice(gr[gr != ch_idx])
                return rnd_idx

    def _read_python(path):
        from six import exec_
        path = op.realpath(op.expanduser(path))
        assert op.exists(path)
        with open(path, 'r') as f:
            contents = f.read()
        metadata = {}
        exec_(contents, {}, metadata)
        metadata = {k.lower(): v for (k, v) in metadata.items()}
        return metadata

    probefile_ch_mapping = _read_python(probefile)['channel_groups']

    from copy import copy
    nsamples = anas.shape[1]
    if copy_signal:
        anas_dup = copy(anas)
    else:
        anas_dup = anas
    if type(bad_channels) is not list:
        bad_channels = [bad_channels]

    for i, ana in enumerate(anas_dup):
        if i in bad_channels:
            rnd = _select_rnd_chan_in_group(probefile_ch_mapping, i)
            anas_dup[i] = anas[rnd]

    return anas_dup

def save_binary_format(filename, signal, spikesorter='klusta'):
    """Saves analog signals into klusta (time x chan) or spyking
    circus (chan x time) binary format (.dat)

    Parameters
    ----------
    filename : string
               absolute path (_klusta.dat or _spycircus.dat are appended)
    signal : np.array 
             2d array of analog signals
    spikesorter : string
                  'klusta' or 'spykingcircus'

    Returns
    -------
    """
    if spikesorter is 'klusta':
        fdat = filename + '_klusta.dat'
        print('Saving ', fdat)
        with open(fdat, 'wb') as f:
            np.transpose(np.array(signal, dtype='float32')).tofile(f)
    elif spikesorter is 'spykingcircus':
        fdat = filename + '_spycircus.dat'
        print('Saving ', fdat)
        with open(fdat, 'wb') as f:
            np.array(signal, dtype='float32').tofile(f)


def create_klusta_prm(pathname, prb_path, nchan=32, fs=30000,
                      klusta_filter=True, filter_low=300, filter_high=6000):
    """Creates klusta .prm files, with spikesorting parameters

    Parameters
    ----------
    pathname : string
               absolute path (_klusta.dat or _spycircus.dat are appended)
    prbpath : np.array 
              2d array of analog signals
    nchan : int
            number of channels
    fs: float
        sampling frequency
    klusta_filter : bool
        filter with klusta or not
    filter_low: float
                low cutoff frequency (if klusta_filter is True)
    filter_high : float
                  high cutoff frequency (if klusta_filter is True)
    Returns
    -------
    full_filename : absolute path of .prm file
    """
    assert pathname is not None
    abspath = op.abspath(pathname)
    assert prb_path is not None
    prb_path = op.abspath(prb_path)
    full_filename = abspath + '.prm'
    print('Saving ', full_filename)
    with open(full_filename, 'w') as f:
        f.write('\n')
        f.write('experiment_name = ' + "r'" + abspath + '_klusta' + "'" + '\n')
        f.write('prb_file = ' + "r'" + prb_path + "'")
        f.write('\n')
        f.write('\n')
        f.write("traces = dict(\n\traw_data_files=[experiment_name + '.dat'],\n\tvoltage_gain=1.,"
                "\n\tsample_rate="+str(fs)+",\n\tn_channels="+str(nchan)+",\n\tdtype='float32',\n)")
        f.write('\n')
        f.write('\n')
        f.write("spikedetekt = dict(")
        if klusta_filter:
            f.write("\n\tfilter_low="+str(filter_low)+",\n\tfilter_high="+str(filter_high)+","
                    "\n\tfilter_butter_order=3,\n\tfilter_lfp_low=0,\n\tfilter_lfp_high=300,\n")
        f.write("\n\tchunk_size_seconds=1,\n\tchunk_overlap_seconds=.015,\n"
                "\n\tn_excerpts=50,\n\texcerpt_size_seconds=1,"
                "\n\tthreshold_strong_std_factor=4.5,\n\tthreshold_weak_std_factor=2,\n\tdetect_spikes='negative',"
                "\n\n\tconnected_component_join_size=1,\n"
                "\n\textract_s_before=16,\n\textract_s_after=48,\n"
                "\n\tn_features_per_channel=3,\n\tpca_n_waveforms_max=10000,\n)")
        f.write('\n')
        f.write('\n')
        f.write("klustakwik2 = dict(\n\tnum_starting_clusters=50,\n)")
                # "\n\tnum_cpus=4,)")
    return full_filename


def remove_stimulation_artifacts(anas, times, trigger, pre=3 * pq.ms, post=5 * pq.ms, 
                                 mode='zero', copy_signal=True):
    """Removes stimulation artifact by either grounding the stimulation window or
    computing and removing the average artifact template.

    Parameters
    ----------
    anas : np.array 
           2d array of analog signals
    times : quantity list
            timestamps
    trigger : quantity list
              timestamps of stimulation triggers
    pre : time quantity
          time to include before trigger times
    post : time quantity
           time to include after trigger times
    mode : string
           'zero' or 'template'
    copy_signal : bool
                  copy signals or not

    Returns
    -------
    anas_rem : analog signals after artifact removal
    avg_artifact : if 'template', average artifact removed
    """
    from copy import copy
    if copy_signal:
        anas_rem = copy(anas)
    else:
        anas_rem = anas       
    print('Removing stimulation artifacts from ', len(trigger), ' triggers...')

    if mode is 'template':
        # Find shortest artifacte length
        idxs = []
        for i, tr in enumerate(trigger):
            idxs.append(len(np.where((times > tr - pre) & (times < tr + post))[0]))

        min_len = np.min(idxs)
        templates = np.zeros((anas.shape[0], len(trigger), min_len))

        # Compute average artifact template
        for i, tr in enumerate(trigger):
            idx = np.where((times > tr - pre) & (times < tr + post))[0][:min_len]
            templates[:, i, ] = np.squeeze(anas[:, idx])
        avg_artifact = np.mean(templates, axis=1)

        # Remove average artifact template
        for i, tr in enumerate(trigger):
            idx = np.where((times > tr - pre) & (times < tr + post))[0][:min_len]
            print(anas_rem[:, idx].shape)
            anas_rem[:, idx] -= avg_artifact

    elif mode is 'zero':
        avg_artifact = []
        for tr in trigger:
            idx = np.where((times > tr - pre) & (times < tr + post))
            anas_rem[:, idx] = 0

    return anas_rem, avg_artifact


def extract_stimulation_waveform(stim, trig, times):
    """Extracts stimulation pulse parameters from stimulation signals and
    trigger times.

    Parameters
    ----------
    stim : np.array 
           2d array of stimulation analog signals
    trigger : quantity list
              timestamps of stimulation triggers
    times : quantity list
            timestamps
    
    Returns
    -------
    stim_clip : single stimulation pulse
    curr : currents for each phase (list)
    phase : phase durations (list)
    """
    period = np.mean(np.diff(times))

    done = False
    while done is False:
        if len(trig) > 1:
            rnd = np.random.permutation(len(trig)-1)[0]
            idx = np.where((times > trig[rnd]-1*period) & (times < trig[rnd+1]-10*period))
            stim_clip = np.squeeze(stim[:, idx[0]])
        else:
            stim_clip = stim

        # clip last zeros
        idx_non_zero = np.where(stim_clip[0][::-1] != 0)
        if len(idx_non_zero[0]) != 0:
            last_non_zero = stim_clip.shape[1] - np.where(stim_clip[0][::-1] != 0)[0][0]
        else:
            last_non_zero = -1
        stim_clip = stim_clip[:,:last_non_zero]

        curr = []
        phase = []
        done=False
        for ch, st in enumerate(stim_clip):
            current_levels = np.unique(stim_clip)
            transitions = np.where(np.diff(st) != 0)
            start=0
            currs=[]
            phases=[]
            for t in transitions:
                currs.append(np.round(st[start+1]).magnitude)
                phases.append(((t-start+1)*period).magnitude)
                start = start+t
            #add last phase
            currs.append(np.round(st[start+1]).magnitude)
            phases.append(((len(st)-(start+1))*period).magnitude)
            try:
                curr.append(pq.Quantity(currs, stim_clip.dimensionality))
                phase.append(pq.Quantity(phases, period.dimensionality))
                done=True
            except ValueError:
                done=False


    return stim_clip, curr, phase

def downsample_250(anas):
    """Downsamples analog signals to 250 Hz using scipy decimate.

    Parameters
    ----------
    anas : np.array 
           2d array of stimulation analog signals
    
    Returns
    -------
    out : downsampled analog signals
    """
    import neo
    import quantities as pq
    import scipy.signal as ss
    out = []
    for an in anas:
        if an.sampling_rate > 250 *pq.Hz:
            q = int(an.sampling_rate / 250 * pq.Hz)
            signal = ss.decimate(an.magnitude.T, q=q, zero_phase=True)
            sampling_rate = an.sampling_rate / q
            t_stop = len(signal) / sampling_rate
            ana = neo.AnalogSignal(signal.T * an.units, t_start=0 * pq.s,
                                   sampling_rate=sampling_rate,
                                   **an.annotations)
            out.append(ana)
        else:
            out.append(ana)
    return out
