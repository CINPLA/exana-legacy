import expipe
import expipe.io
import os
import os.path as op
import numpy as np
from datetime import datetime
import quantities as pq


def auto_denoise(anas, thresh=None):
    '''
    Clean neural data from EMG, chewing, and moving artifact noise
    Rectified signals are smoothed and thresholded to find and remove noisy portion of the signals
    :param anas: np.array analog signals
    :param thresh: (optional) threshold in number of SD on high-pass data
    :return: cleaned_anas
    '''
    from scipy import signal
    from copy import copy
    if thresh:
        thresh = thresh
    else:
        thresh = 2.5

    anas_copy = copy(anas)
    # anas_copy = np.abs(anas_copy)

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


def manual_denoise(anas, thresh=None):
    '''
    Clean neural data from EMG, chewing, and moving artifact noise
    User can select the points to cut out for the denoised signal
    :param anas: np.array analog signals
    :param thresh: (optional) threshold in number of SD on high-pass data
    :return: cleaned_anas
    '''
    from copy import copy
    import matplotlib.pyplot as plt
    anas_copy = copy(anas)

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

    # return anas_copy

def ica_denoise(anas, channels=None, n_comp=None, correlation_thresh=None):
    '''
    Removes noise by ICA. Indepentend components highly correlated to the grand average of the signals are removed.
    Signals are then back projected to the channel space.
    :param anas: analog signals (N channels by T time samples)
    :param channels: channels to be used
    :param n_comp: number of IC
    :param correlation_thresh: threshold on correlation value to identify noisy IC
    :return: anas_sig - cleaned signals
             anas_noise - noise contribution on each channel

    '''
    from sklearn.decomposition import FastICA
    from scipy import stats
    if channels is None:
        channels = np.arange(anas.shape[0])
    if n_comp is None:
        n_comp = 10
    if correlation_thresh is None:
        correlation_thresh = 0.1

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

    # Substitute bad channels with zeros
    anas_sig = np.zeros(anas.shape)
    anas_noise = np.zeros(anas.shape)

    idx_sig = np.where(corr <= correlation_thresh)
    idx_noise = np.where(corr > correlation_thresh)
    anas_sig[channels] = np.squeeze(np.dot(mixing[:, idx_sig], sources[idx_sig]))
    anas_noise[channels] = np.squeeze(np.dot(mixing[:, idx_noise], sources[idx_noise]))

    return anas_sig, anas_noise, sources, mixing, corr


def apply_CAR(anas, channels=None, car_type='mean', split_probe=None):
    '''

    :param anas:
    :param channels:
    :param car_type:
    :param split_probe:
    :return:
    '''
    from copy import copy
    if channels is None:
        channels = np.arange(anas.shape[0])
    anas_car = copy(anas)
    del anas
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
            avg_ref = np.median(anas_car[:split_probe], axis=0)
            anas_car[:split_probe] -= avg_ref
            avg_ref = np.median(anas_car[split_probe:], axis=0)
            anas_car[split_probe:] -= avg_ref
        else:
            avg_ref = np.median(anas_car[channels], axis=0)
            anas_car[channels] -= avg_ref
    else:
        raise AttributeError("'type must be 'mean' or 'median'")

    return anas_car, avg_ref


# TODO use quantities and deal with it
# moved to pyopenephys and pyintan
def extract_sync_times(adc_signal, times):
    '''

    :param adc_signal: analog_signal with sync event ('1' is > 1.65V)
    :param times: array of timestamps of analog signal
    :return: array with rising times
    '''
    idx_high = np.where(adc_signal>1.65)[0]

    rising = []

    if len(idx_high) != 0:
        for i, idx in enumerate(idx_high[:-1]):
            if i==0:
                # first idx is rising
                rising.append(idx)
            elif idx - 1 != idx_high[i-1]:
                rising.append(idx)

    return np.array(times[rising])


def clip_anas(anas, times, clip_times):
    '''

    :param anas:
    :param times:
    :param clip_times:
    :return:
    '''
    if len(clip_times) == 2:
        idx = np.where((times > clip_times[0]) & (times < clip_times[1]))
    elif len(clip_times) ==  1:
        idx = np.where(times > clip_times[0])
    else:
        raise AttributeError('clip_times must be of length 1 or 2')

    if len(anas.shape) == 2:
        anas_clip = anas[:, idx[0]]
    else:
        anas_clip = anas[idx[0]]

    return anas_clip


def clip_digs(digs, clip_times):
    '''

    :param digs:
    :param clip_times:
    :return:
    '''

    digs_clip = []
    if digs.shape == 2:
        for i, dig in enumerate(digs):
            if len(clip_times) == 2:
                idx = np.where((dig > clip_times[0]) & (dig < clip_times[1]))
            elif len(clip_times) == 1:
                idx = np.where(dig > clip_times[0])
            else:
                raise AttributeError('clip_times must be of length 1 or 2')
            digs_clip.append(dig[idx])
    elif digs.shape == 1:
        if len(clip_times) == 2:
            idx = np.where((digs > clip_times[0]) & (digs < clip_times[1]))
        elif len(clip_times) == 1:
            idx = np.where(digs > clip_times[0])
        else:
            raise AttributeError('clip_times must be of length 1 or 2')
        digs_clip = digs[idx]

    return digs_clip


def clip_times(times, clip_times):
    '''

    :param times:
    :param clip_times:
    :return:
    '''

    if len(clip_times) == 2:
        idx = np.where((times > clip_times[0]) & (times < clip_times[1]))
    elif len(clip_times) ==  1:
        idx = np.where(times > clip_times[0])
    else:
        raise AttributeError('clip_times must be of length 1 or 2')
    times_clip = (times[idx])

    return times_clip


def set_timestamps_from_events(software_ts, ttl_events):
    '''

    :param software_ts:
    :param ttl_events:
    :return:
    '''

    # For each software ts find closest ttl_event
    ts = np.zeros(len(software_ts))
    ttl_idx = -1*np.ones(len(software_ts), dtype='int64')

    for i, s_ts in enumerate(software_ts):
        ts[i], ttl_idx[i] = find_nearest(ttl_events, s_ts)

    # A late osc msg might result in an error -> find second closest timestamp in those cases
    wrong_ts_idx = np.where(np.diff(ts) == 0)[0]
    iteration=1
    max_iter=10

    while len(wrong_ts_idx) != 0 and iteration<max_iter:
        print('wrong assignments: ', len(wrong_ts_idx), ' Iteration: ', iteration)

        for i, w_ts in enumerate(wrong_ts_idx):
            val, idx = find_nearest(ttl_events, software_ts[w_ts], not_in_idx=np.unique(ttl_idx))
            ts[w_ts] = val[0]
            ttl_idx[w_ts] = idx[0]
        iteration +=1
        wrong_ts_idx = np.where(np.diff(ts) == 0)[0]

    return ts, ttl_idx


def find_nearest(array, value, n=1, not_in_idx=None):

    if not_in_idx is None:
        if n==1:
            idx = (np.abs(array-value)).argmin()
        else:
            idx = (np.abs(array-value)).argsort()[:n]
        return array[idx], idx
    else:
        if len(array) != 0:
            left_idx = np.ones(len(array), dtype=bool)
            left_idx[not_in_idx] = False
            left_array=array[left_idx]
            if n==1:
                idx = (np.abs(left_array-value)).argmin()
            else:
                idx = (np.abs(left_array-value)).argsort()[:n]
            val = left_array[idx]
            idx = np.where(array==val)
            return array[idx], idx
        else:
            print('Array length must be greater than 0')
            return None, -1


def filter_analog_signals(anas, freq, fs, filter_type='bandpass', order=3):
    '''

    :param anas:
    :param freq:
    :param fs:
    :param filter_type:
    :param order:
    :return:
    '''
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


def ground_bad_channels(anas, bad_channels):
    '''

    :param anas:
    :param bad_channels:
    :return:
    '''

    print('Grounding channels: ', bad_channels, '...')

    from copy import copy
    nsamples = anas.shape[1]
    anas_zeros = copy(anas)
    if type(bad_channels) is not list:
        bad_channels = [bad_channels]

    for i, ana in enumerate(anas_zeros):
        if i in bad_channels:
            anas_zeros[i] = np.zeros(nsamples)

    return anas_zeros


def duplicate_bad_channels(anas, bad_channels, probefile):
    '''

    :param anas:
    :param bad_channels:
    :param probefile:
    :return:
    '''

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
    anas_dup = copy(anas)
    if type(bad_channels) is not list:
        bad_channels = [bad_channels]

    for i, ana in enumerate(anas_dup):
        if i in bad_channels:
            rnd = _select_rnd_chan_in_group(probefile_ch_mapping, i)
            anas_dup[i] = anas[rnd]

    return anas_dup



def save_binary_format(filename, signal, spikesorter='klusta'):
    '''

    :param signal:
    :param spikesorter:
    :param filefolder:
    :param filename:
    :return:
    '''
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
    '''

    :param directory_or_file:
    :param prb_path:
    :param nchan:
    :param fs:
    :param klusta_filter:
    :param filter_low:
    :param filter_high:
    :return:
    '''
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


def remove_stimulation_artifacts(anas, times, trigger, pre=3 * pq.ms, post=5 * pq.ms, mode='template'):
    '''

    :param anas:
    :param times:
    :param trigger:
    :param pre:
    :param post:
    :param mode:
    :return:
    '''
    from copy import copy
    anas_rem = copy(anas)

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


def extract_stimulation_waveform(stim, triggers, times):
    '''

    :param stim:
    :param triggers:
    :param times:
    :return:
    '''
    period = np.mean(np.diff(times))

    if len(triggers) > 1:
        idx = np.where((times > triggers[0]-1*period) & (times < triggers[1]))
        stim_clip = np.squeeze(stim[:, idx[0]])
    else:
        stim_clip = stim

    # clip last zeros
    last_non_zero = stim_clip.shape[1] - np.where(stim_clip[0][::-1] != 0)[0][0]
    stim_clip = stim_clip[:,:last_non_zero]

    curr = []
    phase = []

    for ch, st in enumerate(stim_clip):
        current_levels = np.unique(stim_clip)
        transitions = np.where(np.diff(st) != 0)
        start=0
        currs=[]
        phases=[]
        for t in transitions:
            print(start)
            currs.append(np.round(st[start+1]))
            phases.append((t-start+1)*period)
            start = start+t
        print(start)
        #add last phase
        currs.append(np.round(st[start+1]))
        phases.append((len(st)-(start+1))*period)

        curr.append(currs)
        phase.append(phases)

    print(np.round(current_levels))
    return stim_clip, curr, phase


def downsample_250(anas):
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
