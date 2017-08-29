from exana.waveform.tools import calculate_waveform_features, cluster_waveform_features
from neo import SpikeTrain
import quantities as pq
import numpy as np
import pytest


def test_calculate_waveform_features_half_width_mean():
    sptr = SpikeTrain(times=np.sort(np.random.sample([3])*10)*pq.s, t_stop=10,
                      waveforms=np.array([[[5.6, 2.1, 0.3, -3.4, -5.8, -9.5,
                                            -14.7, -6.2, -1.9, 0.4, 4.9, 3.8,
                                            2.5, 0.4],
                                           [3.8, 1.3, 0.8, -0.9, -4.7, -15.7,
                                            -21.2, -7.8, -1.5, 2.4, 8.7, 5.3,
                                            4.5, 1.6]],
                                          [[4.8, 7.1, 2.4, 1.1, -2.7, -6.4,
                                            -19.5, -17.2, -7.3, -1.2, 3.5, 7.2,
                                            4.5, 2.0],
                                           [1.3, 0.6, 6.2, 0.9, -3.0, -8.1,
                                            -24.8, -16.8, -6.2, -0.5, 4.2,
                                            11.3, 6.3, 2.2]],
                                          [[2.6, 2.3, -0.1, -5.3, -9.7, -11.8,
                                            -16.2, -7.3, -4.9, -0.5, 5.3, 8.1,
                                            6.3, 2.4],
                                           [1.8, 3.4, -0.7, -5.8, -12.5, -22.1,
                                            -20.2, -13.6, -6.7, 2.8, 9.1, 4.2,
                                            4.3, 2.9]]]) * pq.uV,
                      sampling_rate=30000 * pq.Hz)
    sptrs = [sptr]
    computed_half_width_mean = calculate_waveform_features(sptrs)[0]
    expected_half_width_mean = 0.06666666269302368 * pq.ms
    success = abs(computed_half_width_mean[0] -
                  expected_half_width_mean) < 1e-14
    msg = 'Computed spike half width for mean spike != expected'
    assert success, msg


def test_calculate_waveform_features_peak_to_peak_mean():
    sptr = SpikeTrain(times=np.sort(np.random.sample([3])*10)*pq.s, t_stop=10,
                      waveforms=np.array([[[5.6, 2.1, 0.3, -3.4, -5.8, -9.5,
                                            -14.7, -6.2, -1.9, 0.4, 4.9, 3.8,
                                            2.5, 0.4],
                                           [3.8, 1.3, 0.8, -0.9, -4.7, -15.7,
                                            -21.2, -7.8, -1.5, 2.4, 8.7, 5.3,
                                            4.5, 1.6]],
                                          [[4.8, 7.1, 2.4, 1.1, -2.7, -6.4,
                                            -19.5, -17.2, -7.3, -1.2, 3.5, 7.2,
                                            4.5, 2.0],
                                           [1.3, 0.6, 6.2, 0.9, -3.0, -8.1,
                                            -24.8, -16.8, -6.2, -0.5, 4.2,
                                            11.3, 6.3, 2.2]],
                                          [[2.6, 2.3, -0.1, -5.3, -9.7, -11.8,
                                            -16.2, -7.3, -4.9, -0.5, 5.3, 8.1,
                                            6.3, 2.4],
                                           [1.8, 3.4, -0.7, -5.8, -12.5, -22.1,
                                            -20.2, -13.6, -6.7, 2.8, 9.1, 4.2,
                                            4.3, 2.9]]]) * pq.uV,
                      sampling_rate=30000 * pq.Hz)
    sptrs = [sptr]
    computed_peak_to_peak_mean = calculate_waveform_features(sptrs)[1]
    expected_peak_to_peak_mean = 0.13333335518836975 * pq.ms
    success = abs(computed_peak_to_peak_mean[0] -
                  expected_peak_to_peak_mean) < 1e-14
    msg = 'Computed peak-to-peak width for mean spike != expected'
    assert success, msg


def test_calculate_waveform_features_average_firing_rate():
    sptr = SpikeTrain(times=np.sort(np.random.sample([3])*10)*pq.s, t_stop=10,
                      waveforms=np.array([[[5.6, 2.1, 0.3, -3.4, -5.8, -9.5,
                                            -14.7, -6.2, -1.9, 0.4, 4.9, 3.8,
                                            2.5, 0.4],
                                           [3.8, 1.3, 0.8, -0.9, -4.7, -15.7,
                                            -21.2, -7.8, -1.5, 2.4, 8.7, 5.3,
                                            4.5, 1.6]],
                                          [[4.8, 7.1, 2.4, 1.1, -2.7, -6.4,
                                            -19.5, -17.2, -7.3, -1.2, 3.5, 7.2,
                                            4.5, 2.0],
                                           [1.3, 0.6, 6.2, 0.9, -3.0, -8.1,
                                            -24.8, -16.8, -6.2, -0.5, 4.2,
                                            11.3, 6.3, 2.2]],
                                          [[2.6, 2.3, -0.1, -5.3, -9.7, -11.8,
                                            -16.2, -7.3, -4.9, -0.5, 5.3, 8.1,
                                            6.3, 2.4],
                                           [1.8, 3.4, -0.7, -5.8, -12.5, -22.1,
                                            -20.2, -13.6, -6.7, 2.8, 9.1, 4.2,
                                            4.3, 2.9]]]) * pq.uV,
                      sampling_rate=30000 * pq.Hz)
    sptrs = [sptr]
    computed_average_firing_rate = calculate_waveform_features(sptrs)[2]
    expected_average_firing_rate = (float(3) / (10 * pq.s))
    success = abs(computed_average_firing_rate[0] -
                  expected_average_firing_rate) < 1e-10
    msg = 'Computed average firing rate != expected'
    assert success, msg


def test_calculate_waveform_features_half_width_all_spikes():
    sptr = SpikeTrain(times=np.sort(np.random.sample([3])*10)*pq.s, t_stop=10,
                      waveforms=np.array([[[5.6, 2.1, 0.3, -3.4, -5.8, -9.5,
                                            -14.7, -6.2, -1.9, 0.4, 4.9, 3.8,
                                            2.5, 0.4],
                                           [3.8, 1.3, 0.8, -0.9, -4.7, -15.7,
                                            -21.2, -7.8, -1.5, 2.4, 8.7, 5.3,
                                            4.5, 1.6]],
                                          [[4.8, 7.1, 2.4, 1.1, -2.7, -6.4,
                                            -19.5, -17.2, -7.3, -1.2, 3.5, 7.2,
                                            4.5, 2.0],
                                           [1.3, 0.6, 6.2, 0.9, -3.0, -8.1,
                                            -24.8, -16.8, -6.2, -0.5, 4.2,
                                            11.3, 6.3, 2.2]],
                                          [[2.6, 2.3, -0.1, -5.3, -9.7, -11.8,
                                            -16.2, -7.3, -4.9, -0.5, 5.3, 8.1,
                                            6.3, 2.4],
                                           [1.8, 3.4, -0.7, -5.8, -12.5, -22.1,
                                            -20.2, -13.6, -6.7, 2.8, 9.1, 4.2,
                                            4.3, 2.9]]]) * pq.uV,
                      sampling_rate=30000 * pq.Hz)
    sptrs = [sptr]
    comp_hw = calculate_waveform_features(sptrs, calc_all_spikes=True)[3][0]
    computed_half_width_all_spikes = comp_hw.rescale('ms').magnitude
    expected_half_width_all_spikes = np.array([0.06666666, 0.06666666,
                                               0.09999999])
    success = np.allclose(computed_half_width_all_spikes,
                          expected_half_width_all_spikes)
    msg = 'Computed half width for all spikes != expected'
    assert success, msg


def test_calculate_waveform_features_peak_to_peak_all_spikes():
    sptr = SpikeTrain(times=np.sort(np.random.sample([3])*10)*pq.s, t_stop=10,
                      waveforms=np.array([[[5.6, 2.1, 0.3, -3.4, -5.8, -9.5,
                                            -14.7, -6.2, -1.9, 0.4, 4.9, 3.8,
                                            2.5, 0.4],
                                           [3.8, 1.3, 0.8, -0.9, -4.7, -15.7,
                                            -21.2, -7.8, -1.5, 2.4, 8.7, 5.3,
                                            4.5, 1.6]],
                                          [[4.8, 7.1, 2.4, 1.1, -2.7, -6.4,
                                            -19.5, -17.2, -7.3, -1.2, 3.5, 7.2,
                                            4.5, 2.0],
                                           [1.3, 0.6, 6.2, 0.9, -3.0, -8.1,
                                            -24.8, -16.8, -6.2, -0.5, 4.2,
                                            11.3, 6.3, 2.2]],
                                          [[2.6, 2.3, -0.1, -5.3, -9.7, -11.8,
                                            -16.2, -7.3, -4.9, -0.5, 5.3, 8.1,
                                            6.3, 2.4],
                                           [1.8, 3.4, -0.7, -5.8, -12.5, -22.1,
                                            -20.2, -13.6, -6.7, 2.8, 9.1, 4.2,
                                            4.3, 2.9]]]) * pq.uV,
                      sampling_rate=30000 * pq.Hz)
    sptrs = [sptr]
    comp_ptp = calculate_waveform_features(sptrs, calc_all_spikes=True)[4][0]
    computed_peak_to_peak_all_spikes = comp_ptp.rescale('ms').magnitude
    expected_peak_to_peak_all_spikes = np.array([0.13333336, 0.16666669,
                                                 0.16666667])
    success = np.allclose(computed_peak_to_peak_all_spikes,
                          expected_peak_to_peak_all_spikes)
    msg = 'Computed peak-to-peak width for all spikes != expected'
    assert success, msg


def test_calculate_waveform_features_attributeerror():
    sptr = SpikeTrain(times=np.sort(np.random.sample([3])*10)*pq.s, t_stop=10,
                      sampling_rate=30000 * pq.Hz)
    sptrs = [sptr]
    assert pytest.raises(AttributeError, calculate_waveform_features, sptrs)


def test_cluster_idx():
    half_width_mean = [0.12322092, 0.30465815, 0.10547199, 0.17453667,
                       0.38877204, 0.33574967, 0.39441669, 0.18472238,
                       0.37264262, 0.35206679, 0.15082797, 0.1389273,
                       0.1955586]
    peak_to_peak_mean = [0.23485475, 0.52386796, 0.15283638, 0.25323685,
                         0.69555114, 0.58154619, 0.63756934, 0.29113269,
                         0.65643368, 0.55268553, 0.17157666, 0.27780348,
                         0.1981366]
    idx = cluster_waveform_features(half_width_mean, peak_to_peak_mean,
                                    nr_clusters=2)[0]
    if idx[0] == 0:
        msg = 'Computed idx != expected'
        success = np.array_equal(np.array(idx), np.array([0, 1, 0, 0, 1, 1, 1,
                                                          0, 1, 1, 0, 0, 0]))
        assert success, msg
    elif idx[0] == 1:
        msg = 'Computed idx != expected'
        success = np.array_equal(np.array(idx), np.array([1, 0, 1, 1, 0, 0, 0,
                                                          1, 0, 0, 1, 1, 1]))
        assert success, msg
    else:
        print('Error while computing idx. Idx should only contain 0s and 1s\
               when number of clusters is 2.')


def test_cluster_waveform_features_valueerror():
    half_width_mean = [0.12322092, 0.30465815, 0.10547199, 0.17453667,
                       0.38877204, 0.33574967, 0.39441669, 0.18472238,
                       0.37264262, 0.35206679, 0.15082797, 0.1389273,
                       0.1955586]
    peak_to_peak_mean = [0.23485475, 0.52386796, 0.15283638, 0.25323685,
                         0.69555114, 0.58154619, 0.63756934, 0.29113269,
                         0.65643368, 0.55268553, 0.17157666, 0.27780348,
                         0.1981366]
    assert pytest.raises(ValueError, cluster_waveform_features,
                         half_width_mean, peak_to_peak_mean, nr_clusters=6)
    assert pytest.raises(ValueError, cluster_waveform_features,
                         half_width_mean, peak_to_peak_mean, nr_clusters=1)


def test_cluster_waveform_features_len_groups():
    half_width_mean = [0.12322092, 0.30465815, 0.10547199, 0.17453667,
                       0.38877204, 0.33574967, 0.39441669, 0.18472238,
                       0.37264262, 0.35206679, 0.15082797, 0.1389273,
                       0.1955586]
    peak_to_peak_mean = [0.23485475, 0.52386796, 0.15283638, 0.25323685,
                         0.69555114, 0.58154619, 0.63756934, 0.29113269,
                         0.65643368, 0.55268553, 0.17157666, 0.27780348,
                         0.1981366]
    idx, red_group, blue_group = \
        cluster_waveform_features(half_width_mean, peak_to_peak_mean)
    if idx[0] == 0:
        success1 = len(red_group) == 7
        success2 = len(blue_group) == 6
        msg = 'Length of computed groups not as expected'
        assert (success1 and success2), msg
    elif idx[0] == 1:
        success1 = len(red_group) == 6
        success2 = len(blue_group) == 7
        msg = 'Length of computed groups not as expected'
        assert (success1 and success2), msg


def test_cluster_waveform_features_content_groups():
    sptrs = [2, 3, 2, 2, 3, 3, 3, 2, 3, 3, 2, 2, 2]
    half_width_mean = [0.12322092, 0.30465815, 0.10547199, 0.17453667,
                       0.38877204, 0.33574967, 0.39441669, 0.18472238,
                       0.37264262, 0.35206679, 0.15082797, 0.1389273,
                       0.1955586]
    peak_to_peak_mean = [0.23485475, 0.52386796, 0.15283638, 0.25323685,
                         0.69555114, 0.58154619, 0.63756934, 0.29113269,
                         0.65643368, 0.55268553, 0.17157666, 0.27780348,
                         0.1981366]
    idx, red_group, blue_group = \
        cluster_waveform_features(half_width_mean, peak_to_peak_mean)
    if idx[0] == 0:
        success1 = np.array_equal(np.array(red_group),
                                  np.array([0, 2, 3, 7, 10, 11, 12]))
        success2 = np.array_equal(np.array(blue_group),
                                  np.array([1, 4, 5, 6, 8, 9]))
        msg = 'Content of computed groups not as expected'
        assert (success1 and success2), msg
    elif idx[0] == 1:
        success1 = np.array_equal(np.array(red_group),
                                  np.array([1, 4, 5, 6, 8, 9]))
        success2 = np.array_equal(np.array(blue_group),
                                  np.array([0, 2, 3, 7, 10, 11, 12]))
        msg = 'Content of computed groups not as expected'
        assert (success1 and success2), msg
