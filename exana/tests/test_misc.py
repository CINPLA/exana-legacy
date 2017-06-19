import pytest
import numpy as np
import neo
import quantities as pq


def test_concatenate_spiketrains():
    from exana.misc import concatenate_spiketrains
    spiketrain1 = neo.SpikeTrain(times=np.arange(10),
                                 t_stop=10, units='s',
                                 waveforms=np.ones((10,1,5)) * pq.V)
    spiketrain2 = neo.SpikeTrain(times=np.arange(10, 25),
                                 t_stop=25, units='s',
                                 waveforms=np.ones((15,1,5)) * pq.V)
    spiketrain = concatenate_spiketrains([spiketrain1, spiketrain2])
    spiketrain_true = np.concatenate((np.arange(10), np.arange(10, 25)))
    waveforms_true = np.concatenate((np.ones((10,1,5)), np.ones((15,1,5))))
    assert np.array_equal(spiketrain.times.magnitude, spiketrain_true)
    assert spiketrain.times.units == pq.s.units
    assert np.array_equal(spiketrain.waveforms.magnitude, waveforms_true)
    assert spiketrain.waveforms.units == pq.V
    assert spiketrain.t_stop == 25 * pq.s

    with pytest.raises(ValueError):
        spiketrain2 = neo.SpikeTrain(times=np.arange(10, 25),
                                     t_stop=25, units='ms',
                                     waveforms=np.ones((15,1,5)) * pq.V)
        spiketrain = concatenate_spiketrains([spiketrain1, spiketrain2])
