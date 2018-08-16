from .tools import (epoch_overview, make_analog_trials, make_spiketrain_trials,
                    get_epoch, make_stimulus_trials, make_stimulus_off_epoch,
                    compute_spontan_rate, compute_osi, make_orientation_trials,
                    _convert_quantity_scalar_to_string, _convert_string_to_quantity_scalar,
                    compute_orientation_tuning)
from .stimulus_associated_latency import salt, generate_salt_trials, baysian_latency
