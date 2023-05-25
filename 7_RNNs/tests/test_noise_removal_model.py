"""Script to test NoiseRemovalModel"""

import numpy as np
import torch

from lib.models import NoiseRemovalModel
from lib.utilities import prepare_sequences, sample_sine_functions


def test_noise_removal_model_output_shape():
    torch.manual_seed(0)

    val_functions = sample_sine_functions(20)
    _, noisy_val_sequences = prepare_sequences(val_functions)
    model = NoiseRemovalModel(hidden_size=6, shift=10)
    output = model(noisy_val_sequences)
    expected_output_shape = (20, 80, 1)

    err_msg = 'The output shape of NoiseRemovalModel is incorrect'
    np.testing.assert_equal(
        output.shape,
        expected_output_shape,
        err_msg=err_msg)


def test_noise_removal_model_output():
    torch.manual_seed(0)
    np.random.seed(0)
    val_functions = sample_sine_functions(1)
    _, noisy_val_sequences = prepare_sequences(val_functions)
    model = NoiseRemovalModel(hidden_size=3, shift=10)
    output = model(noisy_val_sequences)
    output_arr = output.detach().numpy()
    expected_output = \
        np.array([[[0.07554727], [0.11317387], [0.1299254], [0.13728365], [0.14028195],
                   [0.14163473], [0.14208713], [0.14234711], [0.14235026], [0.1423347],
                   [0.142334], [0.14236048], [0.14222015], [0.14231044], [0.14227742],
                   [0.14233533], [0.14226574], [0.14222425], [0.1420992], [0.1420377],
                   [0.14198445], [0.14191724], [0.1418446], [0.14177987], [0.14173663],
                   [0.14173615], [0.14183725], [0.1419129], [0.14201656], [0.1421648],
                   [0.14230525], [0.14237851], [0.14246511], [0.14262533], [0.1427499],
                   [0.1428181], [0.1428855], [0.14286754], [0.14279327], [0.14278865],
                   [0.14286438], [0.14287041], [0.14257006], [0.14260471], [0.1424693],
                   [0.14234607], [0.14224245], [0.14231978], [0.14229956], [0.14231105],
                   [0.14229691], [0.14224564], [0.14221434], [0.14217232], [0.14219849],
                   [0.14205307], [0.1419358], [0.14183368], [0.14177781], [0.14174089],
                   [0.14174452], [0.14174533], [0.14179328], [0.14188184], [0.14194363],
                   [0.1420848], [0.14230172], [0.14243399], [0.14256854], [0.14264703],
                   [0.14277786], [0.1427066], [0.14263943], [0.14259148], [0.14256063],
                   [0.14254183], [0.14253063], [0.14252406], [0.14252016], [0.14251786]]], dtype=float)

    err_msg = 'The output of NoiseRemovalModel is incorrect'
    np.testing.assert_allclose(
        output_arr,
        expected_output,
        atol=1e-5,
        err_msg=err_msg)


if __name__ == '__main__':
    test_noise_removal_model_output_shape()
    print('Test complete.')
    test_noise_removal_model_output()
    print('Test complete.')
