"""Script to test implementation of TransformerModel"""

import numpy as np
import torch
from lib.transformer import TransformerModel


def test_transformer_no_enc_dec():
    torch.manual_seed(0)

    x1 = torch.randn(2, 2, 4)
    x2 = torch.randn(2, 3, 4)

    en_vocab_size = 4
    de_vocab_size = 4
    max_len = 2
    embed_dim = 4
    enc_layers = 0
    dec_layers = 0
    num_heads = 2

    test_model = \
        TransformerModel(en_vocab_size,
                         de_vocab_size,
                         max_len,
                         embed_dim,
                         enc_layers,
                         dec_layers,
                         num_heads)
    torch.set_printoptions(precision=6)

    decoding, _ = test_model(x1, x2)

    expected_decoding = torch.tensor(
        [
            [
                [0.079619, 0.887964, 0.172323, -0.240035],
                [-0.286800, 0.139949, 0.133315, -0.234130],
                [0.062089, 0.870800, 0.255983, -0.314032],
            ],
            [
                [0.046210, 0.850558, 0.280015, -0.257313],
                [-0.079682, 0.475454, 0.405175, -0.532975],
                [-0.108757, 0.178055, 0.799308, -0.401608],
            ],
        ]
    )

    err_msg = "TransformerModel forward pass not implemented correctly"

    np.testing.assert_almost_equal(
        decoding.detach().numpy(), expected_decoding, decimal=5, err_msg=err_msg
    )


def test_transformer_enc():
    torch.manual_seed(0)

    x1 = torch.randn(2, 2, 4)
    x2 = torch.randn(2, 3, 4)

    en_vocab_size = 4
    de_vocab_size = 4
    max_len = 2
    embed_dim = 4
    enc_layers = 1
    dec_layers = 0
    num_heads = 2

    test_model = \
        TransformerModel(en_vocab_size,
                         de_vocab_size,
                         max_len,
                         embed_dim,
                         enc_layers,
                         dec_layers,
                         num_heads)

    torch.set_printoptions(precision=6)

    decoding, _ = test_model(x1, x2)

    expected_decoding = torch.tensor(
        [
            [
                [7.263068e-01, -1.014227e-01, -3.823230e-02, -1.762019e-01],
                [5.700462e-01, -3.183520e-01, -6.464455e-01, -3.496110e-04],
                [4.366536e-01, 6.375909e-03, -2.316464e-01, -2.000339e-01],
            ],
            [
                [4.548398e-01, 7.573038e-02, -3.700223e-01, -1.584404e-01],
                [-2.621710e-04, -2.487601e-01, -3.381877e-01, -3.074347e-01],
                [8.121517e-02, -4.132197e-01, -4.981286e-01, -3.680885e-01],
            ],
        ]
    )

    err_msg = "TransformerModel forward pass not implemented correctly"

    np.testing.assert_almost_equal(
        decoding.detach().numpy(), expected_decoding, decimal=5, err_msg=err_msg
    )


def test_transformer_dec():
    torch.manual_seed(0)

    x1 = torch.randn(2, 2, 4)
    x2 = torch.randn(2, 3, 4)

    en_vocab_size = 4
    de_vocab_size = 4
    max_len = 2
    embed_dim = 4
    enc_layers = 0
    dec_layers = 1
    num_heads = 2

    test_model = \
        TransformerModel(en_vocab_size,
                         de_vocab_size,
                         max_len,
                         embed_dim,
                         enc_layers,
                         dec_layers,
                         num_heads)
    torch.set_printoptions(precision=6)

    decoding, attention = test_model(x1, x2)

    expected_decoding = torch.tensor(
        [
            [
                [-0.43336588, 0.33048615, 0.97441083, 0.26286614],
                [-0.5296149, 0.8378884, 0.47893664, 0.72079796],
                [-0.473296, 0.17682296, 0.98967594, 0.12253927],
            ],
            [
                [-0.34915602, 0.16015454, 1.0366651, 0.12103006],
                [-0.82871073, 0.5969313, 0.49118805, 0.47178936],
                [-0.53416324, 0.7962548, 0.5633341, 0.67939806],
            ],
        ]
    )

    expected_attention = torch.tensor(
        [
            [
                [[0.4999357, 0.5000643], [0.4998525, 0.5001475], [0.49996793, 0.50003207]],
                [[0.5000395, 0.49996048], [0.49998537, 0.50001466], [0.4999794, 0.50002056]],
            ],
            [
                [[0.49999505, 0.500005], [0.50002605, 0.499974], [0.49999568, 0.5000043]],
                [[0.49998042, 0.50001955], [0.50005645, 0.49994355], [0.5000201, 0.49997988]],
            ],
        ]
    )

    err_msg = "TransformerModel forward pass not implemented correctly"

    np.testing.assert_almost_equal(
        decoding.detach().numpy(), expected_decoding, decimal=5, err_msg=err_msg
    )
    np.testing.assert_almost_equal(
        attention.detach().numpy(),
        expected_attention,
        decimal=5,
        err_msg=err_msg,
    )


def test_transformer():
    torch.manual_seed(0)

    x1 = torch.randn(2, 2, 4)
    x2 = torch.randn(2, 3, 4)

    en_vocab_size = 4
    de_vocab_size = 4
    max_len = 2
    embed_dim = 4
    enc_layers = 2
    dec_layers = 2
    num_heads = 2

    test_model = \
        TransformerModel(en_vocab_size,
                         de_vocab_size,
                         max_len,
                         embed_dim,
                         enc_layers,
                         dec_layers,
                         num_heads)
    torch.set_printoptions(precision=6)

    decoding, attention = test_model(x1, x2)

    expected_decoding = torch.tensor(
        [
            [
                [0.11523178, -0.05033161, 0.39944923, -0.5810448],
                [0.09748262, -0.04799473, 0.47053903, -0.6026679],
                [0.2332843, -0.08974599, 0.2677376, -0.54706573]
            ],

            [
                [0.28606832, -0.10957868, 0.1191448, -0.48608506],
                [0.0232653, -0.07632382, 0.99484867, -0.72195446],
                [-0.02033311, -0.02016366, 0.56735283, -0.5976855]
            ]
        ]
    )

    expected_attention = torch.tensor(
        [
            [
                [[0.500002, 0.49999803], [0.4999907, 0.50000936], [0.50004303, 0.49995694]],
                [[0.50003994, 0.49996006], [0.4999288, 0.5000712], [0.49995798, 0.500042]],
            ],
            [
                [[0.49972504, 0.500275], [0.4997537, 0.5002463], [0.49967343, 0.50032663]],
                [[0.4997248, 0.5002752], [0.5000423, 0.49995774], [0.49986324, 0.50013673]],
            ],
        ]
    )

    err_msg = "TransformerModel forward pass not implemented correctly"

    np.testing.assert_almost_equal(
        decoding.detach().numpy(), expected_decoding, decimal=5, err_msg=err_msg
    )
    np.testing.assert_almost_equal(
        attention.detach().numpy(),
        expected_attention,
        decimal=5,
        err_msg=err_msg,
    )


if __name__ == "__main__":
    test_transformer_no_enc_dec()
    test_transformer_enc()
    test_transformer_dec()
    test_transformer()
    print("Test complete.")
