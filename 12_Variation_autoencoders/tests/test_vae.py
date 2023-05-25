import numpy as np
import torch
from lib.model_vae import VAE


def test_vae():

    batch_size, in_channels, in_height, in_width, hidden_size, latent_size = 2, 1, 28, 28, 100, 2
    model = VAE(in_channels, in_height, in_width, hidden_size, latent_size)
    image_batch = torch.randn((batch_size, in_channels, in_height, in_width))
    decoded_batch, mu, logvar = model(image_batch)

    assert decoded_batch.shape == image_batch.shape, (
        f"Decoded images should be shape {image_batch.shape} but are {decoded_batch.shape}")
    mu_shape = (batch_size, latent_size)
    assert mu.shape == mu_shape, f"Mean should be shape {mu_shape} but is {mu.shape}"
    assert logvar.shape == mu_shape, f"Logvar should be shape {mu_shape} but is {logvar.shape}"

    num_params_truth = 158388
    num_params = sum([np.product(p.shape) for p in model.parameters()])
    assert num_params == num_params_truth, (
        f"Model has {num_params} parameters but should have {num_params_truth}."
        f"Did you use the right amount of linear layers with the correct dimensions?")

    torch.manual_seed(1000)
    model = VAE(1, 5, 5, 20, 2)
    image_batch = torch.randn((2, 1, 5, 5))
    decoded_batch, mu, logvar = model(image_batch)
    decoded_batch = decoded_batch.detach().cpu().numpy()
    TDB = [[[[0.4945720136165619, 0.4981219172477722, 0.5727901458740234, 0.47393736243247986, 0.5422342419624329],
             [0.3777865469455719, 0.6035972833633423, 0.35935717821121216, 0.5248180031776428, 0.6372954845428467],
             [0.7058789730072021, 0.6358954310417175, 0.6242976784706116, 0.40985599160194397, 0.6531753540039062],
             [0.34155869483947754, 0.4859594702720642, 0.5383022427558899, 0.3163289725780487, 0.33808380365371704],
             [0.7438095808029175, 0.6686962842941284, 0.32844650745391846, 0.7034958004951477, 0.5731987357139587]]],

           [[[0.48010358214378357, 0.5389179587364197, 0.528987467288971, 0.3696441054344177, 0.5190967321395874],
             [0.5110399127006531, 0.5203749537467957, 0.4850541651248932, 0.5248721241950989, 0.5043338537216187],
             [0.6214423775672913, 0.5624865889549255, 0.5311568975448608, 0.4427820146083832, 0.5605618357658386],
             [0.4446558952331543, 0.464923232793808, 0.6029647588729858, 0.4420813322067261, 0.42101797461509705],
             [0.5394068360328674, 0.5145258903503418, 0.4823755919933319, 0.4975299537181854, 0.48261526226997375]]]]
    true_decoded_batch = np.array(TDB)
    err_msg = "Output dimensions of VAE model match but values are incorrect"
    np.testing.assert_allclose(decoded_batch, true_decoded_batch, rtol=1e-06, err_msg=err_msg)


if __name__ == "__main__":
    test_vae()
    print('Test complete.')
