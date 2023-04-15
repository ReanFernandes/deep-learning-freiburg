from lib.distributions import std_normal, normal
import numpy as np


def test_distributions():

    # The approximation method is inaccurate so we need lots of samples and a high error tolerance
    moments1 = np.array([-0.008635941490915258, 0.9938376265606932])
    moments2 = np.array([0.9740921755272542, 8.944538639046238])
    n_samples = 10000
    epsilon = 0.2

    # Test creation of standard normal distribution by sampling a uniform distribution.
    target_mean, target_stddev = 0., 1.
    np.random.seed(1024)
    samples = std_normal(n_samples)
    mean = samples.mean()
    var = samples.var()
    assert (mean - target_mean) ** 2 < epsilon and (var - target_stddev ** 2) ** 2 < epsilon * target_stddev, (
        f"Mean and variance should be {target_mean} and {target_stddev ** 2} but are {mean} and {var}")
    err_msg = f"Incorrect scale(b) value used for uniform distribution"
    np.testing.assert_allclose(np.array([mean, var]), moments1, rtol=1e-7, err_msg=err_msg)

    # Test normal distribution
    np.random.seed(1024)
    target_mean, target_stddev = 1., 3.
    samples = normal(target_mean, target_stddev, n_samples)
    mean = samples.mean()
    var = samples.var()
    assert (mean - target_mean) ** 2 < epsilon and (var - target_stddev ** 2) ** 2 < epsilon * target_stddev, (
        f"Mean and variance should be {target_mean} and {target_stddev ** 2} but are {mean} and {var}")
    np.testing.assert_allclose(np.array([mean, var]), moments2, rtol=1e-7, err_msg=err_msg)


if __name__ == "__main__":

    test_distributions()
    print('Test complete.')
