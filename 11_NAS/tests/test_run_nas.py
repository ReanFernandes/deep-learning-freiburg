import os
import shutil
import numpy as np
import neps

from lib.utilities import set_seed, get_results
from run_nas import run_pipeline, pipeline_space


def test_nas():
    set_seed(124)
    err_msg = "run_nas pipeline is not implemented correctly"
    if os.path.exists("results/test_nas"):
        shutil.rmtree("results/test_nas")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/test_nas",
        max_evaluations_total=1,
        searcher="bayesian_optimization"
    )
    results = get_results("results/test_nas")
    np.testing.assert_allclose(results["Losses"][0], 0.7839999995231628, rtol=1e-1, err_msg=err_msg)
    np.testing.assert_allclose(results["Val_errors"][0],
                               np.array(
                                   [0.8579999992847442, 0.8319999995231628, 0.8219999992847442, 0.7879999998807907,
                                    0.7839999995231628]
                               ), rtol=1e-1, err_msg=err_msg)
    np.testing.assert_allclose(results["Val_loss"][0],
                               np.array(
                                   [2.2835668239593505, 2.25950959777832, 2.224282934188843, 2.1688121700286866,
                                    2.1727334079742433]
                               ), rtol=1e-1, err_msg=err_msg)
    np.testing.assert_string_equal(results["Config"][0]["architecture"],
                                   '(C DenseCell (OPS AvgPool1x1) (OPS ReLUConvBN1x1) (OPS ReLUConvBN1x1) '
                                   '(OPS ReLUConvBN3x3) (OPS ReLUConvBN3x3) (OPS AvgPool1x1))')


if __name__ == "__main__":
    test_nas()
    print("Test complete.")
