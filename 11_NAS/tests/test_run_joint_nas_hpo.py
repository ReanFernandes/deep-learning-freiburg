import os
import shutil
import numpy as np
import neps

from lib.utilities import set_seed, get_results
from run_joint_nas_hpo import run_pipeline, pipeline_space


def test_joint_nas_hpo():
    set_seed(124)
    err_msg = "run_joint_nas_hpo pipeline is not implemented correctly"
    if os.path.exists("results/test_joint_nas_hpo"):
        shutil.rmtree("results/test_joint_nas_hpo")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/test_joint_nas_hpo",
        max_evaluations_total=1,
        searcher="bayesian_optimization"
    )

    results = get_results("results/test_joint_nas_hpo")
    np.testing.assert_allclose(results["Losses"][0], 0.867999999821186, atol=1e-1, err_msg=err_msg)
    np.testing.assert_allclose(results["Val_errors"][0],
                               np.array(
                                   [0.8720000010579825, 0.892, 0.8740000002682209, 0.8820000003278256,
                                    0.867999999821186]
                               ), rtol=1e-1, err_msg=err_msg)
    np.testing.assert_allclose(results["Val_loss"][0],
                               np.array(
                                   [2.3149435157775877, 2.3533572149276734, 2.3362564611434937, 2.3220002698898314,
                                    2.3099552936553955]
                               ), rtol=1e-1, err_msg=err_msg)
    np.testing.assert_string_equal(results["Config"][0]["architecture"],
                                   '(C DenseCell (OPS AvgPool1x1) (OPS ReLUConvBN1x1) (OPS ReLUConvBN1x1) '
                                   '(OPS ReLUConvBN3x3) (OPS ReLUConvBN3x3) (OPS AvgPool1x1))')


if __name__ == "__main__":
    test_joint_nas_hpo()
    print("Test complete.")
