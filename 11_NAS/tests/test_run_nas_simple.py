import os
import shutil
import numpy as np
import neps

from lib.utilities import set_seed, get_results
from run_nas_simple import run_pipeline, pipeline_space


def test_nas_simple():
    set_seed(124)
    err_msg = "run_nas_simple pipeline is not implemented correctly"
    if os.path.exists("results/test_nas_simple"):
        shutil.rmtree("results/test_nas_simple")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/test_nas_simple",
        max_evaluations_total=1,
        searcher="bayesian_optimization",
    )
    results = get_results("results/test_nas_simple")
    np.testing.assert_allclose(results["Losses"][0], 0.7479999996423721, rtol=1e-1, err_msg=err_msg)
    np.testing.assert_allclose(results["Val_errors"][0],
                               np.array(
                                   [0.8919999994039536, 0.7159999985694885, 0.7339999997615814, 0.7599999997615814,
                                    0.7479999996423721]
                               ), rtol=1e-1, err_msg=err_msg)
    np.testing.assert_allclose(results["Val_loss"][0],
                               np.array(
                                   [2.2817873554229737, 2.011141532897949, 2.0086560249328613, 2.263609830856323,
                                    2.2587853336334227]
                               ), rtol=1e-1, err_msg=err_msg)
    np.testing.assert_string_equal(results["Config"][0]["architecture"],
                                   '(C DenseCell (OPS AvgPool1x1) (OPS ReLUConvBN3x3) (OPS ReLUConvBN3x3))')


if __name__ == "__main__":
    test_nas_simple()
    print("Test complete.")
