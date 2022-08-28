import dense_visual_odometry.version as dvo


def test__version_type():
    assert(isinstance(dvo.__version__, str))  # noqa: E275
    assert dvo.__version__ == "0.0.1"
