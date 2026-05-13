import fast_exp_analytics as lib


def test_package_imports():
    assert hasattr(lib, "run_ab_test")
    assert hasattr(lib, "run_abc_test")
    assert hasattr(lib, "build_dashboard_url_ab")
