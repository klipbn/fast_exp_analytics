import fast_exp_analytics as lib


def test_package_imports():
    assert hasattr(lib, "run_ab_test")
    assert hasattr(lib, "run_abc_test")
    assert hasattr(lib, "build_dashboard_url_ab")


def test_abcd_imports():
    assert hasattr(lib, "run_abcd_test")
    assert hasattr(lib, "style_table_abcd")
    assert hasattr(lib, "make_synthetic_abcd_dataset")
    assert hasattr(lib, "export_abcd_results_to_excel")
    assert hasattr(lib, "build_abcd_chat_message")
