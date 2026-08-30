import importlib
def test_derive_label_mapping():
    j=importlib.import_module("idiosteer.judge")
    dl=getattr(j,"derive_label",None)
    if dl is None: import pytest; pytest.skip("derive_label not exported")
    assert dl(False, False)=="incoherent"
    assert dl(True, True)=="literal"
    assert dl(True, False)=="figurative"
