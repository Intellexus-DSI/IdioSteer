import importlib, pytest
def test_package_light():
    m=importlib.import_module("idiosteer")
    assert hasattr(m,"load_config") and hasattr(m,"ASSETS")
def test_config_loads():
    import idiosteer
    cfg=idiosteer.load_config()
    assert cfg["seed"]==42 and "llama3b" in cfg["models"]
@pytest.mark.parametrize("mod",["idiosteer.vectors","idiosteer.judge"])
def test_light_submodules(mod):
    importlib.import_module(mod)
def test_torch_submodules():
    pytest.importorskip("torch")
    import idiosteer.pipeline, idiosteer.hooks
    assert hasattr(idiosteer.pipeline,"MODELS")
