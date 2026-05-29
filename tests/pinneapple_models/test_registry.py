def test_models_registry_smoke():
    from pinneapple_models.registry import ModelRegistry

    try:
        from pinneapple_models.register_all import register_all
        register_all()
    except Exception:
        pass

    names = ModelRegistry.list()
    assert isinstance(names, list)

    if names:
        name0 = names[0]
        for kwargs in ({}, {"input_dim": 8, "output_dim": 2}, {"input_dim": 8, "latent_dim": 2}):
            try:
                m = ModelRegistry.build(name0, **kwargs)
                assert m is not None
                break
            except TypeError:
                continue

