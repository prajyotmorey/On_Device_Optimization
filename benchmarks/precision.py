def get_model_precision(model):

    dtypes = set()

    for parameter in model.parameters():
        dtypes.add(str(parameter.dtype))

    return ",".join(sorted(dtypes))