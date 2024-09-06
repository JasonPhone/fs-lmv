import copy


models = {}


# 装饰器函数，将模型类注册到models字典中
def RegisterModel(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


# 加载一个模型
def loadModel(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec["args"])
        model_args.update(args)
    else:
        model_args = model_spec["args"]
    if model_args is not None:
        model = models[model_spec["name"]](**model_args)
    else:
        model = models[model_spec["name"]]()

    if load_sd:
        model.load_state_dict(model_spec["sd"])
    return model.cuda()
