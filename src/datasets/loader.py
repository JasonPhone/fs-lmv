import copy


datasets = {}


def RegisterDataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def loadDataset(dataset_spec, args=None):
    dataset_args = dict()

    if dataset_spec["args"] is not None:
        dataset_args.update(dataset_spec["args"])
    if args is not None:
        dataset_args.update(args)

    # if args is not None:
    #     dataset_args = copy.deepcopy(dataset_spec['args'])
    #     dataset_args.update(args)
    # else:
    #     dataset_args = dataset_spec['args']
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset
