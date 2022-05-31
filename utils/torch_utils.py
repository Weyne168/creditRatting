import torch


def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(force_cpu=False, gpus='0'):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if not cuda:
        print('Using CPU')

    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = min(torch.cuda.device_count(), len(gpus))
        if len(gpus) > torch.cuda.device_count():
            x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        else:
            x = [torch.cuda.get_device_properties(i) for i in gpus]
        print("Using CUDA device0 _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
              (x[0].name, x[0].total_memory / c))
        if ng > 0:
            # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
            for i in range(1, ng):
                print("           device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                      (i, x[i].name, x[i].total_memory / c))

    return device


def opts_parser(opts):
    dims = opts.cross_feats.split(',')
    ds = []
    for d in dims:
        ds.append(int(d))
    opts.cross_feats = ds

    dims = opts.cate_nums.split(',')
    ds = []
    for d in dims:
        ds.append(int(d))
    opts.cate_nums = ds

    gpus = opts.gpus.split(',')
    ds = []
    for d in gpus:
        ds.append(int(d))
    opts.gpus = ds

    args = vars(opts)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    return opts
