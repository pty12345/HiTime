from data_provider.data_loader import UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'UEA': UEAloader,
}

def data_provider(args, flag, **kwargs):
    Data = data_dict[args.data]

    if flag == 'TEST':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        
    # print(flag, shuffle_flag, drop_last, batch_size)
    # exit(0)

    drop_last = False
    data_set = Data(
        dataset=args.model_id,
        root_path=args.root_path,
        flag=flag,
        **kwargs
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x)
    )
    return data_set, data_loader

