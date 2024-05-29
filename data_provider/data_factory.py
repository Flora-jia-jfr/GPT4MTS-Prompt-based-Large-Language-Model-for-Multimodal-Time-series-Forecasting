from data_provider.data_loader import Dataset_GDELT
from torch.utils.data import DataLoader
import torch

def data_provider(args, flag, event_root_code=1, drop_last_test=True, train_all=False, data_filename=None):
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    max_len = args.max_len

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        raise Exception("unknown flag for dataset")

    data_set = Dataset_GDELT(
        root_path=args.root_path,
        data_dir=args.data_dir,
        data_filename=data_filename,
        flag=flag,
        event_root_code=event_root_code,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all,
        channel_independent=args.channel_independent,
        summary=args.summary
    )
    print(flag, len(data_set))
    try:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    except ValueError as e:
        print(f"Erro message: {e}")
        return None, None
    return data_set, data_loader
