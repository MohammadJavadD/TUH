
import tempfile
import numpy as np 

#
import mne
mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

#
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGResNet, EEGNetv4


#
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import torch.nn.functional as F


#
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="main-args")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #load preproccessed data
    dataset_loaded = load_concat_dataset(
        path=cfg.args.TUH_PP_PATH,
        preload=True,
        ids_to_load= range(100), # [10, 30],
        # target_name=('age', 'gender'),
    )

    # We can finally generate compute windows. The resulting dataset is now ready
    # we will create compute windows. We specify a
    # mapping from genders 'M' and 'F' to integers, since this is required for
    # decoding.

    tuh_windows = create_fixed_length_windows(
        dataset_loaded,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=1000,
        window_stride_samples=1000,
        drop_last_window=False,
        mapping={'M': 0, 'F': 1},  # map non-digit targets
    )
    # store the number of windows required for loading later on
    tuh_windows.set_description({
        "n_windows": [len(d) for d in tuh_windows.datasets]})


    ###############################################################################
    # Iterating through the dataset gives x as ndarray(n_channels x 1000), y as
    # [age, gender], and ind. Let's look at the last example again.
    # print(tuh_windows.description)
    x, y, ind = tuh_windows[-1]
    print('x:', x.shape)
    print('y:', y)
    print('ind:', ind)

    ###
    # split
    # import random
    # inds = random.shuffle(range(ind))
    tuh_splits = tuh_windows.split("version")
    # {"train": inds[:int(70*ind)], "valid": inds[int(70*ind):int(90*ind)], "test": inds[int(90*ind)]}
    # )
    ###############################################################################
    # We give the dataset to a pytorch DataLoader, such that it can be used for
    # model training.
    dl_train = DataLoader(
        dataset=tuh_splits["train"],
        batch_size=64,
    )
    dl_eval = DataLoader(
        dataset=tuh_splits["eval"],
        batch_size=128,
    )

    ###############################################################################
    # Iterating through the DataLoader gives batch_X as tensor(4 x n_channels x
    # 1000), batch_y as [tensor([4 x age of subject]), tensor([4 x gender of
    # subject])], and batch_ind. We will iterate to the end to look at the last example
    # again.
    for batch_X, batch_y, batch_ind in dl_train:
        pass
    print('batch_X:', batch_X.shape)
    print('batch_y:', batch_y)
    print('batch_ind:', batch_ind)

    ## training 
    model = Deep4Net(
             in_chans = batch_X.shape[1],
            n_classes = 2,
            input_window_samples=batch_X.shape[2],
            final_conv_length='auto',
            n_filters_time=32,
            n_filters_spat=32,
            filter_time_length=10,
            pool_time_length=3,
            pool_time_stride=3,
            n_filters_2=64,
            filter_length_2=10,
            n_filters_3=128,
            filter_length_3=10,
            n_filters_4=256,
            filter_length_4=10
        )
    optimizer = optim.Adam(model.parameters(), lr=cfg.args.lr, weight_decay=cfg.args.weight_decay)

    for ii in range(cfg.args.epochs): 
        losses = []
        for batch_X, batch_y, batch_ind in dl_train:
            pred = model(batch_X)
            loss = F.cross_entropy(pred, batch_y[1])
            losses.append(loss.detach().numpy())

            # Back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print ('epoch:',ii, 'loss:', np.array(losses).mean() )
    
    # validatin
    from sklearn.metrics import accuracy_score
    y_pred = []
    y_true = [] 
    for batch_X, batch_y, batch_ind in dl_eval:
        model.eval()
        y_pred.extend(torch.argmax(model(batch_X),dim=1).detach().numpy())
        y_true.extend(batch_y[1].numpy())
    # print(y_true,'\n', y_pred)
    print('accuracy_score:', accuracy_score(np.array(y_pred), np.array(y_true)))


if __name__ == '__main__':
    main()
