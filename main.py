
import tempfile
import numpy as np 
import pandas
#
import mne
mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

from sklearn.metrics import accuracy_score, balanced_accuracy_score
#
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGResNet, EEGNetv4, TCN


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

    # Device definition
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:",device) 

    # define random seeds
    torch.manual_seed(2022)

    # define test subjects
    test_rec = torch.randint(0, 2993, (250,))
    train_rec_all = [sub for sub in range(0,2993) if sub not in test_rec]
    train_rec = [train_rec_all[idx] for idx in torch.randint(0, len(train_rec_all), (cfg.args.n_sub_train,))]
    
    #load preproccessed data
    dataset_train = load_concat_dataset(
        path=cfg.args.TUH_PP_PATH,
        preload=True,
        ids_to_load= train_rec, #range(cfg.args.first_sub,2993),
        target_name=('age', 'gender', 'pathological'),
    )

    dataset_test = load_concat_dataset(
        path=cfg.args.TUH_PP_PATH,
        preload=True,
        ids_to_load= list(test_rec.numpy()), #range(cfg.args.first_sub,2993),
        target_name=('age', 'gender', 'pathological'),
    )

    print(dataset_train.description)
    # We can finally generate compute windows. The resulting dataset is now ready
    # we will create compute windows. We specify a
    # mapping from genders 'M' and 'F' to integers, since this is required for
    # decoding.

    tuh_windows_train = create_fixed_length_windows(
        dataset_train,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=750,
        window_stride_samples=750,
        drop_last_window=False,
        mapping={'M': 0, 'F': 1, False: 0, True: 1 },  # map non-digit targets
    )
    # store the number of windows required for loading later on
    tuh_windows_train.set_description({
        "n_windows": [len(d) for d in tuh_windows_train.datasets]})

    tuh_windows_test = create_fixed_length_windows(
        dataset_test,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=750,
        window_stride_samples=750,
        drop_last_window=False,
        mapping={'M': 0, 'F': 1, False: 0, True: 1 },  # map non-digit targets
    )
    # store the number of windows required for loading later on
    tuh_windows_test.set_description({
        "n_windows": [len(d) for d in tuh_windows_test.datasets]})

    ###############################################################################
    # Iterating through the dataset gives x as ndarray(n_channels x 1000), y as
    # [age, gender], and ind. Let's look at the last example again.
    # print(tuh_windows.description)
    x, y, ind = tuh_windows_train[-1]
    print('x:', x.shape)
    print('y:', y)
    print('ind:', ind)

    if cfg.args.split_mode == 'train':
        # split by train val
        tuh_splits = tuh_windows_train.split("train")
    elif cfg.args.split_mode == 'gender':
        ## split by trainn and gender 
        split_ids = {
                    k: list(v)
                    for k, v in tuh_windows_train.description.groupby(["train","gender"]).groups.items()
                }
        splits = list(split_ids.keys())
        print('splits:', splits, [len(x) for x in split_ids.values()])
        tuh_splits = tuh_windows_train.split(split_ids)
    elif cfg.args.split_mode == 'age':
    ### split by train and age
    # Binning of the data based on age
        df = tuh_windows_train.description
        df.loc[df.age < 40, 'age'] = 0
        df.loc[(df.age > 40) & (df.age < 60), 'age'] = 40
        df.loc[df.age > 60, 'age'] = 60
        # tuh_windows.description = df
        # print(tuh_windows.description,df)
        split_ids = {
                    k: list(v)
                    for k, v in df.groupby(["train","age"]).groups.items()
                }
        splits = list(split_ids.keys())
        print('splits:', splits, [len(x) for x in split_ids.values()])
        tuh_splits = tuh_windows_train.split(split_ids)


    ###############################################################################
    # We give the dataset to a pytorch DataLoader, such that it can be used for
    # model training.
    dl_train = DataLoader(
        # dataset=tuh_splits[str(splits[3+cfg.args.train_grp])],
        dataset=tuh_splits["True"],
        batch_size=cfg.args.batch_size,
        num_workers=cfg.args.num_workers,
    )
    dl_eval = DataLoader(
        # dataset=tuh_splits[str(splits[cfg.args.val_grp])],
        dataset=tuh_splits["False"],
        batch_size=128,
        num_workers=cfg.args.num_workers,
    )

    dl_test = DataLoader(
        # dataset=tuh_splits[str(splits[cfg.args.val_grp])],
        dataset=tuh_windows_test,
        batch_size=128,
        num_workers=cfg.args.num_workers,
    )

    ###############################################################################
    # Iterating through the DataLoader gives batch_X as tensor(4 x n_channels x
    # 1000), batch_y as [tensor([4 x age of subject]), tensor([4 x gender of
    # subject])], and batch_ind. We will iterate to the end to look at the last example
    # again.
    for batch_X, batch_y, batch_ind in dl_train:
        break
    print('batch_X:', batch_X.shape)
    print('batch_y:', batch_y)
    print('batch_ind:', len(batch_ind))

    ## training 
    model = Deep4Net(
            in_chans = batch_X.shape[1],
            n_classes = 2,
            input_window_samples=batch_X.shape[2],
            final_conv_length='auto',
            # n_filters_time=64,
            # # n_filters_spat=32,
            # filter_time_length=8,
            # pool_time_length=4,
            # # pool_time_stride=3,
            # # n_filters_2=64,
            # # filter_length_2=10,
            # n_filters_3=256,
            # # filter_length_3=10,
            # # n_filters_4=256,
            # # filter_length_4=10
        )
    # model = TCN(
    #         n_in_chans=batch_X.shape[1] ,
    #         n_outputs=2,
    #         n_filters=55,
    #         n_blocks=5,
    #         kernel_size=8,
    #         drop_prob=0.05270154233150525,
    #         add_log_softmax=True
    #     )
    
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.args.lr, weight_decay=cfg.args.weight_decay)

    #
    train_acc = []
    val_acc = []
    test_acc = []

    for ii in range(cfg.args.epochs): 
        losses = []
        for batch_X, batch_y, batch_ind in dl_train:
            batch_X = batch_X.to(device)
            batch_y = [y.to(device) for y in batch_y]
            pred = model(batch_X)
            # pred = pred[:,:,0].squeeze()
            # print(pred.shape)
            loss = F.cross_entropy(pred, batch_y[2])
            losses.append(loss.cpu().detach().numpy())

            # Back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_acc.append(validatin(dl_train, model, device))
        val_acc.append(validatin(dl_eval, model, device))
        test_acc.append(validatin(dl_test, model, device))

        print ('epoch:',ii, 'loss:', np.array(losses).mean(), '| train accuracy:', train_acc[-1] , '| val accuracy:', val_acc[-1], '| test accuracy:', test_acc[-1])

    # save model
    torch.save(model.state_dict(), 'd4.pth',_use_new_zipfile_serialization=False)

    #load model 
    model.load_state_dict(torch.load('d4.pth'))

    #Save the size and accuracy
    idx = torch.argmax(torch.tensor(val_acc))
    df = pandas.DataFrame(data={'idx': [idx], 'Acc-train': [train_acc[idx]],'Acc-val': [val_acc[idx]] ,'Acc-test': [test_acc[idx]], 'Size': [cfg.args.n_sub_train]})
    print(df)
    df.to_csv('/home/mila/m/mohammad-javad.darvishi-bayasi/projects/TUH/TUH/accuracy_runs.csv', sep=',', mode='a', header=False)


def validatin(dl_eval, model, device):
    # validatin
    y_pred = []
    y_true = [] 
    for batch_X, batch_y, batch_ind in dl_eval:
        batch_X = batch_X.to(device)
        batch_y = [y.to(device) for y in batch_y]
        model.eval()
        y_pred.extend(torch.argmax(model(batch_X),dim=1).cpu().detach().numpy())
        y_true.extend(batch_y[2].cpu().numpy())
    # print(y_true,'\n', y_pred)
    # print('accuracy_score:', accuracy_score(np.array(y_pred), np.array(y_true)))
    return balanced_accuracy_score(np.array(y_pred), np.array(y_true))


if __name__ == '__main__':
    main()
