
import tempfile
import numpy as np 
import pickle
import pandas as pd
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

    # Device definition
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    acc = []
    for bin_start in range(cfg.args.first_sub,2993, 1):
        #load preproccessed data
        dataset_loaded = load_concat_dataset(
            path=cfg.args.TUH_PP_PATH,
            preload=True,
            ids_to_load= range(bin_start,bin_start+1),
            target_name=('age', 'gender', 'pathological'),
        )

        # print(dataset_loaded.description)
        # We can finally generate compute windows. The resulting dataset is now ready
        # we will create compute windows. We specify a
        # mapping from genders 'M' and 'F' to integers, since this is required for
        # decoding.
        # print(len(dataset_loaded))

        tuh_windows = create_fixed_length_windows(
            dataset_loaded,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=750,
            window_stride_samples=750,
            drop_last_window=False,
            mapping={'M': 0, 'F': 1, False: 0, True: 1 },  # map non-digit targets
        )
        # store the number of windows required for loading later on
        tuh_windows.set_description({
            "n_windows": [len(d) for d in tuh_windows.datasets]})

        # print(len(tuh_windows), 'len:')
        ###############################################################################
        # # Iterating through the dataset gives x as ndarray(n_channels x 1000), y as
        # # [age, gender], and ind. Let's look at the last example again.
        # # print(tuh_windows.description)
        # x, y, ind = tuh_windows[-1]
        # print('x:', x.shape)
        # print('y:', y)
        # print('ind:', ind)

        # if cfg.args.split_mode == 'train':
        #     # split by train val
        #     tuh_splits = tuh_windows.split("train")
        # elif cfg.args.split_mode == 'gender':
        #     ## split by trainn and gender 
        #     split_ids = {
        #                 k: list(v)
        #                 for k, v in tuh_windows.description.groupby(["train","gender"]).groups.items()
        #             }
        #     splits = list(split_ids.keys())
        #     print('splits:', splits, [len(x) for x in split_ids.values()])
        #     tuh_splits = tuh_windows.split(split_ids)
        # elif cfg.args.split_mode == 'age':
        # ### split by train and age
        # # Binning of the data based on age
        #     df = tuh_windows.description
        #     df.loc[df.age < 40, 'age'] = 0
        #     df.loc[(df.age > 40) & (df.age < 60), 'age'] = 40
        #     df.loc[df.age > 60, 'age'] = 60
        #     # tuh_windows.description = df
        #     # print(tuh_windows.description,df)
        #     split_ids = {
        #                 k: list(v)
        #                 for k, v in df.groupby(["train","age"]).groups.items()
        #             }
        #     splits = list(split_ids.keys())
        #     print('splits:', splits, [len(x) for x in split_ids.values()])
        #     tuh_splits = tuh_windows.split(split_ids)


        ###############################################################################
        # We give the dataset to a pytorch DataLoader, such that it can be used for
        # model training.
        dl_train = DataLoader(
            # dataset=tuh_splits[str(splits[3+cfg.args.train_grp])],
            dataset=tuh_windows, #tuh_splits["True"],
            batch_size=cfg.args.batch_size,
            num_workers=cfg.args.num_workers,
        )
        # dl_eval = DataLoader(
        #     # dataset=tuh_splits[str(splits[cfg.args.val_grp])],
        #     dataset=tuh_splits["False"],
        #     batch_size=128,
        #     num_workers=cfg.args.num_workers,
        # )

        ###############################################################################
        # Iterating through the DataLoader gives batch_X as tensor(4 x n_channels x
        # 1000), batch_y as [tensor([4 x age of subject]), tensor([4 x gender of
        # subject])], and batch_ind. We will iterate to the end to look at the last example
        # again.
        for batch_X, batch_y, batch_ind in dl_train:
            break
        # print('batch_X:', batch_X.shape)
        # print('batch_y:', batch_y)
        # print('batch_ind:', batch_ind)

        ## training 
        if not 'model' in locals():
            model = Deep4Net(
                    in_chans = batch_X.shape[1],
                    n_classes = 2,
                    input_window_samples=batch_X.shape[2],
                    final_conv_length='auto',
                    # n_filters_time=32,
                    # n_filters_spat=32,
                    # filter_time_length=10,
                    # pool_time_length=3,
                    # pool_time_stride=3,
                    # n_filters_2=64,
                    # filter_length_2=10,
                    # n_filters_3=128,
                    # filter_length_3=10,
                    # n_filters_4=256,
                    # filter_length_4=10
                )
            print(model)
            #load model
            PATH = '~//projects/TUH/TUH/outputs/2022-01-04/15-02-20/d4.pth'
            model.load_state_dict(torch.load(PATH))

        # freez early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # # model.conv_classifier.weight.requires_grad = True 
        # model.bnorm.weight.requires_grad = True
        # # model.bnorm_2.weight.requires_grad = True
        # # model.bnorm_3.weight.requires_grad = True
        # # model.bnorm_4.weight.requires_grad = True
            model.to(device)

        # optimizer = optim.Adam(model.parameters(), lr=cfg.args.lr, weight_decay=cfg.args.weight_decay)

        # for ii in range(cfg.args.epochs): 
        #     losses = []
        #     for batch_X, batch_y, batch_ind in dl_train:
        #         batch_X = batch_X.to(device)
        #         batch_y = [y.to(device) for y in batch_y]
        #         pred = model(batch_X)
        #         loss = F.cross_entropy(pred, batch_y[2])
        #         losses.append(loss.cpu().detach().numpy())

        #         # Back propagate
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
            
            # print ('epoch:',ii, 'loss:', np.array(losses).mean(), '| train accuracy:', validatin(dl_train, model, device), '| val accuracy:', validatin(dl_eval, model, device)  )
        print ('train accuracy:', validatin(dl_train, model, device))#, '| val accuracy:', validatin(dl_eval, model, device))

        # # save model
        # torch.save(model.state_dict(), 'd4.pth',_use_new_zipfile_serialization=False)

        # #load model 
        # model.load_state_dict(torch.load('d4.pth'))
        acc.append([validatin(dl_train, model, device)])#, validatin(dl_eval, model, device)])

        if bin_start % 100 == 0:
            acc_ = np.array(acc)
            print(acc_, acc_.shape)
            np.save('acc.npy', acc_)
            # with open("test.pickle", "wb") as fp:   #Pickling
            #     pickle.dump(acc_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    acc_ = np.array(acc)
    print(acc_, acc_.shape)
    np.save('acc.npy', acc_)

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
    return accuracy_score(np.array(y_pred), np.array(y_true))


if __name__ == '__main__':
    # main()
    # exit()
    # load acc
    # PATH ="~//projects/TUH/TUH/outputs/2022-01-12/21-37-22/acc.npy"
    # with open(PATH, "rb") as fp:
    #     acc = np.load(fp)
    #     # acc = pickle.load(fp)
    # # acc = np.random.randn(2993,1)
    # print(acc.shape)

    # PATH ="~//projects/TUH/TUH/outputs/2022-01-13/14-34-53/acc.npy"
    # with open(PATH, "rb") as fp:
    #     acc2 = np.load(fp)
    #     # acc = pickle.load(fp)
    # # acc = np.random.randn(2993,1)
    # print(acc2.shape)
    # acc = np.concatenate([acc,acc2])
    # print(acc.shape)

    # ##add acc to df
    # #load preproccessed data
    # TUH_PP_PATH= '~//scratch/TUH/edf/preprocessed6'
    # dataset_loaded = load_concat_dataset(
    #     path=TUH_PP_PATH,
    #     preload=True,
    #     # ids_to_load= range(bin_start,bin_start+1),
    #     target_name=('age', 'gender', 'pathological'),
    # )

    # df = dataset_loaded.description

    # print(df)

    # df['acc'] = acc

    # df.to_pickle('df_.pkl')  # where to save it, usually as a .pkl

    df = pd.read_pickle('df_.pkl')

    print(df)
    print(df.groupby(["train", "gender","pathological"]).mean())




