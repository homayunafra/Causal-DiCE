import click
import sys
import numpy as np
import pandas as pd
import utils
import CF_generator as cfg
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import models as mdl
from sklearn.metrics import classification_report
import openpyxl
from os.path import dirname, join, abspath

@click.command()
@click.option("--dataset_id",   default='adult',    help="Which dataset to use.",
              type=click.Choice(["adult", "german", "sangiovese"]))
@click.option("--dataset_path", default="",         help="Dataset for experiments",                 type=click.Path(exists=False))
@click.option("--lr",           default=0.01,       help="Learning rate for the NN model",          type=float)
@click.option("--n_epoch",      default=20,         help="Number of training epochs",               type=int)
@click.option("--batch_size",   default=50,         help="Batch size for NN model training",        type=int)
@click.option("--seed",         default=17,         help="Random seed for running the experiment",   type=int)
@click.option("--test_size",    default=.2,         help="Ratio of test set size to the total data set size", type=float)
@click.option("--save_model",   default=False,      help="Whether or not save the trained model",   is_flag=True)

def main(*args, **kwargs):
    print("call: {}".format(" ".join(sys.argv)))
    run(*args, **kwargs)

def run(dataset_path, dataset_id, lr, n_epoch, batch_size, seed, test_size, save_model):
        
        dataset_path = join(dirname(abspath(__file__)), "Data") + dataset_path
        df = utils.load_dataset(dataset_path, dataset_id)
        n_features = df.shape[1] - 1
        data_params = {'dataframe': df, 'test_size': test_size, 'seed': seed}
        data = utils.proc_data(data_params)

        train_x = utils.trainData(torch.FloatTensor(data.train_df_one_hot_encoded.to_numpy()),
                                  torch.FloatTensor(data.train_df[data.outcome_name].to_numpy()))
        test_x = utils.testData(torch.FloatTensor(data.test_df_one_hot_encoded.to_numpy()))

        tr_loader = DataLoader(dataset=train_x, batch_size=batch_size, shuffle=True)
        tst_loader = DataLoader(dataset=test_x, batch_size=1)

        # train NN model #
        nn_parameters = {'input_size': data.train_df_one_hot_encoded.shape[1], 'hidden_cnt': 1,
                         'hidden_size': 20, 'output_size': 1}
        bb_mdl = mdl.NeuralNetwork(nn_parameters)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(bb_mdl.parameters(), lr=lr)

        for epoch in range(1, n_epoch + 1):
            train_loss, train_acc = 0, 0
            bb_mdl.train()
            for x, y in tr_loader:
                x = x.view(x.shape[0], -1)
                optimizer.zero_grad()

                # forward propagation #
                output = bb_mdl(x)

                # loss and accuracy computation #
                loss = criterion(output, y.unsqueeze(1))
                acc = utils.binary_acc(output, y.unsqueeze(1))

                # backward propagation #
                loss.backward()

                # weight optimization #
                optimizer.step()

                train_loss += loss.item()
                train_acc += acc.item()
            print(
                f'Epoch {epoch + 0:03}: | Loss: {train_loss / len(tr_loader):.5f} | Acc: {train_acc / len(tr_loader):.3f}')

        if save_model:
            torch.save(bb_mdl, "./Model/NN_model_{}.pth".format(dataset_id))

        # make prediction using the trained NN model #
        mdl_prediction = []
        with torch.no_grad():
            bb_mdl.eval()
            for x in tst_loader:
                output = bb_mdl(x)
                y_pred = torch.round(torch.sigmoid(output))
                mdl_prediction.append(y_pred.cpu().numpy())

        mdl_prediction = [y.squeeze().tolist() for y in mdl_prediction]
        print(classification_report(data.test_df.iloc[:, -1], mdl_prediction))
        
        undesirable_indices = data.test_df.index[[ind for ind, val in enumerate(mdl_prediction) if val == 0]]
        undesirable_data = data.test_df.loc[undesirable_indices, :]
        undesirable_data = undesirable_data[undesirable_data.iloc[:, -1] == 0]

        query_instance_all = undesirable_data.sample(n=10, replace=False, random_state=seed)
        query_instance_all.to_csv('all_queries'+'_dataset_'+dataset_id+'.csv', index=False)
        query_instance_all = query_instance_all.to_dict('records')

        # feature perturbation difficulty vector #
        fpd = np.ones(n_features)

        enc_fpd = []
        cat = 0
        for i in range(n_features):
            if i in data.categorical_feature_indexes:
                enc_fpd.extend([fpd[i]] * len(data.encoded_categorical_feature_indexes[cat]))
                cat += 1
            else:
                enc_fpd.append(fpd[i])

        metric_matrix = np.diag(enc_fpd)

        # model global feasibility constraints #
        glb_mat = np.zeros((len(data.encoded_feature_names) + 2, len(data.encoded_feature_names)), dtype=int)
        ''' 
        gbl_mat is a (p+2)xp matrix containing all global feasibility constraints. In this matrix:
           1. The number of rows indexes the number of features including two hypothetical features U^+ and U^-,
           2. The number of columns equals the number of features.
        '''
        usr_input = pd.read_excel("{}_global_feasibility.xlsx".format(dataset_id), header=None, engine='openpyxl')
        usr_input = usr_input.to_numpy()

        extnd_cols = []
        for orig_col in data.feature_names:
            if orig_col in data.categorical_feature_names:
                temp = [data.encoded_feature_names.index(
                    col) for col in data.encoded_feature_names if col.startswith(orig_col)]
            else:
                temp = [data.encoded_feature_names.index(orig_col)]
            extnd_cols.append(temp)

        for row in range(len(extnd_cols)+2):
            for col in range(n_features):
                if usr_input[row, col] == 1:
                    if row < len(extnd_cols):
                        for i in extnd_cols[row]:
                            for j in extnd_cols[col]:
                                glb_mat[i, j] = -1
                    elif row == len(extnd_cols):
                        for j in extnd_cols[col]:
                            glb_mat[len(data.encoded_feature_names), j] = -1
                    else:
                        for j in extnd_cols[col]:
                            glb_mat[len(data.encoded_feature_names)+1, j] = -1

        # generate counterfactuals #
        cfg.generate_cf(bb_mdl, data, query_instance_all, glb_mat, pert_diff_metric=metric_matrix, prx_weight=1.,
                        div_weight=1., cat_penalty=1., stp_threshold=0.5, lr=lr, min_iter=500, max_iter=10000,
                        loss_cnvg_maxiter=1)

        CFs_path = join(join(join(dirname(abspath(__file__)), "Results"), dataset_id), "CFs/")
        utils.proximity_plot(dataset_id, dataset_path, )


if __name__ == "__main__":
    main()

