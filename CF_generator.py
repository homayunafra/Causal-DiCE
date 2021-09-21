import torch
import pandas as pd
import numpy as np
import timeit
import copy

def generate_cf(mdl, data, query_inst_all, glb_mat, pert_diff_metric, prx_weight, div_weight, cat_penalty,
                stp_threshold, lr=0.01, min_iter=500, max_iter=5000, loss_cnvg_maxiter=1):
    """ Generates counterfactuals by graident-descent
    :param mdl: The black-box model.
    :param data: the data class.
    :param query_inst_all: A dictionary of feature names and values. Test point of interest.
    :param total_CFs: Total number of counterfactuals required.
    :param glb_mat: the causal constraint matrix.
    :param pert_diff_metric: Perturbation difficulty metric A that encodes user constraints on feature perturbation.
    :param prx_weight: A positive float. Larger this weight, more close the counterfactuals are to the query_instance.
    :param div_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
    :param stp_threshold: Minimum threshold for counterfactuals target class probability.
    :param cat_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
    :param lr: Learning rate for optimizer.
    :param min_iter: Min iterations to run gradient descent for.
    :param max_iter: Max iterations to run gradient descent for.
    :param loss_cnvg_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence.
    """
    instance = 1
    for query_inst in query_inst_all:
        test_pred = list(query_inst.items())[-1][1]
        query_inst = prepare_query_inst(query_inst, data, encode=True)
        query_inst = query_inst.iloc[0].values
        query_inst_final = torch.FloatTensor(query_inst)

        desired_class = 1.0 - float(test_pred)
        target_cf_class = torch.tensor(desired_class)

        for total_CFs in [2, 4, 6, 8, 10]:

            for causal_constraint in [True, False]:

                cfs = initialize_CFs(data, query_inst, total_CFs)
                optimizer = torch.optim.Adam(cfs, lr=lr, betas=(0.0, 0.0))

                best_backup_cfs = []
                best_backup_cfs_preds = []
                iterations = 0
                loss_diff = 1.0
                prev_loss = 0.0
                loss_diff_thres = 1e-5
                loss_converge_iter = 0
                start_time = timeit.default_timer()
                elapsed = 0
                while stop_loop(mdl, data, cfs, target_cf_class, iterations, min_iter, max_iter, loss_diff,
                                loss_diff_thres, loss_converge_iter, loss_cnvg_maxiter, stp_threshold) is False:

                        # zero all existing gradients #
                        optimizer.zero_grad()
                        mdl.zero_grad()

                        # get loss and backpropogate #
                        loss = compute_loss(mdl, cfs, query_inst_final, pert_diff_metric, total_CFs, data.minx,
                                            data.encoded_categorical_feature_indexes, target_cf_class, prx_weight,
                                            div_weight, cat_penalty)
                        loss_value = loss

                        grads = torch.autograd.grad(loss, cfs, create_graph=True, retain_graph=True, only_inputs=True)

                        loss.backward()
                        if causal_constraint == True:
                            for ftr_ind in np.arange(len(data.encoded_feature_names)+1, -1, -1):
                                # backward loop since the last two features are +1 and -1 which affect other constraints
                                if ftr_ind < len(data.encoded_feature_names):
                                    check = [(np.multiply(glb_mat[ftr_ind], np.sign(grads[ind].detach().numpy()))
                                            - (np.multiply(glb_mat[ftr_ind],
                                    (np.sign(grads[ind][ftr_ind].detach().numpy()) * np.ones(len(grads[ind]))))))
                                            for ind in range(len(grads))]
                                    for ind in range(total_CFs):
                                        if all(check[ind] == 0) or \
                                                (any(check[ind] == 1) and np.sign(grads[ind].data[ftr_ind]) == 0) or \
                                                (any(check[ind] == -1) and np.sign(grads[ind].data[ftr_ind]) == 0):
                                            pass
                                        else:
                                            cfs[ind].grad[ftr_ind] = 0.0
                                else:
                                    check = [np.multiply(glb_mat[ftr_ind], np.sign(grads[grd_ind].detach().numpy()))
                                             for grd_ind in range(len(grads))]
                                    for ind in range(total_CFs):
                                        if ftr_ind == len(data.encoded_feature_names) and any(check[ind] < 0):
                                            # check constraints related to U^+ feature #
                                            indices = np.where(check[ind] < 0)
                                            cfs[ind].grad[indices] = 0.0
                                            grads[ind].data[indices] = 0.0
                                        elif ftr_ind == (len(data.encoded_feature_names) - 1) and any(check[ind] > 0):
                                            # constraints related to U^- feature #
                                            indices = np.where(check[ind] > 0)
                                            cfs[ind].grad[indices] = 0.0
                                            grads[ind].data[indices] = 0.0
                        end_time = timeit.default_timer()
                        elapsed += end_time - start_time
                        start_time = end_time
                        # update the variables #
                        optimizer.step()

                        # projection step #
                        for ix in range(total_CFs):
                            for jx in range(len(data.minx[0])):
                                cfs[ix].data[jx] = torch.clamp(cfs[ix][jx], min=0, max=1)

                        iterations += 1
                        loss_diff = abs(loss_value - prev_loss)
                        prev_loss = loss_value

                        # backing up CFs if they are valid
                        temp_cfs_stored = round_cfs(data, cfs)
                        mdl.eval()
                        test_preds_stored = [torch.sigmoid(mdl(cf.unsqueeze(0))) for cf in temp_cfs_stored]

                        if (bool(target_cf_class == 1 and any(i[0] >= stp_threshold for i in test_preds_stored))):
                            for ind in range(len(temp_cfs_stored)):
                                if bool(test_preds_stored[ind] >= stp_threshold):
                                    if not any([(temp_cfs_stored[ind] == cf_).all() for cf_ in best_backup_cfs]):
                                        best_backup_cfs.append(temp_cfs_stored[ind])
                                        best_backup_cfs_preds.append(test_preds_stored[ind])
                                        if len(best_backup_cfs) > total_CFs:
                                            sorted_dists = np.argsort([best_backup_cfs_preds[i] - stp_threshold
                                                                       for i in range(len(best_backup_cfs_preds))])
                                            best_backup_cfs_preds = [best_backup_cfs_preds[i] for i in sorted_dists[:total_CFs]]
                                            best_backup_cfs = [best_backup_cfs[i] for i in sorted_dists[:total_CFs]]

                assert len(best_backup_cfs) != 0, "The list of counterfactuals is empty!"
                m, s = divmod(elapsed, 60)
                f = open("Time.txt", "a")
                f.write("Query " + str(instance) + " k = " + str(total_CFs) + " causal constraint " +
                        str(causal_constraint) + " = " + str(m) + " minutes and " + str(s) + " seconds.\n")
                f.close()

                final_cfs = best_backup_cfs
                final_cfs_preds = [torch.round(best_backup_cfs_preds[ix]) for ix in range(len(best_backup_cfs_preds))]
                
                if len(final_cfs) == total_CFs:
                    total_CFs_found = total_CFs
                    print('Diverse Counterfactuals found! total time taken: %02d' % m, 'min %02d' % s, 'sec')
                else:
                    total_CFs_found = 0
                    for pred in final_cfs_preds:
                        if ((target_cf_class == 0 and pred < stp_threshold) or (
                                target_cf_class == 1 and pred > stp_threshold)):
                            total_CFs_found += 1

                    print('Only %d (required %d) Diverse Counterfactuals found for the given configuation, '
                          'perhaps try with different values of proximity (or diversity) weights or learning rate...'
                          % ( total_CFs_found, total_CFs), '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

                orig_inst, updated_cfs_list = convert_to_dataframe(query_inst, data, test_pred, final_cfs, final_cfs_preds)
                updated_cfs_df = pd.DataFrame(updated_cfs_list)
                updated_cfs_df.columns = orig_inst.columns
                updated_cfs_df.to_csv('cfs_total_' + str(total_CFs) + '_query_instance_' + str(instance) + '_causal'
                                      + str(causal_constraint) + '.csv', index=False)

                visualize_as_list(orig_inst, test_pred, updated_cfs_list)
        instance += 1

def prepare_query_inst(query_inst, data, encode):
    """Prepares user defined input for CF generator"""

    if isinstance(query_inst, list):
        query_inst = {'row1': query_inst}
        test = pd.DataFrame.from_dict(query_inst, orient='index', columns=data.feature_names)

    elif isinstance(query_inst, dict):
        query_inst = dict(zip(query_inst.keys(), [[q] for q in query_inst.values()]))
        test = pd.DataFrame(query_inst, columns=data.feature_names)

    test = test.reset_index(drop=True)

    if encode is False:
        return data.normalize_data(test)
    else:
        temp = data.prepare_df_for_encoding()

        temp = temp.append(test, ignore_index=True, sort=False)
        temp = data.one_hot_encoder(temp)
        temp = temp.tail(test.shape[0]).reset_index(drop=True)
        temp = data.normalize_data(temp)

        return temp.tail(test.shape[0]).reset_index(drop=True)

def initialize_CFs(data, query_inst, total_CFs):
    """Initialize counterfactuals"""
    cfs = []
    for ix in range(total_CFs):
        one_init = []
        for jx in range(len(data.encoded_feature_names)):
            one_init.append(query_inst[jx] + (ix * 0.01))
        cfs.append(torch.tensor(one_init, dtype=torch.float32))
        cfs[ix] = (cfs[ix] * (10 ** 4)).round() / (10 ** 4)
        cfs[ix].requires_grad = True
    return cfs

def stop_loop(mdl, data, cfs, target_cf_class, itr, min_iter, max_iter, loss_diff, loss_diff_thres, loss_converge_iter,
              loss_cnvg_maxiter, stp_threshold):
        """Determines the stopping condition for gradient descent."""

        # do GD for min iterations
        if itr < min_iter:
            return False

        # stop GD if max iter is reached
        if itr >= max_iter:
            return True

        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold)
        if loss_diff <= loss_diff_thres:
            loss_converge_iter += 1
            if loss_converge_iter < loss_cnvg_maxiter:
                return False
            else:
                temp_cfs = round_cfs(data, cfs)
                test_preds = [torch.sigmoid(mdl(cf.unsqueeze(0))) for cf in temp_cfs]

                if target_cf_class == 0 and all(i[0] <= stp_threshold for i in test_preds):
                    return True
                elif target_cf_class == 1 and all(i[0] >= stp_threshold for i in test_preds):
                    return True
                else:
                    return False
        else:
            return False

def compute_loss(mdl, cfs, query_inst_final, pert_diff_metric, total_CFs, minx, encoded_categorical_feature_indexes,
                 target_cf_class, prx_weight, div_weight, cat_penalty):
    """Computes the overall loss"""

    loss_part1 = compute_first_part_of_loss(mdl, total_CFs, cfs, target_cf_class)
    diagonal = np.array(np.diag(pert_diff_metric))
    feature_weights = torch.tensor(diagonal)
    loss_part2 = compute_second_part_of_loss(cfs, query_inst_final, feature_weights, total_CFs, minx)

    loss_part3 = compute_third_part_of_loss(cfs, total_CFs)

    loss_part4 = compute_fourth_part_of_loss(cfs, total_CFs, encoded_categorical_feature_indexes)

    loss = loss_part1 + (prx_weight * loss_part2) - (div_weight * loss_part3) + (cat_penalty * loss_part4)
    return loss

def compute_first_part_of_loss(mdl, total_CFs, cfs, target_cf_class):
    """Computes the first part (y-loss) of the loss function"""
    loss_part1 = 0.0
    mdl.eval()
    for i in range(total_CFs):
        temp_logits = torch.log10((abs(torch.sigmoid(mdl(cfs[i].unsqueeze(0))) - 0.000001))/(1 -
                                    abs(torch.sigmoid(mdl(cfs[i].unsqueeze(0))) - 0.000001)))
        criterion = torch.nn.ReLU()
        temp_loss = criterion(0.5 - (temp_logits*target_cf_class))

        loss_part1 += temp_loss[0]

    return loss_part1/total_CFs

def compute_dist(cf, x1, feature_weights):
    """Computes weighted distance between two vectors"""
    dist = torch.sum(torch.mul((torch.abs(cf - x1)), feature_weights))
    return dist

def compute_second_part_of_loss(cfs, query_inst_final, feature_weights, total_CFs, minx):
    """Compute the second part (distance from query instance) of the loss function."""
    loss_part2 = 0.0

    for i in range(total_CFs):
        loss_part2 += compute_dist(cfs[i], query_inst_final, feature_weights)
    return loss_part2/(torch.mul(len(minx[0]), total_CFs))

def compute_third_part_of_loss(cfs, total_CFs):
    """Computes the third part (diversity) of the loss function."""
    det_entries = torch.ones((total_CFs, total_CFs))
    for i in range(total_CFs):
        for j in range(total_CFs):
            det_entries[(i, j)] = 1.0 / (torch.exp(compute_dist(cfs[i], cfs[j], torch.eye(cfs[i].size()[0]))))
            if i == j:
                det_entries[(i, j)] += 0.0001

    loss_part3 = torch.det(det_entries)
    return loss_part3

def compute_fourth_part_of_loss(cfs, total_CFs, encoded_categorical_feature_indexes):
    """Add a linear equality constraint to the loss function to ensure all levels of a categorical variable sums to 1"""
    loss_part4 = torch.zeros(1)
    for i in range(total_CFs):
        for v in encoded_categorical_feature_indexes:
            if torch.sum(cfs[i][v[0]:v[-1] + 1]) > 1:
                loss_part4 += torch.pow((torch.sum(cfs[i][v[0]:v[-1] + 1]) - 1.0), 2)
    return torch.sqrt(loss_part4)

def round_cfs(data, cfs):
    """Round off the categorical features for displaying purposes"""
    temp_cfs = []
    for index, tcf in enumerate(cfs):
        cf = tcf.detach().clone().numpy()
        for i, v in enumerate(data.encoded_continuous_feature_indexes):
            org_cont = (cf[v] * (data.maxx[0][i] - data.minx[0][i])) + data.minx[0][i]
            if data.data_df[data.encoded_feature_names[v]].dtype == "int64":
                org_cont = round(org_cont)  # rounding off
            else:
                org_cont = round(org_cont, 4)  # rounding off

            normalized_cont = (org_cont - data.minx[0][i]) / (data.maxx[0][i] - data.minx[0][i])
            cf[v] = normalized_cont  # assign the projected continuous value

        for v in data.encoded_categorical_feature_indexes:
            maxs = np.argwhere(cf[v[0]:v[-1]+1] == np.amax(cf[v[0]:v[-1]+1])).flatten().tolist()
            ix = maxs[0]
            for vi in range(len(v)):
                if vi == ix:
                    cf[v[vi]] = 1.0
                else:
                    cf[v[vi]] = 0.0

        temp_cfs.append(torch.tensor(cf))

    return temp_cfs

def convert_to_dataframe(query_inst, data, test_pred, final_cfs, final_cfs_preds):
    query_inst_updated = pd.DataFrame(np.array([np.append(query_inst, test_pred)]),
                                          columns = data.encoded_feature_names+[data.outcome_name])

    org_inst = from_dummies(query_inst_updated, data.categorical_feature_names)
    org_inst = org_inst[data.feature_names + [data.outcome_name]]
    org_inst = de_normalize_data(org_inst, data.continuous_feature_names, data.continuous_feature_range)

    cfs = np.array([t.numpy() for t in final_cfs])

    result = get_decoded_data(cfs, data.encoded_feature_names, data.categorical_feature_names)
    result = de_normalize_data(result, data.continuous_feature_names, data.continuous_feature_range)
    for ix, feature in enumerate(data.continuous_feature_names):
        result[feature] = (result[feature].astype(float) * 10).round() / 10

    # predictions for CFs
    test_preds = [np.round(preds.flatten().tolist(), 3) for preds in final_cfs_preds]
    test_preds = [item for sublist in test_preds for item in sublist]
    test_preds = np.array(test_preds)

    result[data.outcome_name] = test_preds
    final_cfs_df = result[data.feature_names + [data.outcome_name]]
    final_cfs_list = final_cfs_df.values.tolist()

    return org_inst, final_cfs_list

def from_dummies(data, categorical_feature_names, prefix_sep='_'):
    """Gets the original data from dummy encoded data with k levels"""
    out = data.copy()
    for l in categorical_feature_names:
        cols, labs = [[c.replace(x, "") for c in data.columns if l+prefix_sep in c] for x in ["", l+prefix_sep]]
        out[l] = pd.Categorical(np.array(labs)[np.argmax(data[cols].values, axis=1)])
        out.drop(cols, axis=1, inplace=True)
    return out

def get_decoded_data(data, encoded_feature_names, categorical_feature_names):
    """Returns the original data from dummy encoded data"""
    if isinstance(data, np.ndarray):
        index = [i for i in range(0, len(data))]
        data = pd.DataFrame(data=data, index=index, columns=encoded_feature_names)
    return from_dummies(data, categorical_feature_names)

def de_normalize_data(df, continuous_feature_names, continuous_feature_range):
    """De-normalizes continuous features from [0,1] range to original range"""
    result = df.copy()
    for feature_name in continuous_feature_names:
        max_value = continuous_feature_range[feature_name][1]
        min_value = continuous_feature_range[feature_name][0]
        result[feature_name] = (df[feature_name]*(max_value - min_value)) + min_value
    return result

def visualize_as_list(query_inst, test_pred, final_cfs_list, show_only_changes=False):
    # original instance
    print('Query instance (original outcome : %i)' %round(float(test_pred)))
    print(query_inst.values.tolist()[0])
    # CFs
    print('\nDiverse Counterfactual set (new outcome : %i)' %(1-round(float(test_pred))))
    print_list(query_inst, final_cfs_list, show_only_changes)

def print_list(query_inst, li, show_only_changes):
    if show_only_changes is False:
        for ix in range(len(li)):
            print(li[ix])
    else:
        newli = copy.deepcopy(li)
        org = query_inst.values.tolist()[0]
        for ix in range(len(newli)):
            for jx in range(len(newli[ix])):
                if newli[ix][jx] == org[jx]:
                    newli[ix][jx] = '-'
            print(newli[ix])
