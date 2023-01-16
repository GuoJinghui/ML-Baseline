#Import libraries:
import pandas as pd
import argparse
import numpy as np
import xlwt

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=False,
                        default='Highlighted/drone_final.xls',
                        help='set path to data dir')
    parser.add_argument('--dependent_vars', nargs="+",
                        default=['app_allow', 'comfort_to_clean', 'travel_clean', 'train'],
                        help='dependent variables')
    parser.add_argument('--independent_vars', nargs="+",
                        default=['gender', 'Age', 'area_kind', 'time_area', 'home_kind', 'home_num', 'tech', 'int_net',
                                 'top_internet', 'social', 'social_freq', 'drone_freq', 'drone_negative_things',
                                 'concerns_drones', 'MosqDs_Kd', 'feel_risk', 'tk_measures', 'yard'],
                        help='independent variables')
    parser.add_argument('--inputs', nargs="+",
                        default=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
                        help='independent variables')
    parser.add_argument('--country', nargs="+",
                        default=['Malaysia', 'Mexico', 'Turkey'],
                        help='country level data')

    opt = parser.parse_args()
    print(opt)
    return opt


def readfile(data_path):
    xls = pd.ExcelFile(data_path)  # use r before absolute file path

    sheetX = xls.parse(0)  # 2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

    return sheetX


# prepare input data
def prepare_inputs(train, test, inputs):
    oe = OrdinalEncoder()
    oe.fit(inputs)
    train_encode = oe.transform(train)
    test_encode = oe.transform(test)
    return train_encode, test_encode


# prepare target
def prepare_targets(train, test, Y):
    le = LabelEncoder()
    le.fit(train)
    train_encode = le.transform(train)
    test_encode = le.transform(test)
    return train_encode, test_encode


def xgBoost_GridSearchCV(X_encode, Y_encode, n, file=None):
    n_estimators = [70, 50, 50, 90]
    min_child_weight = [9, 3, 1, 10]
    gamma = [0.0, 0.0, 1.08, 0.95]
    subsample = [0.1, 1.0, 1.0, 0.1]
    colsample_bytree = [0.1, 0.1, 0.1, 0.5]
    reg_lambda = [30, 40, 100, 20]
    reg_alpha = [0.01, 0.08, 0.01, 0.6]
    eta = [0.01, 0.08, 0.01, 0.6]
    other_params = {'use_label_encoder': False, 'eval_matric': 'mlogloss', 'eta': eta[n],
                    'n_estimators': n_estimators[n], 'gamma': gamma[n], 'max_depth': 1,
                    'min_child_weight': min_child_weight[n], 'colsample_bytree': colsample_bytree[n],
                    'colsample_bylevel': 1,
                    'subsample': subsample[n], 'reg_lambda': reg_lambda[n], 'reg_alpha': reg_alpha[n], 'seed': 33}
    cv_params = {'eta': np.logspace(-2, 0, 10)}
    model = xgb.XGBClassifier(**other_params)

    gs = GridSearchCV(model, cv_params, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(X_encode, Y_encode)
    print("Params Best: ", gs.best_params_)
    print("Model Best: ", gs.best_score_)
    file.writelines("n_estimators: {} \n".format(n_estimators[n]))
    file.writelines("max_depth: 1 \n")
    file.writelines("min_child_weight: {} \n".format(min_child_weight[n]))
    file.writelines("gamma: {} \n".format(gamma[n]))
    file.writelines("subsample: {} \n".format(subsample[n]))
    file.writelines("colsample_bytree: {} \n".format(colsample_bytree[n]))
    file.writelines("reg_lambda: {} \n".format(reg_lambda[n]))
    file.writelines("reg_alpha: {} \n".format(reg_alpha[n]))
    file.writelines("Params Best: {} \n".format(gs.best_params_))
    file.writelines("Model Best: {}\n\n".format(gs.best_score_))


def lightGBM_GridSearchCV(X_encode, Y_encode, n, file=None):
    max_depth = [4, 6, 4, 4]
    min_child_samples = [19, 20, 20, 19]
    bagging_fraction = [1, 1, 1, 0.8]
    bagging_freq = [2, 2, 2, 3]
    reg_alpha = [0.001, 0.001, 0.001, 0.01]
    reg_lambda = [8, 8, 8, 6]
    learning_rate = [0.03, 0.01, 0.01, 0.02]
    # min_child_weight = [9, 3, 1, 10]
    # gamma = [0.0, 0.0, 1.08, 0.95]
    # subsample = [0.1, 1.0, 1.0, 0.1]
    # colsample_bytree = [0.1, 0.1, 0.1, 0.5]
    # reg_lambda = [30, 40, 100, 20]
    # reg_alpha = [0.01, 0.08, 0.01, 0.6]
    # eta = [0.01, 0.08, 0.01, 0.6]
    parameters = {
        'learning_rate': np.logspace(-2, 0, 10)
    }
    model = lgb.LGBMClassifier(objective= 'multiclass',
                         is_unbalance = True,
                         metric = 'multi_logloss',
                         max_depth = max_depth[n],
                         num_leaves = 20,
                         learning_rate = learning_rate[n],
                         feature_fraction = 0.7,
                         min_child_samples = min_child_samples[n],
                         min_child_weight = 0.001,
                         bagging_fraction = bagging_fraction[n],
                         bagging_freq = bagging_freq[n],
                         reg_alpha = reg_alpha[n],
                         reg_lambda = reg_lambda[n],
                         cat_smooth = 0,
                         num_iterations = 200)

    gs = GridSearchCV(model, param_grid=parameters, refit=True, cv=5, n_jobs=-1)
    # alg.set_params(n_estimators=cvresult.shape[0])

    # gs = GridSearchCV(model, cv_params, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(X_encode, Y_encode)
    print("Params Best: ", gs.best_params_)
    print("Model Best: ", gs.best_score_)
    file.writelines("max_depth: {} \n".format(max_depth[n]))
    file.writelines("min_child_samples: {} \n".format(min_child_samples[n]))
    # file.writelines("feature_fraction: {} \n".format(feature_fraction[n]))
    # file.writelines("gamma: {} \n".format(gamma[n]))
    # file.writelines("subsample: {} \n".format(subsample[n]))
    # file.writelines("colsample_bytree: {} \n".format(colsample_bytree[n]))
    # file.writelines("reg_lambda: {} \n".format(reg_lambda[n]))
    # file.writelines("reg_alpha: {} \n".format(reg_alpha[n]))
    file.writelines("Params Best: {} \n".format(gs.best_params_))
    file.writelines("Model Best: {}\n\n".format(gs.best_score_))

def runGridSearchCV():
    args = parse_args()
    data_path = args.data_path
    dependent_vars = args.dependent_vars
    independent_vars = args.independent_vars

    """ independent variable inputs corresponding to the dependent variables """
    independent_vars = np.array(independent_vars)

    """ read data """
    sheetX = readfile(data_path)
    input_data = independent_vars[:17]
    sheetX['inputs'] = sheetX[[x for x in input_data]].values.tolist()
    inputs_list = np.array(sheetX['inputs'])
    inputs = []
    for i in range(len(inputs_list)):
        # print(inputs[i])
        inputs.append(np.array(inputs_list[i]))

    # """ write output into an excel """
    # wb = xlwt.Workbook()
    # wb_xgBoost = xlwt.Workbook()
    # wb_lightGBM = xlwt.Workbook()
    # sheet1 = wb.add_sheet('sheet1')
    # sheet1.write(0, 3, 'xgboost')
    # sheet1.write(0, 7, 'lightGBM')
    # sheet1.write(0, 11, 'Linear SVM')
    # sheet1.write(0, 15, 'Polynomial Kernel SVM')
    # for i in range(2, 16, 4):
    #     sheet1.write(1, i, 'Training Accuracy')
    #     sheet1.write(1, i + 2, 'Testing Accuracy')

    file1 = open("Tuning_Params/lightGBM/learning_rate.txt", "w")
    """ independent variables inputs """
    for n in range(len(dependent_vars)):
        var = dependent_vars[n]
        print(var + '\n')
        file1.writelines(var)
        """ generate inputs X and targets Y """
        Y = sheetX[var]

        """ encode inputs and targets """
        X_encode = prepare_inputs(inputs)
        Y_encode = prepare_targets(Y)

        # wb_xgBoost = xlwt.Workbook()
        # xgBoost_GridSearchCV(X_encode, Y_encode, n, file1)
        lightGBM_GridSearchCV(X_encode, Y_encode, n, file1)
        # wb_xgBoost.save('xgBoost.xls')
        # lightGBM_mean, lightGBM_std = lightGBM_cv(X_encode, Y_encode)
        # SVM_res = SVM_cv(X_encode, Y_encode)

    #     sheet1.write(n + 2, 0, var)
    #     sheet1.write(n + 2, 2, train_avg_xgBoost)
    #     sheet1.write(n + 2, 4, test_avg_xgBoost)
    #     # sheet1.write(n + 2, 6, lightGBM_mean)
    #     # sheet1.write(n + 2, 8, lightGBM_std)
    #     # for i in range(len(SVM_res)):
    #     #     sheet1.write(n + 2, 4 * i + 10, SVM_res[i][0])
    #     #     sheet1.write(n + 2, 4 * i + 12, SVM_res[i][1])
    #
    # wb_xgBoost.save('xgBoost.xls')
    # wb.save('result_all_train_test.xls')


def xgBoost_CV(n):
    n_estimators = [70, 50, 50, 90]
    min_child_weight = [9, 3, 1, 10]
    gamma = [0.0, 0.0, 1.08, 0.95]
    subsample = [0.1, 1.0, 1.0, 0.1]
    colsample_bytree = [0.1, 0.1, 0.1, 0.5]
    reg_lambda = [30, 40, 100, 20]
    reg_alpha = [0.01, 0.08, 0.01, 0.6]
    eta = [0.01, 0.08, 0.01, 0.6]
    other_params = {'use_label_encoder': False, 'eval_matric': 'mlogloss', 'eta': eta[n],
                    'n_estimators': n_estimators[n], 'gamma': gamma[n], 'max_depth': 1,
                    'min_child_weight': min_child_weight[n], 'colsample_bytree': colsample_bytree[n],
                    'colsample_bylevel': 1,
                    'subsample': subsample[n], 'reg_lambda': reg_lambda[n], 'reg_alpha': reg_alpha[n], 'seed': 33}
    model = xgb.XGBClassifier(**other_params)
    # model.fit(x_train, y_train)
    #
    # print('xgBoost Training accuracy {:.4f}'.format(model.score(x_train, y_train)))
    # print('xgBoost Testing accuracy {:.4f}'.format(model.score(x_test, y_test)))
    # return model.score(x_train, y_train), model.score(x_test, y_test)
    return model

def cv_train_test_withfile(X_encode, Y_encode, m, dvar, wb):
    # model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', num_class=6)
    n_splits = 5
    ss = ShuffleSplit(n_splits=n_splits, random_state=2333, test_size=0.25, train_size=None)
    # ss = StratifiedKFold(n_splits=n_splits, random_state=2333, shuffle=True)
    train_avg, test_avg, n = 0, 0, 1
    # wb = xlwt.Workbook()
    sheet1 = wb.add_sheet(str(dvar))
    sheet1.write(0, 2, 'Training Accuracy')
    sheet1.write(0, 4, 'Testing Accuracy')
    for train_index, test_index in ss.split(X_encode):
        x_train = X_encode[train_index]
        x_test = X_encode[test_index]
        y_train = Y_encode[train_index]
        y_test = Y_encode[test_index]
        """ fit xgBoost model no training data"""
        model = None
        if m == 'xgBoost':
            model = xgBoost_CV(dvar)
            model.fit(x_train, y_train)
        elif m == 'lightGBM':
            model = lgb.LGBMClassifier(learning_rate=0.09, random_state=50)
            model.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train, y_train)],
                      verbose=20, eval_metric='logloss')
        elif m == 'SVM_linear':
            model = SVC(kernel='linear')
            model.fit(x_train, y_train)
        elif m == 'SVM_poly':
            model = SVC(kernel='poly', degree=8)
            model.fit(x_train, y_train)
        else:
            print('Model Error')
            return

        """ make predictions for test data """
        # y_pred = model.predict(x_test)
        # predictions = [round(value) for value in y_pred]

        """ evaluate predictions """
        # accuracy = accuracy_score(y_test, predictions)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))
        # print('xgBoost Round {}: Training accuracy {:.4f}'.format(n, model.score(x_train, y_train)))
        # print('xgBoost Round {}: Testing accuracy {:.4f}'.format(n, model.score(x_test, y_test)))
        sheet1.write(n, 2, model.score(x_train, y_train))
        sheet1.write(n, 4, model.score(x_test, y_test))
        train_avg += model.score(x_train, y_train)
        test_avg += model.score(x_test, y_test)
        n += 1
    train_avg = train_avg/n_splits
    test_avg = test_avg/n_splits
    sheet1.write(n, 0, 'Average')
    sheet1.write(n, 2, train_avg)
    sheet1.write(n, 4, test_avg)
    return train_avg, test_avg

def lightGBM_CV(n):
    max_depth = [4, 6, 4, 4]
    min_child_samples = [19, 20, 20, 19]
    bagging_fraction = [1, 1, 1, 0.8]
    bagging_freq = [2, 2, 2, 3]
    reg_alpha = [0.001, 0.001, 0.001, 0.01]
    reg_lambda = [8, 8, 8, 6]
    learning_rate = [0.03, 0.01, 0.01, 0.02]

    model = lgb.LGBMClassifier(objective='multiclass',
                               is_unbalance=True,
                               metric='multi_logloss',
                               max_depth=max_depth[n],
                               num_leaves=20,
                               learning_rate=learning_rate[n],
                               feature_fraction=0.7,
                               min_child_samples=min_child_samples[n],
                               min_child_weight=0.001,
                               bagging_fraction=bagging_fraction[n],
                               bagging_freq=bagging_freq[n],
                               reg_alpha=reg_alpha[n],
                               reg_lambda=reg_lambda[n],
                               cat_smooth=0,
                               num_iterations=200)
    # model.fit(x_train, y_train)
    #
    # print('xgBoost Training accuracy {:.4f}'.format(model.score(x_train, y_train)))
    # print('xgBoost Testing accuracy {:.4f}'.format(model.score(x_test, y_test)))
    # return model.score(x_train, y_train), model.score(x_test, y_test)
    return model

def cv_train_test(x_train, x_test, y_train, y_test, m, dvar):
    model = None
    if m == 'xgBoost':
        model = xgBoost_CV(dvar)
        # model = xgb.XGBClassifier()
        model.fit(x_train, y_train)
    elif m == 'lightGBM':
        # model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
        model = lightGBM_CV(dvar)
        model.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train, y_train)],
                  verbose=20, eval_metric='logloss')
    elif m == 'SVM_linear':
        model = SVC(kernel='linear')
        model.fit(x_train, y_train)
    elif m == 'SVM_poly':
        model = SVC(kernel='poly', degree=8)
        model.fit(x_train, y_train)
    else:
        print('Model Error')
        return

    """ make predictions for test data """
    # y_pred = model.predict(x_test)
    # predictions = [round(value) for value in y_pred]
    return model.score(x_train, y_train), model.score(x_test, y_test)


def create_sheet(wb, sheet_name):
    sheet1 = wb.add_sheet(sheet_name)
    sheet1.write(0, 3, 'xgboost')
    sheet1.write(0, 7, 'lightGBM')
    sheet1.write(0, 11, 'Linear SVM')
    sheet1.write(0, 15, 'Polynomial Kernel SVM')

    for i in range(2, 16, 4):
        sheet1.write(1, i, 'Train Accuracy')
        sheet1.write(1, i + 2, 'Test Accuracy')
        # sheet1.write(1, i, 'Training')
        # sheet1.write(1, i + 1, 'std')
        # sheet1.write(1, i + 2, 'Testing')
        # sheet1.write(1, i + 3, 'std')
    return sheet1


def run_train_test(sheet, inputs, Y, X_train, X_test, Y_train, Y_test, n, var):
    """ encode inputs and targets """
    X_train_encode, X_test_encode = prepare_inputs(X_train, X_test, inputs)
    Y_train_encode, Y_test_encode = prepare_targets(Y_train, Y_test, Y)

    # wb_xgBoost = xlwt.Workbook()
    train_xgBoost, test_xgBoost = cv_train_test(X_train_encode, X_test_encode,
                                                Y_train_encode, Y_test_encode, 'xgBoost', n)
    train_lightGBM, test_lightGBM = cv_train_test(X_train_encode, X_test_encode,
                                                Y_train_encode, Y_test_encode, 'lightGBM', n)
    train_SVM_linear, test_SVM_linear = cv_train_test(X_train_encode, X_test_encode,
                                                Y_train_encode, Y_test_encode, 'SVM_linear', n)
    train_SVM_poly, test_SVM_poly = cv_train_test(X_train_encode, X_test_encode,
                                                Y_train_encode, Y_test_encode, 'SVM_poly', n)

    sheet.write(n + 2, 0, var)
    sheet.write(n + 2, 2, train_xgBoost)
    # sheet.write(n + 2, 3, std_train_xgBoost)
    sheet.write(n + 2, 4, test_xgBoost)
    # sheet.write(n + 2, 5, std_test_xgBoost)

    sheet.write(n + 2, 6, train_lightGBM)
    # sheet.write(n + 2, 7, std_train_lightGBM)
    sheet.write(n + 2, 8, test_lightGBM)
    # sheet.write(n + 2, 9, std_test_lightGBM)

    sheet.write(n + 2, 10, train_SVM_linear)
    # sheet.write(n + 2, 11, std_train_SVM_linear)
    sheet.write(n + 2, 12, test_SVM_linear)
    # sheet.write(n + 2, 13, std_test_SVM_linear)

    sheet.write(n + 2, 14, train_SVM_poly)
    # sheet.write(n + 2, 15, std_train_SVM_poly)
    sheet.write(n + 2, 16, test_SVM_poly)
    # sheet.write(n + 2, 17, std_test_SVM_poly)

    return sheet


def run_all_train_test():
    args = parse_args()
    data_path = args.data_path
    dependent_vars = args.dependent_vars
    independent_vars = args.independent_vars

    """ independent variable inputs corresponding to the dependent variables """
    independent_vars = np.array(independent_vars)

    """ read data """
    sheetX = readfile(data_path)
    input_data = independent_vars[:17]
    sheetX['inputs'] = sheetX[[x for x in input_data]].values.tolist()
    inputs_list = np.array(sheetX['inputs'])
    inputs = []
    for i in range(len(inputs_list)):
        # print(inputs[i])
        inputs.append(np.array(inputs_list[i]))

    # file = open('all_tune_lightGBM.txt', 'w')
    """ write output into an excel """
    wb = xlwt.Workbook()

    sheet1 = wb.add_sheet('sheet1')
    sheet1.write(0, 3, 'xgboost')
    sheet1.write(0, 7, 'lightGBM')
    sheet1.write(0, 11, 'Linear SVM')
    sheet1.write(0, 15, 'Polynomial Kernel SVM')
    for i in range(2, 16, 4):
        sheet1.write(1, i, 'Training')
        sheet1.write(1, i + 1, 'std')
        sheet1.write(1, i + 2, 'Testing')
        sheet1.write(1, i + 3, 'std')

    """ independent variables inputs """
    for n in range(len(dependent_vars)):
        var = dependent_vars[n]
        print(var)
        """ generate inputs X and targets Y """
        Y = sheetX[var]

        """ encode inputs and targets """
        X_encode = prepare_inputs(inputs)
        Y_encode = prepare_targets(Y)

        train_avg_xgBoost, test_avg_xgBoost, std_train_xgBoost, std_test_xgBoost = \
            cv_train_test(X_encode, Y_encode, 'xgBoost', n)
        train_avg_lightGBM, test_avg_lightGBM, std_train_lightGBM, std_test_lightGBM = \
            cv_train_test(X_encode, Y_encode, 'lightGBM', n)
        train_avg_SVM_linear, test_avg_SVM_linear, std_train_SVM_linear, std_test_SVM_linear = \
            cv_train_test(X_encode, Y_encode, 'SVM_linear', n)
        train_avg_SVM_poly, test_avg_SVM_poly, std_train_SVM_poly, std_test_SVM_poly = \
            cv_train_test(X_encode, Y_encode, 'SVM_poly', n)

        sheet1.write(n + 2, 0, var)
        sheet1.write(n + 2, 2, train_avg_xgBoost)
        sheet1.write(n + 2, 3, std_train_xgBoost)
        sheet1.write(n + 2, 4, test_avg_xgBoost)
        sheet1.write(n + 2, 5, std_test_xgBoost)

        sheet1.write(n + 2, 6, train_avg_lightGBM)
        sheet1.write(n + 2, 7, std_train_lightGBM)
        sheet1.write(n + 2, 8, test_avg_lightGBM)
        sheet1.write(n + 2, 9, std_test_lightGBM)

        sheet1.write(n + 2, 10, train_avg_SVM_linear)
        sheet1.write(n + 2, 11, std_train_SVM_linear)
        sheet1.write(n + 2, 12, test_avg_SVM_linear)
        sheet1.write(n + 2, 13, std_test_SVM_linear)

        sheet1.write(n + 2, 14, train_avg_SVM_poly)
        sheet1.write(n + 2, 15, std_train_SVM_poly)
        sheet1.write(n + 2, 16, test_avg_SVM_poly)
        sheet1.write(n + 2, 17, std_test_SVM_poly)

    wb.save('all_train_test_tune.xls')


def cross_country_train_test(dependent_vars, sheetX, inputs, slice_index, sheet, category, idx=0, to_idx=0):
    X_slice = inputs[slice_index[idx]:slice_index[idx + 1]]
    X_remain = inputs[0:slice_index[idx]] + inputs[slice_index[idx + 1]:]
    for n in range(len(dependent_vars)):
        var = dependent_vars[n]
        # file.writelines('\t{}:\n'.format(var))
        print(var)
        """ generate targets Y """
        Y = sheetX[var]
        if category == 'country2country':
            X_test = inputs[slice_index[to_idx]:slice_index[to_idx + 1]]
            Y_train = Y[slice_index[idx]:slice_index[idx + 1]]
            Y_test = Y[slice_index[to_idx]:slice_index[to_idx + 1]]
            sheet = run_train_test(sheet, inputs, Y, X_slice, X_test, Y_train, Y_test, n, var)
        elif category == 'country2remain':
            Y_train = Y[slice_index[idx]:slice_index[idx + 1]]
            Y_test = Y[0:slice_index[idx]] + Y[slice_index[idx + 1]:]
            sheet = run_train_test(sheet, inputs, Y, X_slice, X_remain, Y_train, Y_test, n, var)
        elif category == 'country2ALL':
            Y_train = Y[slice_index[idx]:slice_index[idx + 1]]
            sheet = run_train_test(sheet, inputs, Y, X_slice, inputs, Y_train, Y, n, var)
        elif category == 'remain2country':
            Y_train = Y[0:slice_index[idx]] + Y[slice_index[idx + 1]:]
            Y_test = Y[slice_index[idx]:slice_index[idx + 1]]
            sheet = run_train_test(sheet, inputs, Y, X_remain, X_slice, Y_train, Y_test, n, var)
        elif category == 'ALL2country':
            Y_test = Y[slice_index[idx]:slice_index[idx + 1]]
            sheet = run_train_test(sheet, inputs, Y, inputs, X_slice, Y, Y_test, n, var)
        else:
            print("Category Error")
            return
    return sheet

def main():
    args = parse_args()
    data_path = args.data_path
    dependent_vars = args.dependent_vars
    independent_vars = args.independent_vars

    """ independent variable inputs corresponding to the dependent variables """
    independent_vars = np.array(independent_vars)

    """ read data """
    sheetX = readfile(data_path)
    input_data = independent_vars[:17]
    sheetX['inputs'] = sheetX[[x for x in input_data]].values.tolist()
    inputs_list = np.array(sheetX['inputs'])
    inputs = []
    for i in range(len(inputs_list)):
        inputs.append(np.array(inputs_list[i]))

    """ country level """
    country = np.array(sheetX['country'])
    slice_index = [0]
    country_list = [country[0]]
    i, j = 0, 0
    while i < len(country):
        if country[i] == country_list[j]:
            i += 1
        else:
            country_list.append(country[i])
            slice_index.append(i)
            i += 1
            j += 1
    slice_index.append(len(country))
    print(slice_index)
    print(country_list)

    wb = xlwt.Workbook()
    for idx in range(len(country_list)):
        country_remain = ''
        """ 
        a) one country to another one country 
            category: 'country2country'
            Train: country_list[idx]
            Test: country_list[to_idx]
        """
        for to_idx in range(len(country_list)):
            if idx != to_idx:
                country_remain += country_list[to_idx] + ', '
                """ write output into an excel """
                sheet_name = country_list[idx] + '_to_' + country_list[to_idx]
                sheet1 = create_sheet(wb, sheet_name)
                sheet1 = cross_country_train_test(dependent_vars, sheetX, inputs, slice_index,
                                                  sheet1, "country2country", idx=idx, to_idx=to_idx)

        """ 
        b) one country to the remaining countries
            category: 'country2remain'
            Train: country_list[idx]
            Test: remaining countries
        """
        sheet_name = country_list[idx] + '_to_' + country_remain[:-2]
        sheet2 = create_sheet(wb, sheet_name)
        sheet2 = cross_country_train_test(dependent_vars, sheetX, inputs, slice_index,
                                                  sheet2, "country2remain", idx=idx)

        """ 
        c) one country to all countries
            category: 'country2ALL'
            Train: country_list[idx]
            Test: ALL countries
        """
        sheet_name = country_list[idx] + '_to_ALL'
        sheet3 = create_sheet(wb, sheet_name)
        sheet3 = cross_country_train_test(dependent_vars, sheetX, inputs, slice_index,
                                                  sheet3, "country2ALL", idx=idx)

        """ 
        d) remaining countries to one country
            category: 'remain2country'
            Train: remaining countries
            Test: country_list[idx]
        """
        sheet_name = country_remain[:-2] + '_to_' + country_list[idx]
        sheet4 = create_sheet(wb, sheet_name)
        sheet4 = cross_country_train_test(dependent_vars, sheetX, inputs, slice_index,
                                                  sheet4, "remain2country", idx=idx)

        """ 
        f) all countries to one country
            category: 'ALL2country'
            Train: ALL countries
            Test: country_list[idx]
        """
        sheet_name = 'ALL_to_' + country_list[idx]
        sheet5 = create_sheet(wb, sheet_name)
        sheet5 = cross_country_train_test(dependent_vars, sheetX, inputs, slice_index,
                                                  sheet5, "ALL2country", idx=idx)

    wb.save('cross_country.xls')


if __name__ == '__main__':
    # runGridSearchCV()
    # run_all_train_test()
    main()