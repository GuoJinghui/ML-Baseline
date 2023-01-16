import argparse
import pandas as pd
import numpy as np

import xlwt

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

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


def inputs(target, independent_vars):
    if target == 'app_allow':
        return independent_vars[:-1]
    if target == 'comfort_to_clean':
        return independent_vars[:14]
    if target == 'travel_clean':
        return independent_vars[:14]
    if target == 'train':
        return independent_vars[:14]
    if target == 'experience_share':
        return independent_vars[:14]
    if target == 'MosqCt_BhP':
        return independent_vars[:6]
    if target == 'cont_hoft':
        return independent_vars[:6]
    if target == 'yard':
        return independent_vars[:6]
    if target == 'cditches_howoften':
        return independent_vars[:6] + independent_vars[14:]
    if target == 'intent_mcont':
        return independent_vars[:6] + independent_vars[14:17]
    if target == 'check_breeding':
        return independent_vars[:6] + independent_vars[14:17]
    return []

def readfile(data_path):
    xls = pd.ExcelFile(data_path)  # use r before absolute file path
    sheetX = xls.parse(0)  # 2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis
    return sheetX


# prepare input data
def prepare_inputs(X):
    oe = OrdinalEncoder()
    oe.fit(X)
    X_encode = oe.transform(X)
    # X_test_enc = oe.transform(X_test)
    return X_encode


# prepare target
def prepare_targets(Y):
    le = LabelEncoder()
    le.fit(Y)
    Y_encode = le.transform(Y)
    # y_test_enc = le.transform(y_test)
    return Y_encode


def xgBoost(X, Y):
    """ split training and testing data """
    seed = 7
    test_size = 0.33
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    """ fit xgBoost model no training data"""
    model = XGBClassifier()
    model.fit(x_train, y_train)

    """ make predictions for test data """
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]

    """ evaluate predictions """
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('xgBoost Training accuracy {:.4f}'.format(model.score(x_train, y_train)))
    print('xgBoost Testing accuracy {:.4f}'.format(model.score(x_test, y_test)))
    return model.score(x_train, y_train), model.score(x_test, y_test)

def lightGBM(X, Y):
    """ split training and testing data """
    # seed = 7
    # random_state = 42
    test_size = 0.33
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train, y_train)],
              verbose=20, eval_metric='logloss')
    print('lightGBM Training accuracy {:.4f}'.format(model.score(x_train, y_train)))
    print('lightGBM Testing accuracy {:.4f}'.format(model.score(x_test, y_test)))
    return model.score(x_train, y_train), model.score(x_test, y_test)


def SVM(X, Y):
    acc = []
    """ split training and testing data """
    # seed = 7
    # random_state = 42
    test_size = 0.33
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    """ Linear SVM """
    linear_model = SVC(kernel='linear')
    linear_model.fit(x_train, y_train)
    linear_train_acc = linear_model.score(x_train, y_train)
    linear_test_acc = linear_model.score(x_test, y_test)
    print('Linear SVM Training accuracy {:.4f}'.format(linear_train_acc))
    print('Linear SVM Testing accuracy {:.4f}'.format(linear_test_acc))
    acc.append([linear_train_acc, linear_test_acc])

    """ Polynomial Kernel SVM """
    poly_model = SVC(kernel='poly', degree=8)
    poly_model.fit(x_train, y_train)
    poly_train_acc = poly_model.score(x_train, y_train)
    poly_test_acc = poly_model.score(x_test, y_test)
    print('Polynomial Kernel SVM Training accuracy {:.4f}'.format(poly_train_acc))
    print('Polynomial Kernel SVM Testing accuracy {:.4f}'.format(poly_test_acc))
    acc.append([poly_train_acc, poly_test_acc])

    # """ Gaussian Kernel SVM """
    # gaussian_model = SVC(kernel='rbf')
    # gaussian_model.fit(x_train, y_train)
    # gaussian_train_acc = gaussian_model.score(x_train, y_train)
    # gaussian_test_acc = gaussian_model.score(x_test, y_test)
    # print('Gaussian Kernel SVM Training accuracy {:.4f}'.format(gaussian_train_acc))
    # print('Gaussian Kernel SVM Testing accuracy {:.4f}'.format(gaussian_test_acc))
    # acc.append([gaussian_train_acc, gaussian_test_acc])
    #
    # """ Sigmoid Kernel SVM """
    # sigmoid_model = SVC(kernel='sigmoid')
    # sigmoid_model.fit(x_train, y_train)
    # sigmoid_train_acc = sigmoid_model.score(x_train, y_train)
    # sigmoid_test_acc = sigmoid_model.score(x_test, y_test)
    # print('Sigmoid Kernel SVM Training accuracy {:.4f}'.format(sigmoid_train_acc))
    # print('Sigmoid Kernel SVM Testing accuracy {:.4f}'.format(sigmoid_test_acc))
    # acc.append([sigmoid_train_acc, sigmoid_test_acc])

    return acc


def KNN(X, Y):
    """ split training and testing data """
    # seed = 7
    # random_state = 42
    test_size = 0.33
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    print('KNN Training accuracy {:.4f}'.format(model.score(x_train, y_train)))
    print('KNN Testing accuracy {:.4f}'.format(model.score(x_test, y_test)))
    return model.score(x_train, y_train), model.score(x_test, y_test)


def run_lightGBM(X_encode, Y0_encode, Y1_encode, Y2_encode, Y3_encode):
    print("lightGBM Y0")
    lightGBM(X_encode, Y0_encode)
    print("lightGBM Y1")
    lightGBM(X_encode, Y1_encode)
    print("lightGBM Y2")
    # lightGBM(X_encode, Y2_encode)
    print("lightGBM Y3")
    lightGBM(X_encode, Y3_encode)


def xgBoost_cv(X_encode, Y_encode):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', num_class=6)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
    results = cross_val_score(model, X_encode, Y_encode, cv=kfold, error_score='raise')
    print("xgBoost_cv Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    return results.mean(), results.std()


def xgBoost_GridSearchCV(X_encode, Y_encode):
    other_params = {'use_label_encoder': False, 'eval_matric': 'mlogloss', 'eta': 0.3, 'n_estimators': 500, 'gamma': 0,
                    'max_depth': 6, 'min_child_weight': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1,
                    'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0, 'seed': 33}
    cv_params = {'n_estimators': np.linspace(100, 1000, 10, dtype=int)}
    model = xgb.XGBClassifier(**other_params)

    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=1000, nfold=cv_folds,
                      metrics='mlogloss', early_stopping_rounds=50)
    alg.set_params(n_estimators=cvresult.shape[0])

    gs = GridSearchCV(model, cv_params, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(X_encode, Y_encode)
    print("Params Best: ", gs.best_params_)
    print("Model Best: ", gs.best_score_)


def runGridSearchCV():
    args = parse_args()
    data_path = args.data_path
    dependent_vars = args.dependent_vars
    independent_vars = args.independent_vars
    # print(independent_vars)

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

    """ independent variables inputs """
    for n in range(len(dependent_vars)):
        var = dependent_vars[n]
        print(var)
        """ generate inputs X and targets Y """
        Y = sheetX[var]

        """ encode inputs and targets """
        X_encode = prepare_inputs(inputs)
        Y_encode = prepare_targets(Y)

        # wb_xgBoost = xlwt.Workbook()
        xgBoost_GridSearchCV(X_encode, Y_encode)
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

def cv_train_test(X_encode, Y_encode, m, wb, dvar):
    # model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', num_class=6)
    # ss = ShuffleSplit(n_splits=10, random_state=0, test_size=0.25, train_size=None)
    n_splits = 10
    ss = StratifiedKFold(n_splits=n_splits, random_state=0)
    train_avg, test_avg, n = 0, 0, 1
    # wb = xlwt.Workbook()
    sheet1 = wb.add_sheet(dvar)
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
            model = XGBClassifier()
            model.fit(x_train, y_train)
        elif m == 'lightGBM':
            model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
            model.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train, y_train)],
                      verbose=20, eval_metric='logloss')
        elif m == 'SVM_linear':
            model = SVC(kernel='linear')
            model.fit(x_train, y_train)
        elif m == 'SVM_poly':
            poly_model = SVC(kernel='poly', degree=8)
            poly_model.fit(x_train, y_train)
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
    # return train_avg, test_avg


def run_all_train_test():
    args = parse_args()
    data_path = args.data_path
    dependent_vars = args.dependent_vars
    independent_vars = args.independent_vars
    # print(independent_vars)

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

    """ write output into an excel """
    wb = xlwt.Workbook()
    wb_xgBoost = xlwt.Workbook()
    wb_lightGBM = xlwt.Workbook()
    sheet1 = wb.add_sheet('sheet1')
    sheet1.write(0, 3, 'xgboost')
    sheet1.write(0, 7, 'lightGBM')
    sheet1.write(0, 11, 'Linear SVM')
    sheet1.write(0, 15, 'Polynomial Kernel SVM')
    for i in range(2, 16, 4):
        sheet1.write(1, i, 'Training Accuracy')
        sheet1.write(1, i + 2, 'Testing Accuracy')

    """ independent variables inputs """
    for n in range(len(dependent_vars)):
        var = dependent_vars[n]
        print(var)
        """ generate inputs X and targets Y """
        Y = sheetX[var]

        """ encode inputs and targets """
        X_encode = prepare_inputs(inputs)
        Y_encode = prepare_targets(Y)

        # wb_xgBoost = xlwt.Workbook()
        train_avg_xgBoost, test_avg_xgBoost = cv_train_test(X_encode, Y_encode, wb_xgBoost, var)
        # wb_xgBoost.save('xgBoost.xls')
        # lightGBM_mean, lightGBM_std = lightGBM_cv(X_encode, Y_encode)
        # SVM_res = SVM_cv(X_encode, Y_encode)

        sheet1.write(n + 2, 0, var)
        sheet1.write(n + 2, 2, train_avg_xgBoost)
        sheet1.write(n + 2, 4, test_avg_xgBoost)
        # sheet1.write(n + 2, 6, lightGBM_mean)
        # sheet1.write(n + 2, 8, lightGBM_std)
        # for i in range(len(SVM_res)):
        #     sheet1.write(n + 2, 4 * i + 10, SVM_res[i][0])
        #     sheet1.write(n + 2, 4 * i + 12, SVM_res[i][1])

    wb_xgBoost.save('xgBoost.xls')
    wb.save('result_all_train_test.xls')

def lightGBM_cv(X_encode, Y_encode):
    model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
    results = cross_val_score(model, X_encode, Y_encode, cv=kfold)
    print("lightGBM_cv Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    return results.mean(), results.std()


def SVM_cv(X_encode, Y_encode):
    res = []
    """ Linear SVM """
    linear_model = SVC(kernel='linear')
    linear_kfold= StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
    linear_results = cross_val_score(linear_model, X_encode, Y_encode, cv=linear_kfold)
    print("SVM_linear_cv Accuracy: %.2f%% (%.2f%%)" % (linear_results.mean() * 100, linear_results.std() * 100))
    res.append([linear_results.mean(), linear_results.std()])

    """ Polynomial Kernel SVM """
    poly_model = SVC(kernel='poly', degree=8)
    poly_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
    poly_results = cross_val_score(poly_model, X_encode, Y_encode, cv=poly_kfold)
    print("SVM_poly_cv Accuracy: %.2f%% (%.2f%%)" % (poly_results.mean() * 100, poly_results.std() * 100))
    res.append([poly_results.mean(), poly_results.std()])
    return res


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
        """ write output into an excel """
        sheet1 = wb.add_sheet(country_list[idx])
        sheet1.write(0, 3, 'xgboost')
        sheet1.write(0, 7, 'lightGBM')
        sheet1.write(0, 11, 'Linear SVM')
        sheet1.write(0, 15, 'Polynomial Kernel SVM')
        for i in range(2, 16, 4):
            sheet1.write(1, i, 'Mean')
            sheet1.write(1, i + 2, 'std')

        print(country_list[idx])
        """ country level slicing """
        X_slice = inputs[slice_index[idx]:slice_index[idx + 1]]

        """ independent variables inputs """
        for n in range(len(dependent_vars)):
            var = dependent_vars[n]
            print(var)
            """ generate inputs X and targets Y """
            Y = sheetX[var]
            Y_slice = Y[slice_index[idx]:slice_index[idx + 1]]

            """ encode inputs and targets """
            X_encode = prepare_inputs(X_slice)
            Y_encode = prepare_targets(Y_slice)

            xgb_mean, xgb_std = xgBoost_cv(X_encode, Y_encode)
            lightGBM_mean, lightGBM_std = lightGBM_cv(X_encode, Y_encode)
            SVM_res = SVM_cv(X_encode, Y_encode)

            sheet1.write(n + 2, 0, var)
            sheet1.write(n + 2, 2, xgb_mean)
            sheet1.write(n + 2, 4, xgb_std)
            sheet1.write(n + 2, 6, lightGBM_mean)
            sheet1.write(n + 2, 8, lightGBM_std)
            for i in range(len(SVM_res)):
                sheet1.write(n + 2, 4 * i + 10, SVM_res[i][0])
                sheet1.write(n + 2, 4 * i + 12, SVM_res[i][1])

    wb.save('result_country_v1.xls')


def run_all():
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

    """ write output into an excel """
    wb = xlwt.Workbook()
    sheet1 = wb.add_sheet('sheet1')
    sheet1.write(0, 3, 'xgboost')
    sheet1.write(0, 7, 'lightGBM')
    sheet1.write(0, 11, 'Linear SVM')
    sheet1.write(0, 15, 'Polynomial Kernel SVM')
    for i in range(2, 16, 4):
        sheet1.write(1, i, 'Mean')
        sheet1.write(1, i + 2, 'std')

    """ independent variables inputs """
    for n in range(len(dependent_vars)):
        var = dependent_vars[n]
        print(var)
        """ generate inputs X and targets Y """
        Y = sheetX[var]

        """ encode inputs and targets """
        X_encode = prepare_inputs(inputs)
        Y_encode = prepare_targets(Y)

        xgb_mean, xgb_std = xgBoost_cv(X_encode, Y_encode)
        lightGBM_mean, lightGBM_std = lightGBM_cv(X_encode, Y_encode)
        SVM_res = SVM_cv(X_encode, Y_encode)

        sheet1.write(n + 2, 0, var)
        sheet1.write(n + 2, 2, xgb_mean)
        sheet1.write(n + 2, 4, xgb_std)
        sheet1.write(n + 2, 6, lightGBM_mean)
        sheet1.write(n + 2, 8, lightGBM_std)
        for i in range(len(SVM_res)):
            sheet1.write(n + 2, 4 * i + 10, SVM_res[i][0])
            sheet1.write(n + 2, 4 * i + 12, SVM_res[i][1])

    wb.save('result_all.xls')

if __name__ == '__main__':
    runGridSearchCV()
