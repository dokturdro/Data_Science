class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


def results_kfold(X, y, models, regression=True):
    result_dict = {}
    cv = KFold(n_splits=5, random_state=42, shuffle=True)

    reg_columns = ['RMSE', 'MAE', 'R2']
    clf_columns = ['F1', 'PRECIS', 'RECALL', 'ROC AUC', 'T POS', 'F POS', 'F NEG', 'T NEG']

    if regression:

        for name, regressor in models:

            rmse = []
            mae = []
            r2 = []

            for train_index, test_index in cv.split(X):
                X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
                    test_index]
                score = regressor.fit(X_train, y_train)
                prediction = score.predict(X_test)

                rmse.append(np.sqrt(mean_squared_error(y_test, prediction)))
                mae.append(mean_absolute_error(y_test, prediction))
                r2.append(score.score(X_train, y_train))

            if name not in result_dict:
                result_dict[name] = []

            result_dict[name].append(np.mean(rmse))
            result_dict[name].append(np.mean(mae))
            result_dict[name].append(np.mean(r2))

        result_dict = pd.DataFrame.from_dict(result_dict, orient='index')
        result_dict.columns = reg_columns

    else:

        for name, classifier in models:

            f1 = []
            precis = []
            recall = []
            roc_auc = []
            t_pos = []
            f_pos = []
            f_neg = []
            t_neg = []

            for train_index, test_index in cv.split(X):
                X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
                    test_index]
                score = classifier.fit(X_train, y_train)
                prediction = score.predict(X_test)

                p_r_f1 = precision_recall_fscore_support(y_test, prediction, average='weighted')
                precis.append(p_r_f1[0])
                recall.append(p_r_f1[1])
                f1.append(p_r_f1[2])
                roc_auc.append(roc_auc_score(y_test, prediction))

                cm = confusion_matrix(y_test, prediction)
                t_pos.append(cm[0][0])
                f_pos.append(cm[0][1])
                f_neg.append(cm[1][0])
                t_neg.append(cm[1][1])

            if name not in result_dict:
                result_dict[name] = []

            result_dict[name].append(np.mean(f1))
            result_dict[name].append(np.mean(precis))
            result_dict[name].append(np.mean(recall))
            result_dict[name].append(np.mean(roc_auc))
            result_dict[name].append(np.mean(t_pos))
            result_dict[name].append(np.mean(f_pos))
            result_dict[name].append(np.mean(f_neg))
            result_dict[name].append(np.mean(t_neg))

        result_dict = pd.DataFrame.from_dict(result_dict, orient='index')
        result_dict.columns = clf_columns

    return result_dict