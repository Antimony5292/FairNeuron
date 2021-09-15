import logging
from secml.data import CDataset
from secml.array import CArray

def attack_keras_model(X, Y, S, nb_attack=25, dmax=0.1):
    """
    Generates an adversarial attack on a general model.
    :param X: Original inputs on which the model is trained
    :param Y: Original outputs on which the model is trained
    :param S: Original protected attributes on which the model is trained
    :return: Adversarial dataset (i.e. new data points + original input)
    """



    data_set_encoded_secML = CArray(X, dtype=float, copy=True)
    data_set_encoded_secML = CDataset(data_set_encoded_secML, Y)

    n_tr = round(0.66 * X.shape[0])
    n_ts = X.shape[0] - n_tr

    logger.debug(X.shape)
    logger.debug(n_tr)
    logger.debug(n_ts)

    from secml.data.splitter import CTrainTestSplit
    splitter = CTrainTestSplit(train_size=n_tr, test_size=n_ts)

    tr_set_secML, ts_set_secML = splitter.split(data_set_encoded_secML)

    from secml.ml.classifiers import CClassifierSVM
    from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
    from secml.ml.kernel import CKernelRBF
    clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())

    xval_params = {'C': [1e-4, 1e-3, 1e-2, 0.1, 1], 'kernel.gamma': [0.01, 0.1, 1, 10, 100, 1e3]}

    random_state = 999

    from secml.data.splitter import CDataSplitterKFold
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

    logger.debug("Estimating the best training parameters...")
    best_params = clf.estimate_parameters(
        dataset=tr_set_secML,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy',
        perf_evaluator='xval'
    )
    logger.debug("The best training parameters are: ", best_params)

    logger.debug(clf.get_params())
    logger.debug(clf.num_classifiers)

    from secml.ml.peval.metrics import CMetricAccuracy
    metric = CMetricAccuracy()

    # Train the classifier
    clf.fit(tr_set_secML)
    logger.debug(clf.num_classifiers)

    # Compute predictions on a test set
    y_pred = clf.predict(ts_set_secML.X)

    # Evaluate the accuracy of the classifier
    acc = metric.performance_score(y_true=ts_set_secML.Y, y_pred=y_pred)

    logger.debug("Accuracy on test set: {:.2%}".format(acc))

    # Prepare attack configuration

    noise_type = 'l2'   # Type of perturbation 'l1' or 'l2'
    lb, ub = 0, 1       # Bounds of the attack space. Can be set to `None` for unbounded
    y_target = None     # None if `error-generic` or a class label for `error-specific`

    solver_params = {
        'eta': 0.1,         # grid search resolution
        'eta_min': 0.1,
        'eta_max': None,    # None should be ok
        'max_iter': 1000,
        'eps': 1e-2         # Tolerance on the stopping crit.
    }

    # Run attack

    from secml.adv.attacks.evasion import CAttackEvasionPGDLS
    pgd_ls_attack = CAttackEvasionPGDLS(
        classifier=clf,
        surrogate_classifier=clf,
        surrogate_data=tr_set_secML,
        distance=noise_type,
        dmax=dmax,
        lb=lb, ub=ub,
        solver_params=solver_params,
        y_target=y_target)

    nb_feat = X.shape[1]

    result_pts = np.empty([nb_attack, nb_feat])
    result_class = np.empty([nb_attack, 1])

    # take a point at random being the starting point of the attack and run the attack
    import random
    for nb_iter in range(0, nb_attack):
        rn = random.randint(0, ts_set_secML.num_samples - 1)
        x0, y0 = ts_set_secML[rn, :].X, ts_set_secML[rn, :].Y,

        try:
            y_pred_pgdls, _, adv_ds_pgdls, _ = pgd_ls_attack.run(x0, y0)
            adv_pt = adv_ds_pgdls.X.get_data()
            # np.asarray([np.asarray(row, dtype=float) for row in y_tr], dtype=float)
            result_pts[nb_iter] = adv_pt
            result_class[nb_iter] = y_pred_pgdls.get_data()[0]
        except ValueError:
            logger.warning("value error on {}".format(nb_iter))

    return result_pts, result_class, ts_set_secML[:nb_attack, :].Y