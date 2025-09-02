import numpy as np
from sklearn.metrics import log_loss, accuracy_score, classification_report
from purged_Kfold import PurgedKFold


def cvScore(clf, obs_frame=None, features=[], scoring="neg_log_loss", sample_weight=None, cv=None, cvGen=None,
            embargo_size=None, report=False):
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise Exception("Wrong scoring method")

    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, embargo_size=embargo_size)

    score = []
    for idx, (train, test) in enumerate(cvGen.split(obs_frame=obs_frame)):
        X, y = obs_frame.loc[train, features], obs_frame.loc[train, "barrier"]
        fit = clf.fit(X=X, y=y, sample_weight=sample_weight)

        if report:
            print("Fold ", idx)

            print("Train Performace:")
            print(classification_report(obs_frame.loc[train, "barrier"], fit.predict(obs_frame.loc[train, features])))

            print("Test Performace:")
            print(classification_report(obs_frame.loc[test, "barrier"], fit.predict(obs_frame.loc[test, features])))

        if scoring == "neg_log_loss":
            prob = fit.predict_proba(obs_frame.loc[test, features])
            score_ = -log_loss(obs_frame.loc[test, "barrier"], prob, sample_weight=sample_weight, labels=clf.classes_)
        else:
            pred = fit.predict(obs_frame.loc[test, features])
            score_ = accuracy_score(obs_frame.loc[test, "barrier"], pred, sample_weight=sample_weight)

        score.append(score_)

    return np.array(score)
