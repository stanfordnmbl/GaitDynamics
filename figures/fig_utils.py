import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr


def get_scores(y_true, y_pred, y_fields, exclude_swing_phase=False):
    scores = []
    for col, field in enumerate(y_fields):
        if exclude_swing_phase:
            non_zero_idx = np.where(y_true[:, col] != 0)[0]
            true_, pred_ = y_true[non_zero_idx, col], y_pred[non_zero_idx, col]
        else:
            true_, pred_ = y_true[:, col], y_pred[:, col]

        # plt.figure()
        # plt.plot(true_, label='true')
        # plt.plot(pred_, label='pred')
        # plt.legend()
        # plt.title(field)

        r2 = r2_score(true_, pred_)
        rmse = np.sqrt(mse(true_, pred_))
        cor_value = pearsonr(true_, pred_)[0]
        score_one_field = {'field': field, 'r2': r2, 'rmse': rmse, 'cor_value': cor_value}
        scores.append(score_one_field)
    return scores
