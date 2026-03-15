import numpy as np
import scipy

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def distance_metrics(preds,gt):
    preds=preds[:,:,0:3]
    gt=gt[:,:,0:3]
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(),errors[:,-1].mean(),errors

def metric(pred, true):
    mae = MAE(pred, true) # 平均绝对误差 (Mean Absolute Error)
    mse = MSE(pred, true) # 均方误差 (Mean Squared Error)
    rmse = RMSE(pred, true) # 均方根误差 (Root Mean Squared Error)
    mape = MAPE(pred, true) # 平均绝对百分比误差 (Mean Absolute Percentage Error)
    mspe = MSPE(pred, true) # 均方百分比误差 (Mean Square Percentage Error)
    rse = RSE(pred, true) # 相对平方误差 (Relative Squared Error)
    corr = CORR(pred, true) # 相关系数 (Correlation Coefficient)
    eulid, _, _ = distance_metrics(pred, true) # 欧氏距离 (Euclidean Distance)
    rae = np.sum(np.abs(true - pred)) / np.sum(np.abs(true - np.mean(true))) # Relative Absolute Error相对绝对值误差
    rel_rmse = rmse / np.mean(true) # Relative Root Mean Squared Error 相对均方根误差
    return mae, mse, rmse, mape, mspe, rse, corr, eulid, rae, rel_rmse
