import numpy as np
import torch

def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_scores = 1-cal_smx[np.arange(n),cal_labels]
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    prediction_sets = val_smx >= (1-qhat)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
    
def aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
        
def raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    lam_reg = 0.01
    k_reg = min(5, cal_smx.shape[1])
    disallow_zero_sets = False 
    rand = True
    reg_vec = np.array(k_reg*[0,] + (cal_smx.shape[1]-k_reg)*[lam_reg,])[None,:]

    cal_pi = cal_smx.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
    
def cqr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    cal_scores = np.maximum(cal_labels-cal_upper, cal_lower-cal_labels)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    prediction_sets = [val_lower - qhat, val_upper + qhat]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(val_upper + qhat - (val_lower - qhat))
    return prediction_sets, cov, eff

def qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    prediction_sets = [val_lower, val_upper]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(val_upper - val_lower)
    return prediction_sets, cov, eff

def threshold(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    
    prediction_sets = np.take_along_axis(val_srt <= 1-alpha, val_pi.argsort(axis=1), axis=1)
    prediction_sets[np.arange(prediction_sets.shape[0]), val_pi[:, 0]] = True

    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff


def run_conformal_classification(pred, data, n, alpha, score = 'aps', 
                                 calib_eval = False, validation_set = False, 
                                 use_additional_calib = False, return_prediction_sets = False, calib_fraction = 0.5): 
    if calib_eval:
        n_base = int(n * (1-calib_fraction))
    else:
        n_base = n
        
    logits = torch.nn.Softmax(dim = 1)(pred).detach().cpu().numpy()
    
    if validation_set:
        smx = logits[data.valid_mask]
        labels = data.y[data.valid_mask].detach().cpu().numpy()
        n_base = int(len(np.where(data.valid_mask)[0])/2)
    else:
        if calib_eval:
            smx = logits[data.calib_test_real_mask]
            labels = data.y[data.calib_test_real_mask].detach().cpu().numpy()
        else:
            smx = logits[data.calib_test_mask]
            labels = data.y[data.calib_test_mask].detach().cpu().numpy()

    cov_all = []
    eff_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
        
    for k in range(100):
        idx = np.array([1] * n_base + [0] * (smx.shape[0]-n_base)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_smx, val_smx = smx[idx,:], smx[~idx,:]
        cal_labels, val_labels = labels[idx], labels[~idx]
        
        if use_additional_calib and calib_eval:
            smx_add = logits[data.calib_eval_mask]
            labels_add = data.y[data.calib_eval_mask].detach().cpu().numpy()
            cal_smx = np.concatenate((cal_smx, smx_add))
            cal_labels = np.concatenate((cal_labels, labels_add))
            
        n = cal_smx.shape[0]
        
        if score == 'tps':
            prediction_sets, cov, eff = tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
        elif score == 'aps':
            prediction_sets, cov, eff = aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
        elif score == 'raps':
            prediction_sets, cov, eff = raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
        elif score == 'threshold':
            prediction_sets, cov, eff = threshold(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
            
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)
    
    if return_prediction_sets:
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all)

def run_conformal_regression(pred, data, n, alpha, calib_eval = False, validation_set = False, use_additional_calib = False, return_prediction_sets = False, calib_fraction = 0.5, score = 'cqr'): 
    if calib_eval:
        n_base = int(n * (1-calib_fraction))
    else:
        n_base = n
    
    try:
        pred = pred.detach().cpu().numpy()
    except:
        pass
                     
    if validation_set:
        smx = pred[data.valid_mask]
        labels = data.y[data.valid_mask].detach().cpu().numpy().reshape(-1)
        n_base = int(len(np.where(data.valid_mask)[0])/2)
    else:
        if calib_eval:
            smx = pred[data.calib_test_real_mask]
            labels = data.y[data.calib_test_real_mask].detach().cpu().numpy().reshape(-1)
        else:
            smx = pred[data.calib_test_mask]
            labels = data.y[data.calib_test_mask].detach().cpu().numpy().reshape(-1)
    
    cov_all = []
    eff_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
    for k in range(100):
        upper, lower = smx[:, 2], smx[:, 1]

        idx = np.array([1] * n_base + [0] * (labels.shape[0]-n_base)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_labels, val_labels = labels[idx], labels[~idx]
        cal_upper, val_upper = upper[idx], upper[~idx]
        cal_lower, val_lower = lower[idx], lower[~idx]
        if score == 'cqr':
            prediction_sets, cov, eff = cqr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha)
        elif score == 'qr':
            prediction_sets, cov, eff = qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha)
            
        
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)
    
    if return_prediction_sets:
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all)