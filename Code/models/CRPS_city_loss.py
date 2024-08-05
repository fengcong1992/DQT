import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
 
def CRPS_cityloss(y_pred,y_true,city): 
    ### inp dtype: torch.Tensor
    ###     shape: (batch size, 24 hours' obs, 11 percentiles)
    ### target dtype: torch.Tensor
    ###     shape: (batch size, 24 hours' obs)
    ### inp and target should be on the same device
    n = len(y_true)
    loss = [] 
    
    
    if city == 'Donalsonville':
        y_pred = y_pred[:,6:22,:]
        y_true = y_true[:,6:22]
        fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*16,(16,11))) 
    elif city == 'SanAntonio':
        y_pred = y_pred[:,6:22, :]
        y_true = y_true[:,6:22]
        fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*16,(16,11))) 
    elif city == 'Amity':
        y_pred = y_pred[:,5:23, :]
        y_true = y_true[:,5:23]
        fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*18,(18,11))) 
    elif city == 'Waianae':
        y_pred = y_pred[:,5:21, :]
        y_true = y_true[:,5:21]
        fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*16,(16,11))) 
    else:
        print(f"Invalid city: {city}")
        y_pred = None
        
    for i in range(n): 
        obs = y_true[i,:]
        fx = y_pred[i,:,:]
        tmp = continuous_ranked_probability_score(obs, fx, fx_prob) 
        loss.append(tmp) 
    loss = torch.Tensor(loss)
    # print("loss: ", loss)
    loss.requires_grad_()
    return loss

def continuous_ranked_probability_score(obs, fx, fx_prob):
    """Continuous Ranked Probability Score (CRPS).

    .. math::

        \\text{CRPS} = \\frac{1}{n} \\sum_{i=1}^n \\int_{-\\infty}^{\\infty}
        (F_i(x) - \\mathbf{1} \\{x \\geq y_i \\})^2 dx

    where :math:`F_i(x)` is the CDF of the forecast at time :math:`i`,
    :math:`y_i` is the observation at time :math:`i`, and :math:`\\mathbf{1}`
    is the indicator function that transforms the observation into a step
    function (1 if :math:`x \\geq y`, 0 if :math:`x < y`). In other words, the
    CRPS measures the difference between the forecast CDF and the empirical CDF
    of the observation. The CRPS has the same units as the observation. Lower
    CRPS values indicate more accurate forecasts, where a CRPS of 0 indicates a
    perfect forecast. [1]_ [2]_ [3]_

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n, d) array_like
        Forecasts (physical units) of the right-hand-side of a CDF with d
        intervals (d >= 2), e.g., fx = [10 MW, 20 MW, 30 MW] is interpreted as
        <= 10 MW, <= 20 MW, <= 30 MW.
    fx_prob : (n, d) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    crps : float
        The Continuous Ranked Probability Score, with the same units as the
        observation.

    Raises
    ------
    ValueError
        If the forecasts have incorrect dimensions; either a) the forecasts are
        for a single sample (n=1) with d CDF intervals but are given as a 1D
        array with d values or b) the forecasts are given as 2D arrays (n,d)
        but do not contain at least 2 CDF intervals (i.e. d < 2).

    Notes
    -----
    The CRPS can be calculated analytically when the forecast CDF is of a
    continuous parametric distribution, e.g., Gaussian distribution. However,
    since the Solar Forecast Arbiter makes no assumptions regarding how a
    probabilistic forecast was generated, the CRPS is instead calculated using
    numerical integration of the discretized forecast CDF. Therefore, the
    accuracy of the CRPS calculation is limited by the precision of the
    forecast CDF. In practice, this means the forecast CDF should 1) consist of
    at least 10 intervals and 2) cover probabilities from 0% to 100%.

    References
    ----------
    .. [1] Matheson and Winkler (1976) "Scoring rules for continuous
           probability distributions." Management Science, vol. 22, pp.
           1087-1096. doi: 10.1287/mnsc.22.10.1087
    .. [2] Hersbach (2000) "Decomposition of the continuous ranked probability
           score for ensemble prediction systems." Weather Forecast, vol. 15,
           pp. 559-570. doi: 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
    .. [3] Wilks (2019) "Statistical Methods in the Atmospheric Sciences", 4th
           ed. Oxford; Waltham, MA; Academic Press.

    """
    # match observations to fx shape: (n,) => (n, d)
    # if np.ndim(fx) < 2:
    if fx.dim()<2:
        raise ValueError("forecasts must be 2D arrays (expected (n,d), got"
                         f"{np.shape(fx)})")
    # elif np.shape(fx)[1] < 2:
    elif fx.shape[1] < 2:
        raise ValueError("forecasts must have d >= 2 CDF intervals "
                         f"(expected >= 2, got {np.shape(fx)[1]})")

    n = len(fx)

    # extend CDF min to ensure obs within forecast support
    # fx.shape = (n, d) ==> (n, d + 1)
    fx_min = torch.minimum(obs, fx[:, 0])
    fx = torch.hstack([fx_min[:, np.newaxis], fx])
    fx_prob = torch.hstack([torch.zeros([n, 1]), fx_prob])

    # extend CDF max to ensure obs within forecast support
    # fx.shape = (n, d + 1) ==> (n, d + 2)
    idx = (fx[:, -1] < obs)
    fx_max = torch.maximum(obs, fx[:, -1])
    # fx = torch.hstack([fx, fx_max[:, np.newaxis]])
    fx = torch.hstack([fx, fx_max[:, None]])
    fx_prob = torch.hstack([fx_prob, torch.full([n, 1], 100)])

    # indicator function:
    # - left of the obs is 0.0
    # - obs and right of the obs is 1.0
    o = torch.where(fx >= obs[:, None], 1.0, 0.0)

    # correct behavior when obs > max fx:
    # - should be 0 over range: max fx < x < obs
    o[idx, -1] = 0.0

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # integrate along each sample, then average all samples
    crps = torch.mean(torch.trapezoid((f - o) ** 2, x=fx, axis=1))

    return crps