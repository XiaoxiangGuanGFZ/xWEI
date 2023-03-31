"""
EtA functions in xWEI computation

Author: Xiaoxiang Guan (guan@gfz-potsdam.de)
Latest updated: 2023-03-30

"""

import numpy as np
import pandas as pd
from scipy.stats import genextreme
from scipy import interpolate
import math


def GEV_para_derive(dGEV_para, durations=[24, 48, 72, 96]):
    """
    derive the GEV parameters from the dGEV parameter with the durations;
    see reference: F. S. Fauer, J. Ulrich, O. E. Jurado and H. W. Rust.
        Flexible and consistent quantile estimation for intensity–duration–frequency curves
        with doi:  https://doi.org/10.5194/hess-25-6479-2021
    Parameters:
        dGEV_para: the dGEV parameter dataframe
        with the columns: ['ID', 'mut', 'sigma0', 'xi', 'theta', 'eta', 'tau', 'eta2']
    """
    for duration in durations:
        Col_name = 'mut_' + str(duration)
        dGEV_para[Col_name] = dGEV_para.apply(
            lambda x: x.mut * (x.sigma0 * pow(duration + x.theta, -x.eta) + x.tau), axis=1
        )
        Col_name = 'sigma_' + str(duration)
        dGEV_para[Col_name] = dGEV_para.apply(
            lambda x: x.sigma0 * pow(duration + x.theta, -(x.eta + x.eta2)) + x.tau, axis=1
        )
    return dGEV_para


# ------ functions about EtA curves -----------
def EtA_CDF(df_rr_1event, dGEV_para_domain, duration=24, duration_max=96):
    """
    derive the return period(cdf) of the rainfall intensity for each grid cell winthin the EPE domain
    Parameters:
        df_rr_1event: a dataframe with rainfall intensities for one EPE event
        within a domain (covering several grid cells)
        dGEV_para_domain: GEV parameter for different durations and grids
    Return:
        a dataframe with the return period corresponding to each cell in df_rr_1event
    """
    nrow = df_rr_1event.shape[0]
    rp_list = []  # list of return periods
    for i in range(0, nrow):
        # iterate each row; each row represents each grid within the EPE domain
        GEV_xi = dGEV_para_domain['xi'].iloc[i]
        GEV_loc = dGEV_para_domain['mut_' + str(duration)].iloc[i]
        GEV_scale = dGEV_para_domain['sigma_' + str(duration)].iloc[i]
        # cdfs = genextreme.cdf(df_rr_1event.iloc[i, :], c=GEV_xi, loc=GEV_loc, scale=GEV_scale)
        cdfs = genextreme.cdf(df_rr_1event.iloc[i], c=GEV_xi, loc=GEV_loc, scale=GEV_scale)
        if duration == duration_max:
            if cdfs > 0.999:
                cdfs = 0.999
        else:
            cdfs[cdfs > 0.999] = 0.999
        rp = 1/(1-cdfs)  # 1-D array
        rp_list.append(pd.Series(rp))  # 1-D pd.Series
    out = pd.concat(rp_list, axis=1).T  # the (row)index is natural [0,1,...]
    return out


def EtA_curve(df_rp_1event, Area):
    """
    compute the EtA curve from sorted return periods
    Parameters:
        df_rp_1event: a dataframe with return periods for one EPE event
        within a domain (covering several grid cells)
        Area: the area vector for the grids within the EPE domain
    """
    ncol = df_rp_1event.shape[1]
    nrow = df_rp_1event.shape[0]
    Eta_max = []
    Eta_curves = []
    for k in range(0, ncol):
        # sort the rp values in the decreasing order
        df_rp = pd.concat([Area, df_rp_1event.iloc[:, k]],
                          axis=1).set_axis(['area', 'rp'],
                                           axis=1, inplace=False).sort_values('rp', ascending=False)
        n = np.array(range(1, nrow + 1))
        A = np.cumsum(df_rp.area)
        item_rp = np.cumsum(np.log(df_rp.rp))
        Eta = item_rp / n * np.sqrt(A / np.pi)
        Eta_max.append(Eta.max())
        Eta_curve = pd.concat([A, Eta], axis=1)
        Eta_curves.append(Eta_curve)
    # return the index of the largest value in the 1-D array
    index_max = np.argmax(np.array(Eta_max))
    # return a pd.DataFrame: two columns: area and Eta
    out = Eta_curves[index_max].set_axis(['area', 'Eta'],
                                         axis=1, inplace=False).reset_index(drop=True)
    return out


def EtA_curve_Mul(df_rr_event, dGEV_para_domain, Area, durations=[24, 48, 72, 96]):
    """
    compute the EtA curves for EPEs
    Parameters:
         df_rr_event: dataframe of rainfall intensity
                    ncol: the number of durations
                    nrow: the number of grids within the EPE domain
         dGEV_para_domain: dataframe; the GEV parameters used to derive the return periods
         Area: a 1-D pd.Series (vector); the area (size) of the grids
    Return:
        duration-related EtA curves; input for xWEI functions
    """
    EtA_mul = []
    for duration in durations:
        if duration == 24:
            df_rr_1event = df_rr_event.copy()
        elif duration == 48:
            df_rr_1event = pd.concat(
                [
                    df_rr_event.iloc[:, [0, 1]].mean(axis=1),
                    df_rr_event.iloc[:, [1, 2]].mean(axis=1),
                    df_rr_event.iloc[:, [2, 3]].mean(axis=1)
                ], axis=1
            )
        elif duration == 72:
            df_rr_1event = pd.concat(
                [
                    df_rr_event.iloc[:, [0, 1, 2]].mean(axis=1),
                    df_rr_event.iloc[:, [1, 2, 3]].mean(axis=1)
                ], axis=1
            )
        else:  # duration == 96
            df_rr_1event = df_rr_event.mean(axis=1)
        df_rp_1event = EtA_CDF(df_rr_1event, dGEV_para_domain,
                               duration=duration, duration_max=np.array(durations).max())
        EtA_duration = EtA_curve(df_rp_1event, Area)
        EtA_duration = EtA_duration.set_axis(['area_'+str(duration), 'Eta_' + str(duration)],
                                             axis=1, inplace=False)
        EtA_mul.append(EtA_duration)
    out = pd.concat(
        EtA_mul, axis=1
    )
    return out


def inter_df(df, durations=[24, 48, 72, 96]):
    """
    interpolate the EtA curves at the axis of area
    Parameters:
        df: EtA dataframe, output from EtA_curve_Mul() function
    """
    box_area = int(math.floor(df.iloc[:, 0].tail(1) / 100) * 100)  # REAL AREA (DOMAIN SIZE)
    x_new = np.arange(100, box_area + 100, 100)
    y_out = []
    for duration in durations:
        cname_area = 'area_' + str(duration)
        cname_Eta = 'Eta_' + str(duration)
        x = np.insert(np.array(df[cname_area]), 0, 100.0)
        y0 = np.array(df[cname_Eta])
        y = np.insert(y0, 0, y0[0])
        f = interpolate.interp1d(x, y)
        y_new = f(x_new)
        y_out.append(pd.Series(y_new))
    y_out = pd.concat(y_out, axis=1)
    out = pd.concat(
        [pd.Series(x_new), y_out], axis=1
    )
    out.set_axis(['area'] + [str(duration) for duration in durations],
                 axis='columns', inplace=True)
    return out

