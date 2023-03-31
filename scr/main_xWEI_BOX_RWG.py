"""
xWEI computation
- RWG outputs (1950-2021; 72 years)
- 100 runs
- daily
- 37 BOXes across Germany (100km*100km)

Author: Xiaoxiang Guan (guan@gfz-potsdam.de)
Latest updated: 2023-03-30

"""

import numpy as np
import pandas as pd
import datetime
import math
import os.path
from os import listdir

import xWEI_weight
import xWEI_EtA

ws = 'D:/WEI_RWG/data_EOBS_RWG/'
# ---- Combine the Paul's weight generation and my interpolation --------
# -- check the size (domain area) of each box
# for box in range(1, 38):
#     fp_box = ws + 'FID-' + str(box) + '-Eta/'
#     events_list = listdir(fp_box)
#     df = pd.read_csv(fp_box + events_list[0],
#                      header=0, sep=','
#                      )
#     box_area = int(math.floor(df.area.tail(1) / 100) * 100)
#     print(
#         'Area_precise: ' + str(list(df.area.tail(1))[0]) + '; Size: ' + str(box_area)
#     )

# --- generate the weights ----
ws_weight = ws + 'xWEI/weights/'

for box in range(1, 7):
    fp_box = ws + 'FID-' + str(box) + '-Eta/'
    weight_name = ['vtx-' + str(box), 'wts-' + str(box)]
    if not os.path.exists(ws_weight + weight_name[0] + '.csv'):
        events_list = listdir(fp_box)
        # only import the first dataset file
        # derive the area of the box (the length)
        df = pd.read_csv(fp_box + events_list[0],
                         header=0, sep=','
                         )
        box_area = int(math.floor(df.area.tail(1) / 100) * 100)  # REAL AREA (DOMAIN SIZE)
        # most 10000 km2
        print(str(box) + ': ' + str(box_area))
        # compute the weights
        xWEI_weight.weight_generate(domain_size=int(box_area / 100),  # scaling size
                                    resolution=1000,  # 1 km
                                    durations=['24', '48', '72', '96'],
                                    outpath=ws_weight, outnames=weight_name
                                    )

# ----- import data
Box_id = pd.read_csv(
    # the ids of boxes, together with the E-OBS grid ids within the boxes
    'D:/WEI_RWG/GIS/ASCII/EOBS540_BOX100_union_IDs.csv',
    header=0, sep=','
).dropna().reset_index(drop=True)
FIDs = Box_id['fid'].unique()  # an 1-D array
Box_id['Area'] = Box_id['Area'].transform(lambda x: round(x / 1000000, 2))
Box_id['Area_2'] = Box_id['Area_2'].transform(lambda x: round(x / 1000000, 2))
Box_id['piece_area'] = Box_id['piece_area'].transform(lambda x: round(x / 1000000, 2))
Box_id.EOBS_ID = pd.to_numeric(Box_id.EOBS_ID, downcast='integer')

dGEV_para = pd.read_csv(
    # the dGEV parameters of the E-OBS grids
    ws + '/dGEV_parameters-Felix_HESS2021_4d.csv',
    header=None, sep=','
).set_axis(['ID', 'mut', 'sigma0', 'xi', 'theta', 'eta', 'tau', 'eta2'], axis=1, inplace=False)

dGEV_para = xWEI_EtA.GEV_para_derive(dGEV_para)

# -- compute the EtA and xWEI
date_series = pd.date_range(datetime.datetime.strptime('1950-01-01', '%Y-%m-%d'),
                            datetime.datetime.strptime('2021-12-31', '%Y-%m-%d'),
                            freq='D')
len(date_series)  # the length of the daily rainfall observations
# df_rr.shape

durations = [24, 48, 72, 96]  # 4 durations
duration_levels = [str(duration) for duration in durations]
scaling = np.log(10000) / np.log(100)

ws_RWG_output = 'Y:/ClimXtreme/RWG/Output/sim_rr_tg_tn_tx_corrected/'
runs = range(1, 101)   # RWG runs; range(100, 0, -1)
for run in runs:
    df_name = 'rr_sim_non_cla6_B2_' + str(run) + '.txt'
    fp_xwei = ws + 'xWEI/output/RWG_' + 'rr_sim_non_cla6_B2_' + str(run) + '.csv'
    if not os.path.exists(fp_xwei):
        df_rr = pd.read_csv(
            # df for the generated daily rainfall
            ws_RWG_output + df_name,
            header=None, sep=' '
        )
        for fid in FIDs:
            # --- import weights for the box
            fname_weight = ['vtx-' + str(fid), 'wts-' + str(fid)]
            if fid < 6:
                vtx = pd.read_csv(ws_weight + fname_weight[0] + '.csv',
                                  header=0, sep=','
                                  )
                wts = pd.read_csv(ws_weight + fname_weight[1] + '.csv',
                                  header=0, sep=','
                                  )
            else:
                # for these Boxes, with the same area: 10000 km2
                vtx = pd.read_csv(ws_weight + 'vtx-6.csv',
                                  header=0, sep=','
                                  )
                wts = pd.read_csv(ws_weight + 'wts-6.csv',
                                  header=0, sep=','
                                  )
            # --- import rr data, GEV parameters
            Box_id_domain = Box_id[Box_id.fid == fid].sort_values('EOBS_ID').reset_index(drop=True)
            EOBS_area = Box_id_domain.piece_area
            # filter out the dGEV params for the domain (box)
            dGEV_para_domain = dGEV_para.iloc[Box_id_domain.EOBS_ID - 1, :].reset_index(drop=True)
            # filter the rr_data for the domain
            # only the data array, without the date columns
            ri_data_domain = df_rr.iloc[:, Box_id_domain.EOBS_ID - 1]

            xWEI = []
            dates = []
            for i in range(0, len(date_series) - (len(durations) - 1)):
                if ri_data_domain.iloc[i, :].mean() > 0:
                    # iterate each day (mean rainfall > 0: rainy day), then compute the EtA
                    # extraction: rr field for one EPE
                    df_rr_1 = ri_data_domain.iloc[range(i, i + 4), :].T.reset_index(drop=True)
                    EtA_df = xWEI_EtA.EtA_curve_Mul(df_rr_1, dGEV_para_domain, Area=EOBS_area, durations=durations)
                    df = xWEI_EtA.inter_df(EtA_df, durations=durations)
                    xwei = xWEI_weight.xWEI_weight(
                        df.iloc[:, 1:5],
                        vtx, wts, resolution=1000,
                        duration_levels=duration_levels
                    ) * scaling
                    date = date_series[i].strftime('%Y-%m-%d')
                    xWEI.append(xwei)
                    dates.append(date)
                    print('RUN_' + str(run) + ': BOX_' + str(fid) + ': ' + date)
            out = pd.DataFrame(
                {'BOX_id': fid, 'date': dates, 'xWEI': xWEI}
            )
            out.to_csv(fp_xwei, index=False, mode='a', header=False)


