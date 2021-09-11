import os
import sys
import time
import pickle
import paddle
from urllib import request
from collections import OrderedDict


def pth2pdparams(model, pth_path):
    import torch
    pd_pw = model.state_dict()
    pt_pw = torch.load(pth_path) # pth参数
    pd_new_dict = OrderedDict()
    # 修改对应
    pt_pw_tmp = pt_pw.copy()
    for k in pt_pw_tmp.keys():
        sq = k.split(".")
        if sq[-1] == "running_mean":
            sq[-1] = "_mean"
        elif sq[-1] == "running_var":
            sq[-1] = "_variance"
        nk = ".".join(sq)
        pt_pw[nk] = pt_pw.pop(k)
    # print(len(pt_pw), len(pd_pw))
    # for kt, kp in zip(pt_pw.keys(), pd_pw.keys()):
    #     print(kt, kp)
    for kp in pd_pw.keys():
        if "fc" in kp.split("."):
            pd_new_dict[kp] = pt_pw[kp].detach().numpy().T
        else:
            pd_new_dict[kp] = pt_pw[kp].detach().numpy()
    pdparams_path = pth_path.split(".")[0] + '.pdparams'
    with open(pdparams_path, 'wb') as f:
        pickle.dump(pd_new_dict, f)
    print('\nConvert finished!')
    return pdparams_path