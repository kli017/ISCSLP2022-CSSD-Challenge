# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

from cmath import log
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from scipy.signal import medfilt
from loguru import logger
import torch
from functools import partial
import sys
#sys.path.append("/home/likai/eend_fastapi/eend/")
from eend.pytorch_backend.models import fix_state_dict
from eend.pytorch_backend.models import ConfomerDiarization
import feature
import kaldi_data
import time
from spectralcluster import SpectralClusterer


def _gen_chunk_indices(data_len, chunk_size):
    step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step


def get_cl_sil(args, acti, cls_num):
    n_chunks = len(acti)
    mean_acti = np.array([np.mean(acti[i], axis=0)
                         for i in range(n_chunks)]).flatten()
    n = args.num_speakers
    sil_spk_th = args.sil_spk_th

    cl_lst = []
    sil_lst = []
    for chunk_idx in range(n_chunks):
        if cls_num is not None:
            if args.num_speakers > cls_num:
                mean_acti_bi = np.array([mean_acti[n * chunk_idx + s_loc_idx]
                                        for s_loc_idx in range(n)])
                min_idx = np.argmin(mean_acti_bi)
                mean_acti[n * chunk_idx + min_idx] = 0.0

        for s_loc_idx in range(n):
            a = n * chunk_idx + (s_loc_idx + 0) % n
            b = n * chunk_idx + (s_loc_idx + 1) % n
            if mean_acti[a] > sil_spk_th and mean_acti[b] > sil_spk_th:
                cl_lst.append((a, b))
            else:
                if mean_acti[a] <= sil_spk_th:
                    sil_lst.append(a)

    return cl_lst, sil_lst


def clustering(args, svec, cls_num, ahc_dis_th, cl_lst, sil_lst):
    org_svec_len = len(svec)
    svec = np.delete(svec, sil_lst, 0)

    # update cl_lst idx
    _tbl = [i - sum(sil < i for sil in sil_lst) for i in range(org_svec_len)]
    cl_lst = [(_tbl[_cl[0]], _tbl[_cl[1]]) for _cl in cl_lst]

    distMat = distance.cdist(svec, svec, metric='euclidean')
    for cl in cl_lst:
        distMat[cl[0], cl[1]] = args.clink_dis
        distMat[cl[1], cl[0]] = args.clink_dis


    scp_clusterer = SpectralClusterer(
         min_clusters=2,
         max_clusters=7,
         autotune=None,
         laplacian_type=None,
         refinement_options=None,
         custom_dist="cosine")
    
    labels = scp_clusterer.predict(svec)

    if cls_num is not None:
        print("oracle n_clusters is known")
    else:
        print("oracle n_clusters is unknown")
        print("estimated n_clusters by constraind AHC: {}"
              .format(len(np.unique(labels))))
        cls_num = len(np.unique(labels))

    sil_lab = cls_num
    insert_sil_lab = [sil_lab for i in range(len(sil_lst))]
    insert_sil_lab_idx = [sil_lst[i] - i for i in range(len(sil_lst))]
    print("insert_sil_lab : {}".format(insert_sil_lab))
    print("insert_sil_lab_idx : {}".format(insert_sil_lab_idx))
    clslab = np.insert(labels,
                       insert_sil_lab_idx,
                       insert_sil_lab).reshape(-1, args.num_speakers)
    print("clslab : {}".format(clslab))

    return clslab, cls_num


def kmclustering(args, svec, cls_num, ahc_dis_th, cl_lst, sil_lst):
    org_svec_len = len(svec)
    svec = np.delete(svec, sil_lst, 0)

    # update cl_lst idx
    _tbl = [i - sum(sil < i for sil in sil_lst) for i in range(org_svec_len)]
    cl_lst = [(_tbl[_cl[0]], _tbl[_cl[1]]) for _cl in cl_lst]

    distMat = distance.cdist(svec, svec, metric='euclidean')
    for cl in cl_lst:
        distMat[cl[0], cl[1]] = args.clink_dis
        distMat[cl[1], cl[0]] = args.clink_dis
    
    clusterer = AgglomerativeClustering(
            n_clusters=cls_num,
            affinity='precomputed',
            linkage='average',
            distance_threshold=ahc_dis_th)
    clusterer.fit(distMat)

    if cls_num is not None:
        print("oracle n_clusters is known")
    else:
        print("oracle n_clusters is unknown")
        print("estimated n_clusters by constraind AHC: {}"
              .format(len(np.unique(clusterer.labels_))))
        cls_num = len(np.unique(clusterer.labels_))

    sil_lab = cls_num
    insert_sil_lab = [sil_lab for i in range(len(sil_lst))]
    insert_sil_lab_idx = [sil_lst[i] - i for i in range(len(sil_lst))]
    print("insert_sil_lab : {}".format(insert_sil_lab))
    print("insert_sil_lab_idx : {}".format(insert_sil_lab_idx))
    clslab = np.insert(clusterer.labels_,
                       insert_sil_lab_idx,
                       insert_sil_lab).reshape(-1, args.num_speakers)
    print("clslab : {}".format(clslab))

    return clslab, cls_nu


def merge_act_max(act, i, j):
    for k in range(len(act)):
        act[k, i] = max(act[k, i], act[k, j])
        act[k, j] = 0.0
    return act


def merge_acti_clslab(args, acti, clslab, cls_num):
    sil_lab = cls_num
    for i in range(len(clslab)):
        _lab = clslab[i].reshape(-1, 1)
        distM = distance.cdist(_lab, _lab, metric='euclidean').astype(np.int64)
        for j in range(len(distM)):
            distM[j][:j] = -1
        idx_lst = np.where(np.count_nonzero(distM == 0, axis=1) > 1)
        merge_done = []
        for j in idx_lst[0]:
            for k in (np.where(distM[j] == 0))[0]:
                if j != k and clslab[i, j] != sil_lab and k not in merge_done:
                    print("merge : (i, j, k) == ({}, {}, {})".format(i, j, k))
                    acti[i] = merge_act_max(acti[i], j, k)
                    clslab[i, k] = sil_lab
                    merge_done.append(j)

    return acti, clslab


def stitching(args, acti, clslab, cls_num):
    n_chunks = len(acti)
    s_loc = args.num_speakers
    sil_lab = cls_num
    s_tot = max(cls_num, s_loc-1)

    # Extend the max value of s_loc_idx to s_tot+1
    add_acti = []
    for chunk_idx in range(n_chunks):
        zeros = np.zeros((len(acti[chunk_idx]), s_tot+1))
        if s_tot+1 > s_loc:
            zeros[:, :-(s_tot+1-s_loc)] = acti[chunk_idx]
        else:
            zeros = acti[chunk_idx]
        add_acti.append(zeros)
    acti = np.array(add_acti)

    out_chunks = []
    for chunk_idx in range(n_chunks):
        # Make sloci2lab_dct.
        # key: s_loc_idx
        # value: estimated label by clustering or sil_lab
        cls_set = set()
        for s_loc_idx in range(s_tot+1):
            cls_set.add(s_loc_idx)

        sloci2lab_dct = {}
        for s_loc_idx in range(s_tot+1):
            if s_loc_idx < s_loc:
                sloci2lab_dct[s_loc_idx] = clslab[chunk_idx][s_loc_idx]
                if clslab[chunk_idx][s_loc_idx] in cls_set:
                    cls_set.remove(clslab[chunk_idx][s_loc_idx])
                else:
                    if clslab[chunk_idx][s_loc_idx] != sil_lab:
                        raise ValueError
            else:
                sloci2lab_dct[s_loc_idx] = list(cls_set)[s_loc_idx-s_loc]

        # Sort by label value
        sloci2lab_lst = sorted(sloci2lab_dct.items(), key=lambda x: x[1])

        # Select sil_lab_idx
        sil_lab_idx = None
        for idx_lab in sloci2lab_lst:
            if idx_lab[1] == sil_lab:
                sil_lab_idx = idx_lab[0]
                break
        if sil_lab_idx is None:
            raise ValueError

        # Get swap_idx
        # [idx of label(0), idx of label(1), ..., idx of label(s_tot)]
        swap_idx = [sil_lab_idx for j in range(s_tot+1)]
        for lab in range(s_tot+1):
            for idx_lab in sloci2lab_lst:
                if lab == idx_lab[1]:
                    swap_idx[lab] = idx_lab[0]

        print("swap_idx {}".format(swap_idx))
        swap_acti = acti[chunk_idx][:, swap_idx]
        swap_acti = np.delete(swap_acti, sil_lab, 1)
        out_chunks.append(swap_acti)

    return out_chunks


def prepare_model_for_eval(args):
    in_size = feature.get_input_dim(
            args.frame_size,
            args.context_size,
            args.input_transform)
    model_parameter_dict = torch.load(args.model_file)['model']
    model_all_n_speakers =\
        fix_state_dict(model_parameter_dict)["embed.weight"].shape[0]
    net = TransformerDiarization(
           n_speakers=args.num_speakers,
           in_size=in_size,
           n_units=args.hidden_size,
           n_heads=args.transformer_encoder_n_heads,
           n_layers=args.transformer_encoder_n_layers,
           dropout_rate=0,
           all_n_speakers=model_all_n_speakers,
           d=args.spkv_dim)

    device = [device_id for device_id in range(torch.cuda.device_count())]
    net.load_state_dict(fix_state_dict(model_parameter_dict))
    net.eval()
    net = net.to("cuda")
    # print('GPU device {} is used'.format(device))
    # print('Prepared model')
    logger.info('GPU device {} is used'.format(device))
    logger.info('Prepared model')

    return net


def prediction(args, net, kaldi_obj, recid):
    acti_lst = []
    svec_lst = []
    # Prepare input features
    data, rate = kaldi_obj.load_wav(recid)
    Y = feature.stft(data, args.frame_size, args.frame_shift)
    Y = feature.transform(Y, transform_type=args.input_transform)
    Y = feature.splice(Y, context_size=args.context_size)
    Y = Y[::args.subsampling]

    with torch.no_grad():
        for start, end in _gen_chunk_indices(len(Y), args.chunk_size):
            if start > 0 and start + args.chunk_size > end:
                # Ensure last chunk size
                Y_chunked = torch.from_numpy(Y[end-args.chunk_size:end])
            else:
                Y_chunked = torch.from_numpy(Y[start:end])
            Y_chunked = Y_chunked.to('cpu')

            outputs = net.batch_estimate(torch.unsqueeze(Y_chunked, 0))
            ys = outputs[0]

            for i in range(args.num_speakers):
                spkivecs = outputs[i+1]
                svec_lst.append(spkivecs[0].cpu().detach().numpy())

            if start > 0 and start + args.chunk_size > end:
                # Ensure last chunk size
                ys = list(ys)
                ys[0] = ys[0][args.chunk_size-(end-start):args.chunk_size]

            acti = ys[0].cpu().detach().numpy()
            acti_lst.append(acti)

    # acti_arr = np.array(acti_lst)
    # svec_arr = np.array(svec_lst)
    logger.info("Prediction Done!")
    return acti_lst, svec_lst




def infertest(args):
    # Prepare model
    net = prepare_model_for_eval(args)

    kaldi_obj = kaldi_data.KaldiData_infer(args.data_dir)
    for recid in kaldi_obj.wavs:
        print("recid : {}".format(recid))
        # prediction
        acti, svec = prediction(args, net, kaldi_obj, recid)
        n_chunks = len(acti)
        # initialize clustering setting
        cls_num = None
        ahc_dis_th = args.ahc_dis_th
        # Get cannot-link index list and silence index list
        cl_lst, sil_lst = get_cl_sil(args, acti, cls_num)

        n_samples = n_chunks * args.num_speakers - len(sil_lst)
        min_n_samples = 2
        if cls_num is not None:
            min_n_samples = cls_num

        if n_samples >= min_n_samples:
            # clustering (if cls_num is None, update cls_num)
            clslab, cls_num =\
                 clustering(args, svec, cls_num, ahc_dis_th, cl_lst, sil_lst)
            # merge
            acti, clslab = merge_acti_clslab(args, acti, clslab, cls_num)
            # stitching
            out_chunks = stitching(args, acti, clslab, cls_num)
        else:
            out_chunks = acti

        outdata = np.vstack(out_chunks)
        # Saving the resuts
        outfname = recid + '.h5'
        outpath = os.path.join(args.out_dir, outfname)
        with h5py.File(outpath, 'w') as wf:
            # 'T_hat': key
            wf.create_dataset('T_hat', data=outdata)


def make_rttm(args, data, session, rttm_file):
    with open(rttm_file, 'a') as wf:
        a = np.where(data > 0.5, 1, 0)
        a = medfilt(a, (11,1))
        for spkid, frames in enumerate(a.T):
            frames = np.pad(frames, (1,1), 'constant')
            changes, = np.where(np.diff(frames, axis=0) != 0)
            fmt = "SPEAKER {:s} 1 {:7.3f} {:7.3f} <NA> <NA> {:s} <NA>"
            for s, e in zip(changes[::2], changes[1::2]):
                print(fmt.format(
                    session,
                    s * float(args.frame_shift) * float(args.subsampling) / float(args.sampling_rate),
                    (e - s) * float(args.frame_shift) * float(args.subsampling) / float(args.sampling_rate),
                    session + "_" + str(spkid)), file=wf)
