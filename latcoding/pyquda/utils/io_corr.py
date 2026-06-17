from pathlib import Path

import h5py
import numpy as np
import re


# -----------------------------------------------------------------------------
# Shared tag/path helpers
# -----------------------------------------------------------------------------

# Build a compact sample identifier for log files.
def get_sample_log_tag(ama, src, sm):

    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    log_sample = ama_tag + "_" + src_tag + "_" + sm_tag

    return log_sample


# Build the standard point-source two-point output tag.
def get_c2pt_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".c2pt"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/c2pt/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


# Build the standard qTMD output tag used by proton and pion qTMD applications.
def get_qTMD_file_tag(data_dir, lat, cfg, ama,src, sm):
    
    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMD"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMD/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


# Build the standard disconnected qTMD one-point output tag.
def get_disconnected_qTMD_1pt_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMD1pt"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMD1pt/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


# Build the standard qTMDWF output tag.
def get_qTMDWF_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMDWF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMDWF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


# -----------------------------------------------------------------------------
# Pion EMFF tag helpers
# -----------------------------------------------------------------------------

# Build the pion electromagnetic form-factor output tag.
def get_pion_EMFF_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".pion_EMFF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/pion_EMFF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


# -----------------------------------------------------------------------------
# Pion soft-factor tag helpers
# -----------------------------------------------------------------------------

# Build the pion soft-factor four-point output tag.
def get_pion_soft_factor_file_tag(data_dir, lat, cfg, ama, src, sm, quarkmom1, quarkmom2):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".pion_soft_factor"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    mom1_tag = "qx"+str(quarkmom1[0]) + "qy"+str(quarkmom1[1]) + "qz"+str(quarkmom1[2])
    mom2_tag = "qx"+str(quarkmom2[0]) + "qy"+str(quarkmom2[1]) + "qz"+str(quarkmom2[2])
    sm_tag  = str(sm)

    return data_dir + "/pion_soft_factor/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag + ".fw_" + mom1_tag + ".bw_" + mom2_tag


# Build the pion soft-factor qTMDWF diagnostic output tag.
def get_pion_soft_factor_qTMDWF_file_tag(data_dir, lat, cfg, ama, src, sm, quarkmom1, quarkmom2):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".pion_soft_factor_qTMDWF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    mom1_tag = "qx"+str(quarkmom1[0]) + "qy"+str(quarkmom1[1]) + "qz"+str(quarkmom1[2])
    mom2_tag = "qx"+str(quarkmom2[0]) + "qy"+str(quarkmom2[1]) + "qz"+str(quarkmom2[2])
    sm_tag  = str(sm)

    return data_dir + "/pion_soft_factor_qTMDWF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag + ".fw_" + mom1_tag + ".bw_" + mom2_tag


# Build the pion soft-factor wall-source two-point diagnostic output tag.
def get_pion_soft_factor_c2pt_file_tag(data_dir, lat, cfg, ama, src, sm, quarkmom1, quarkmom2):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".pion_soft_factor_c2pt"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    mom1_tag = "qx"+str(quarkmom1[0]) + "qy"+str(quarkmom1[1]) + "qz"+str(quarkmom1[2])
    mom2_tag = "qx"+str(quarkmom2[0]) + "qy"+str(quarkmom2[1]) + "qz"+str(quarkmom2[2])
    sm_tag  = str(sm)

    return data_dir + "/pion_soft_factor_c2pt/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag + ".fw_" + mom1_tag + ".bw_" + mom2_tag


# Build the saved wall-source propagator tag for the pion soft-factor workflow.
def get_pion_soft_factor_prop_file_tag(data_dir, lat, cfg, ama, src, sm, quarkmom):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".pion_soft_factor_prop"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    mom_tag = "qx"+str(quarkmom[0]) + "qy"+str(quarkmom[1]) + "qz"+str(quarkmom[2])
    sm_tag  = str(sm)

    return data_dir + "/pion_soft_factor_prop/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag + "." + mom_tag


# Ensure the parent directory exists before opening an HDF5 file.
def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# EMT file-name helpers
# -----------------------------------------------------------------------------

# Build the canonical EMT source-position tag.
def _emt_site_tag(src):
    return "x" + str(src[0]) + "y" + str(src[1]) + "z" + str(src[2]) + "t" + str(src[3])


# Build the gluon EMT one-point output tag.
def get_emt_gluon_1pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMTg" / (str(lat) + ".EMTg." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


# Build the quark EMT one-point output tag.
def get_emt_quark_1pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMTc" / (str(lat) + ".EMTc." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


# Build the pion/meson quark EMT three-point output tag.
def get_emt_quark_3pt_file_tag(data_dir, lat, cfg, ama, src, sm, spin):
    return str(Path(data_dir) / "EMT3pt" / (str(lat) + ".EMT3pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm) + ".spin" + str(spin)))


# Build the pion/meson EMT two-point diagnostic output tag.
def get_emt_meson_2pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMT2pt" / (str(lat) + ".EMT2pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


# Build the proton EMT two-point diagnostic output tag.
def get_emt_proton_2pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMTproton2pt" / (str(lat) + ".EMTproton2pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


# Build the proton quark EMT three-point output tag.
def get_emt_proton_quark_3pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMTproton3pt" / (str(lat) + ".EMTproton3pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


# -----------------------------------------------------------------------------
# Shared HDF5 helpers
# -----------------------------------------------------------------------------


# Attach optional metadata to an HDF5 file or group.
def _write_h5_attrs(obj, attrs):
    if not attrs:
        return
    for key, value in attrs.items():
        if value is None:
            continue
        obj.attrs[key] = value


# Open a fresh HDF5 file after creating its parent directory.
def _prepare_h5_file(path, attrs=None):
    ensure_parent_dir(path)
    f = h5py.File(path, "w")
    _write_h5_attrs(f, attrs)
    return f


# -----------------------------------------------------------------------------
# EMT HDF5 writers
# -----------------------------------------------------------------------------

# Save flowed quark EMT one-point data and ringed-fermion CHI building blocks.
def save_emt_quark_1pt_hdf5(tag, Tmunu_pervec, CHI_pervec, Tmunu, CHI, attrs=None, source_bookkeeping=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        raw = f.require_group("raw")
        raw.create_dataset("Tmunu_pervec", data=Tmunu_pervec)
        raw.create_dataset("CHI_pervec", data=CHI_pervec)
        if source_bookkeeping is not None:
            for name, values in source_bookkeeping.items():
                raw.create_dataset(name, data=np.asarray(values, dtype=np.int32))

        avg = f.require_group("avg")
        avg.create_dataset("CHI", data=CHI)
        g_t = avg.require_group("Tmunu")
        g_t.attrs["upper_triangle_only"] = True
        for mu in range(4):
            for nu in range(mu, 4):
                g_t.create_dataset(f"T{mu+1}{nu+1}", data=Tmunu[mu, nu])


# Save quark EMT three-point functions together with the matching two-point data.
def save_emt_quark_3pt_hdf5(tag, C2, C3_chi, C3_Tmunu, momentum_transfer_list=None, attrs=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        f.create_dataset("C2", data=C2)
        f.create_dataset("C3_chi", data=C3_chi)
        f.create_dataset("C3_Tmunu", data=C3_Tmunu)
        if momentum_transfer_list is not None:
            f.create_dataset("momentum_transfer_list", data=np.asarray(momentum_transfer_list, dtype=np.int32))


# Save pion/meson EMT two-point functions and their gamma/momentum metadata.
def save_emt_meson_2pt_hdf5(tag, C2, gamma_list, momentum_list, attrs=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        f.create_dataset("C2", data=C2)
        f.create_dataset("gamma_list", data=np.asarray(gamma_list, dtype="S"))
        f.create_dataset("momentum_list", data=np.asarray(momentum_list, dtype=np.int32))


# Save flowed gluon EMT one-point data.
def save_emt_gluon_1pt_hdf5(tag, Tmunu_t, attrs=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        g_t = f.require_group("Tmunu")
        g_t.attrs["upper_triangle_only"] = True
        for mu in range(4):
            for nu in range(mu, 4):
                g_t.create_dataset(f"T{mu+1}{nu+1}", data=Tmunu_t[mu, nu])


# -----------------------------------------------------------------------------
# Two-point and qTMD HDF5 writers
# -----------------------------------------------------------------------------

# Save the standard two-point function with source-time rolling.
def save_c2pt_hdf5(corr, tag, gammalist, plist):

    src_match = None
    for part in tag.split("."):
        src_match = re.search(r"^x-?\d+y-?\d+z-?\d+t(-?\d+)$", part)
        if src_match is not None:
            break
    if src_match is None:
        raise ValueError(f"Could not parse source time from c2pt tag: {tag}")
    roll = -int(src_match.group(1))

    save_h5 = tag + ".h5"
    f = _prepare_h5_file(save_h5)
    sm = f.create_group("SS")
    for ig, gm in enumerate(gammalist):
        g = sm.create_group(gm)
        for ip, p in enumerate(plist):
            dataset_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            #print('DEBUG:', np.shape(corr), np.shape(gammalist), ig, ip)
            g.create_dataset(dataset_tag, data=np.roll(corr[ig][ip], roll, axis=0))
    f.close()


# W_index_list[bT, bz, eta, Tdir]
# Save proton qTMD/PDF three-point data after the application has already rolled time.
def save_qTMD_proton_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list, tsep, latt_info, attrs=None):

    bT_list = ['b_X', 'b_Y']

    #g.message("-->>",W_index_list)

    save_h5 = tag + ".h5"
    f = _prepare_h5_file(save_h5, attrs)

    if latt_info.mpi_rank == 0:
        print(f"no roll")
        print(f"corr.shape, {np.shape(corr)}")
        print(f"plist.shape, {np.shape(plist)}")
    sm = f.require_group("SS")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                g_data.create_dataset('bz'+str(idx[1]), data=corr[i][ip][ig][:tsep+2])
    f.close()


# Save pion qTMD/PDF data using the same HDF5 layout as the proton writer.
def save_qTMD_pion_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list, tsep, latt_info, attrs=None):
    save_qTMD_proton_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list, tsep, latt_info, attrs=attrs)


# Save disconnected qTMD/PDF one-point loops.
def save_disconnected_qTMD_1pt_hdf5(tag, loop_pervec, loop_avg, gammalist, plist, W_index_list, attrs=None, source_bookkeeping=None):
    save_h5 = tag + ".h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        raw = f.require_group("raw")
        raw.create_dataset("loop_pervec", data=loop_pervec)
        if source_bookkeeping is not None:
            for name, values in source_bookkeeping.items():
                raw.create_dataset(name, data=np.asarray(values, dtype=np.int32))

        f.create_dataset("gamma_list", data=np.asarray(gammalist, dtype="S"))
        f.create_dataset("momentum_list", data=np.asarray(plist, dtype=np.int32))
        f.create_dataset("W_index_list", data=np.asarray(W_index_list, dtype=np.int32))

        bT_list = ["b_X", "b_Y"]
        sm = f.require_group("avg").require_group("SS")
        for ig, gm in enumerate(gammalist):
            g_gm = sm.require_group(gm)
            for ip, p in enumerate(plist):
                p_tag = "PX" + str(p[0]) + "PY" + str(p[1]) + "PZ" + str(p[2])
                g_p = g_gm.require_group(p_tag)
                for i, idx in enumerate(W_index_list):
                    path = bT_list[idx[3]] + "/" + "eta" + str(idx[2]) + "/" + "bT" + str(idx[0])
                    g_data = g_p.require_group(path)
                    g_data.create_dataset("bz" + str(idx[1]), data=loop_avg[i, ig, ip])


# -----------------------------------------------------------------------------
# Pion EMFF HDF5 writers
# -----------------------------------------------------------------------------

# Save pion electromagnetic form-factor three-point data.
def save_pion_EMFF_hdf5_noRoll(corr, tag, gammalist, qlist, tsep, latt_info):

    save_h5 = tag + ".h5"
    f = _prepare_h5_file(save_h5)

    if latt_info.mpi_rank == 0:
        print(f"no roll")
        print(f"corr.shape, {np.shape(corr)}")
        print(f"qlist.shape, {np.shape(qlist)}")
    sm = f.require_group("SS")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for iq, q in enumerate(qlist):
            q_tag = "PX"+str(q[0])+"PY"+str(q[1])+"PZ"+str(q[2])
            g_gm.create_dataset(q_tag, data=corr[iq][ig][:tsep+2])
    f.close()


# -----------------------------------------------------------------------------
# Pion soft-factor HDF5 writers
# -----------------------------------------------------------------------------

# Save the pion soft-factor four-point correlator.
def save_pion_soft_factor_hdf5_noRoll(corr, tag, pion_src_keys, pion_sink_keys, gamma1_keys, gamma2_keys, bT_dir, bT_length, tseplist, latt_info):
    save_h5 = tag + ".h5"
    f = _prepare_h5_file(save_h5)

    bT_list = ["bX", "bY", "bZ"]
    if latt_info.mpi_rank == 0:
        print(f"no roll")
        print(f"corr.shape, {np.shape(corr)}")
    for i, src_key in enumerate(pion_src_keys):
        sink_key = pion_sink_keys[i]
        g_src = f.require_group(f"src{src_key}_sink{sink_key}")
        for j, gamma1_key in enumerate(gamma1_keys):
            gamma2_key = gamma2_keys[j]
            g_gm = g_src.require_group(f"{gamma1_key}_{gamma2_key}")
            for k, direction in enumerate(bT_dir):
                for bT in range(bT_length + 1):
                    g_bT = g_gm.require_group(bT_list[direction] + "_" + str(bT))
                    for its, tsep in enumerate(tseplist):
                        g_bT.create_dataset("ts" + str(tsep), data=corr[its, i, j, k, bT])
    f.close()


# Save the wall-source qTMDWF diagnostic used by the pion soft-factor workflow.
def save_pion_soft_factor_qTMDWF_hdf5_noRoll(corr, tag, src_key, momentum, bT_dir, bT_length, bz_length, latt_info):
    save_h5 = tag + ".h5"
    f = _prepare_h5_file(save_h5)

    bT_list = ["b_X", "b_Y", "b_Z"]
    if latt_info.mpi_rank == 0:
        print(f"no roll")
        print(f"corr.shape, {np.shape(corr)}")
    sm = f.require_group("SP")
    g_src = sm.require_group(str(src_key))
    p_tag = "PX"+str(momentum[0])+"PY"+str(momentum[1])+"PZ"+str(momentum[2])
    g_p = g_src.require_group(p_tag)
    idx = 0
    for direction in bT_dir:
        g_T = g_p.require_group(bT_list[direction])
        for bT in range(bT_length + 1):
            g_bT = g_T.require_group("bT" + str(bT))
            for bz in range(bz_length + 1):
                g_bT.create_dataset("bz" + str(bz), data=corr[idx])
                idx += 1
    f.close()


# Save the wall-to-wall two-point diagnostic used by the pion soft-factor workflow.
def save_pion_soft_factor_c2pt_hdf5_noRoll(corr, tag, src_key, sink_keys, momentum, latt_info):
    save_h5 = tag + ".h5"
    f = _prepare_h5_file(save_h5)

    if latt_info.mpi_rank == 0:
        print(f"no roll")
        print(f"corr.shape, {np.shape(corr)}")
    sm = f.require_group("SS")
    g_src = sm.require_group(str(src_key))
    p_tag = "PX"+str(momentum[0])+"PY"+str(momentum[1])+"PZ"+str(momentum[2])
    for isink, sink_key in enumerate(sink_keys):
        g_sink = g_src.require_group(str(sink_key))
        g_sink.create_dataset(p_tag, data=corr[isink])
    f.close()


# -----------------------------------------------------------------------------
# qTMDWF HDF5 writers
# -----------------------------------------------------------------------------

# Save the standard qTMDWF output after the application has already rolled time.
def save_qTMDWF_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list):

    bT_list = ['b_X', 'b_Y']

    save_h5 = tag + ".h5"
    f = _prepare_h5_file(save_h5)

    sm = f.require_group("SP")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                g_data.create_dataset('bz'+str(idx[1]), data=corr[i][ip][ig])
    f.close()
