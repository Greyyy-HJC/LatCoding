import gpt as g
import os
import numpy as np
import h5py
from proton_qPDF_class import proton_qPDF

"""
================================================================================
                        Collinear PDF Calculation
================================================================================
This script calculates collinear parton distribution functions (PDFs) for the proton.

The key concept is the z separation: PDFs are calculated using matrix elements
of the form <p'|O(z)|p> where O(z) is a non-local operator with a Wilson line
of length z in the z-direction.

For collinear PDFs, we compute:
- Wilson lines only in the z-direction (no transverse separation)
- Different z separations from 0 to z_max
- Matrix elements for both up and down quark flavors
- Different nucleon polarizations (PpSzp, PpSzm)
- Multiple time separations (tsep values)
- Multiple hadron momenta

The z separation corresponds to the spatial extent of the non-local operator
and is crucial for extracting the momentum dependence of PDFs.

Output:
- HDF5 files saved in ./output/ directory
- Structure: 2pt|3pt/momentum/[tsep/z_separation/flavor/polarization]/gamma_structure/real_imag
================================================================================
"""

# Import gamma structures from the PDF class
from proton_qPDF_class import my_gammas

def save_correlator_hdf5(results_dict, filename):
    """Save correlation data to HDF5 file with hierarchical structure
    
    Args:
        results_dict: Dictionary with structure:
            {
                "2pt": {momentum: {gamma: {real/imag: array[N_conf, Lt]}}},
                "3pt": {momentum: {tsep: {z_sep: {flavor: {pol: {gamma: {real/imag: array[N_conf, Lt]}}}}}}}
            }
        filename: Base filename for output
    """
    os.makedirs("output", exist_ok=True)
    filepath = f"output/{filename}.h5"
    
    with h5py.File(filepath, 'w') as f:
        # Create 2pt group and save data
        twopoint_group = f.create_group("2pt")
        for momentum_label, momentum_data in results_dict["2pt"].items():
            momentum_group = twopoint_group.create_group(momentum_label)
            for gamma_name, gamma_data in momentum_data.items():
                gamma_group = momentum_group.create_group(gamma_name)
                gamma_group.create_dataset("real", data=gamma_data["real"])
                gamma_group.create_dataset("imag", data=gamma_data["imag"])
        
        # Create 3pt group and save data
        threepoint_group = f.create_group("3pt")
        for momentum_label, momentum_data in results_dict["3pt"].items():
            momentum_group = threepoint_group.create_group(momentum_label)
            
            for tsep, tsep_data in momentum_data.items():
                tsep_group = momentum_group.create_group(f"tsep_{tsep}")
                
                for z_sep, z_data in tsep_data.items():
                    z_group = tsep_group.create_group(f"z_{z_sep}")
                    
                    for flavor, flavor_data in z_data.items():
                        flavor_group = z_group.create_group(flavor)
                        
                        for pol, pol_data in flavor_data.items():
                            pol_group = flavor_group.create_group(pol)
                            
                            for gamma_name, gamma_data in pol_data.items():
                                gamma_group = pol_group.create_group(gamma_name)
                                gamma_group.create_dataset("real", data=gamma_data["real"])
                                gamma_group.create_dataset("imag", data=gamma_data["imag"])
    
    g.message(f"Saved correlator data to {filepath}")

def collect_gamma_data(corr_data):
    """Process correlator data into gamma-separated structure"""
    result = {}
    
    # corr_data has shape: (gamma_list, tsep)
    for gamma_idx, gamma_name in enumerate(my_gammas):
        gamma_data = corr_data[gamma_idx]
        
        result[f"gamma_{gamma_name}"] = {
            "real": np.array([val.real for val in gamma_data]),
            "imag": np.array([val.imag for val in gamma_data])
        }
    
    return result

def process_quark_correlator(results, momentum_label, tsep, current_z_sep, pol_name, quark_type, corr_data, conf_n, Lt):
    """Helper function to process correlator data for a specific quark type"""
    if pol_name not in results["3pt"][momentum_label][tsep][current_z_sep][quark_type]:
        results["3pt"][momentum_label][tsep][current_z_sep][quark_type][pol_name] = {}
    
    quark_data = collect_gamma_data(corr_data[0])  # shape: (gamma_list, tsep)
    for gamma_name, gamma_data in quark_data.items():
        if gamma_name not in results["3pt"][momentum_label][tsep][current_z_sep][quark_type][pol_name]:
            results["3pt"][momentum_label][tsep][current_z_sep][quark_type][pol_name][gamma_name] = {
                "real": np.zeros((N_conf, Lt)),
                "imag": np.zeros((N_conf, Lt))
            }
        for component in ["real", "imag"]:
            results["3pt"][momentum_label][tsep][current_z_sep][quark_type][pol_name][gamma_name][component][conf_n] = gamma_data[component]


# configuration tag
lat_tag = "S8T32"
Ls = 8
Lt = 32

# Number of configurations
N_conf = 30  # Adjust this value as needed

# parameters for PDF calculation
parameters = {
    "boost_in": [0, 0, 0],
    "boost_out": [0, 0, 0],
    "width": 2.0,  # width of Gaussian source
    "pol": ["PpSzp"],  # polarization of the proton
    "tsep_list": [8],  # time separation list for three point function
    "z_max": 0,  # maximum z separation for PDF calculation (in lattice units)
}
hadron_momenta = [[0, 0, 0, 0]]

# Initialize results dictionary
results = {"2pt": {}, "3pt": {}}

conf_path = f"../../conf/{lat_tag}_cg/gauge"
vtrans_path = f"../../conf/{lat_tag}_cg/Vtrans"
precision = 1e-08

# Loop over configurations
for conf_n in range(N_conf):
    g.message(f"Processing configuration {conf_n}")
    
    # load gauge configuration
    g.message("Loading gauge configuration")
    grid = g.grid([Ls, Ls, Ls, Lt], g.double)
    rng = g.random(f"seed text")
    U_cg = g.convert(g.load(f"{conf_path}/wilson_b6.cg.{precision}.{conf_n}"), g.double)
    trafo = g.convert(g.load(f"{vtrans_path}/V_trans.{precision}.{conf_n}"), g.double)  
    # U = g.qcd.gauge.random(grid, rng)
    # U_cg, trafo = g.gauge_fix(U, maxiter=5000)
    g.message("Finished loading gauge config")

    # prepare inverter
    p = {
        "kappa": 0.1255997387525434, # 0.12623 for 300 MeV pion; 0.1256 for 670 MeV pion
        "csw_r": 1.0336,
        "csw_t": 1.0336,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1, 1, 1, -1],
    }
    w = g.qcd.fermion.wilson_clover(U_cg, p)
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.cg({"eps": 1e-8, "maxiter": 10000})
    prop_exact = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

    # Loop over hadron momenta
    for hadron_momentum in hadron_momenta:
        parameters["hadron_momentum"] = hadron_momentum
        PDF_Measurement = proton_qPDF(parameters)
        PDF_Measurement.print_calculation_info()

        # prepare wilson line indices for PDF
        W_index_list_PDF = PDF_Measurement.create_PDF_Wilsonline_index_list()
        g.message("W_index_list_PDF:")
        g.message(W_index_list_PDF)
        
        # create the source and the propagator
        pos = [0, 0, 0, 0]  # position of the source
        srcDp = PDF_Measurement.create_src_2pt(pos, trafo, U_cg[0].grid)
        prop_exact_f = g.eval(prop_exact * srcDp)
        
        momentum_label = PDF_Measurement.get_momentum_label(hadron_momentum)
        g.message(f"Processing hadron momentum = {hadron_momentum} (label: {momentum_label})")
        
        # Initialize momentum results if not exists
        if momentum_label not in results["2pt"]:
            results["2pt"][momentum_label] = {}
        if momentum_label not in results["3pt"]:
            results["3pt"][momentum_label] = {}
        
        #! 2pt calculation
        g.message("Contraction: Starting 2pt (includes sink smearing) for PDF")
        phases_2pt = PDF_Measurement.make_mom_phases_2pt(U_cg[0].grid, pos)
        corr_2pt = PDF_Measurement.contract_2pt_PDF(prop_exact_f, phases_2pt, trafo) # shape: (polarizations, 1, gamma_list, tsep), because calculate 1 momentum at a time
                
        corr_2pt_collect_gamma = collect_gamma_data(corr_2pt[0][0])
        
        for gamma_name, gamma_data in corr_2pt_collect_gamma.items():
            if gamma_name not in results["2pt"][momentum_label]:
                results["2pt"][momentum_label][gamma_name] = {
                    "real": np.zeros((N_conf, Lt)),
                    "imag": np.zeros((N_conf, Lt))
                }
            results["2pt"][momentum_label][gamma_name]["real"][conf_n] = gamma_data["real"]
            results["2pt"][momentum_label][gamma_name]["imag"][conf_n] = gamma_data["imag"]
        
        g.message("Contraction: Done 2pt (includes sink smearing) for PDF")
        
        #! 3pt calculation
        # Initialize 3pt results for this momentum and configuration
        if conf_n == 0:
            for tsep in PDF_Measurement.tsep_list:
                results["3pt"][momentum_label][tsep] = {}
                
        # Loop over time separations
        for tsep in PDF_Measurement.tsep_list:
            g.message(f"Processing time separation tsep = {tsep}")
            
            # Create sequential propagators
            sequential_bw_prop_down = PDF_Measurement.create_bw_seq(
                prop_exact, prop_exact_f, trafo, 2, tsep, hadron_momentum, pos
            )
            sequential_bw_prop_up = PDF_Measurement.create_bw_seq(
                prop_exact, prop_exact_f, trafo, 1, tsep, hadron_momentum, pos
            )
            
            # Loop over Wilson line indices
            for WL_indices in W_index_list_PDF:
                current_z_sep = WL_indices[1]
                g.message(f"Processing Wilson line with z separation = {current_z_sep}, index: {WL_indices}")
                
                # Initialize z_sep results if not exists
                if current_z_sep not in results["3pt"][momentum_label][tsep]:
                    results["3pt"][momentum_label][tsep][current_z_sep] = {
                        "down": {},
                        "up": {}
                    }
                
                # Create Wilson line and forward prop
                W = PDF_Measurement.create_PDF_Wilsonline(U_cg, WL_indices)
                pdf_forward_prop = PDF_Measurement.create_fw_prop_PDF(
                    prop_exact_f, [W], [WL_indices]
                )

                # Zero momentum phases for now
                phases = [g.identity(g.complex(U_cg[0].grid))]

                # Calculate correlators
                corr_down = PDF_Measurement.contract_PDF(
                    pdf_forward_prop, sequential_bw_prop_down, phases
                ) # shape: (polarizations, 1, gamma_list, tsep), because calculate 1 momentum at a time
                corr_up = PDF_Measurement.contract_PDF(
                    pdf_forward_prop, sequential_bw_prop_up, phases
                )
                
                # Process correlator data for each polarization
                for pol_index, pol_name in enumerate(PDF_Measurement.pol_list):
                    # Process both quark types
                    for quark_type, corr_data in [("down", corr_down), ("up", corr_up)]:
                        process_quark_correlator(
                            results, momentum_label, tsep, current_z_sep,
                            pol_name, quark_type, corr_data[pol_index], conf_n, Lt
                        )

                g.message(f"Processed correlators for tsep = {tsep}, momentum = {momentum_label}, z separation = {current_z_sep}")

# Save all results to HDF5 file
save_correlator_hdf5(results, f"{lat_tag}_Cg5_proton_PDF")

g.message("PDF calculation completed!")
g.message(f"Output file saved as ./output/{lat_tag}_Cg5_proton_PDF.h5")
g.message(f"Total configurations: {N_conf}")
g.message(f"Total z separations calculated: {PDF_Measurement.get_z_separations()}")
g.message(f"Time separations: {PDF_Measurement.tsep_list}")
g.message(f"Hadron momenta: {hadron_momenta}")
g.message(f"Polarizations: {PDF_Measurement.pol_list}")
g.message("HDF5 structure: 2pt|3pt/momentum/[tsep/z_separation/flavor/polarization]/gamma_structure/real_imag")





