import numpy as np
import glob
import os
import argparse
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
# use vectorized versions of special functions
from scipy.special import erf
from numpy import pi
from scipy.optimize import least_squares

def ubg_model(q, bkg, s, B1, pack, mesh, delt, kI):
    # Enforce positivity (Igor uses abs())
    bkg  = abs(bkg)
    s    = abs(s)
    B1   = abs(B1)
    pack = abs(pack)
    mesh = abs(mesh)
    delt = abs(delt)
    kI   = abs(kI)

    # Parameterization: fit s in [0,1] such that Rg1 = 0.5 * mesh * s    
    s = np.asarray(s)
    # ensures s in [0,1]
    try:
        s = np.clip(np.abs(s), 0.0, 1.0)
    except Exception:
        s = float(min(max(abs(s), 0.0), 1.0))

    Rg1 = 0.5 * mesh * s

    Rad = np.sqrt(pack * mesh * Rg1 / 4)

    # No-correlation intensity
    G1 = B1 * Rg1**4 * Rad / (2*Rg1 + Rad)
    B2 = 2 * G1 / (Rg1**2)

    qstar1 = q / (erf(q * np.abs(Rg1) / np.sqrt(6))**3)

    Rg2 = np.sqrt(Rad**2 / 2 + Rg1**2 / 3)
    qstar2 = q / (erf(q * np.abs(Rg2) / np.sqrt(6))**3)

    G2 = G1 * (Rad / Rg1)**2

    I = (
        G1 * np.exp(-q*q*Rg1*Rg1/3)
        + B1 * qstar1**(-4)
        + G2 * np.exp(-q*q*Rg2*Rg2/3)
        + np.exp(-q*q*Rg1*Rg1/3) * B2 * qstar2**(-2)
    )

    # Apply correlation modification to q
    q_corr = q * np.exp(delt * (q - 2*pi/mesh) / q)

    # Structure factor S(q)
    if pack != 0:
        corrnum = q_corr * mesh
        gNoIters = 20
        DelAngle = pi / (2 * gNoIters)

        sumtheta = 0
        for i in range(gNoIters):
            angle = i * DelAngle
            if np.cos(angle) == 0:
                theta = 1
            else:
                theta = np.sin(corrnum * np.cos(angle)) / (corrnum * np.cos(angle))
            sumtheta += theta * pi/2

        S_q = 1 + pack * (sumtheta / gNoIters) * np.exp(-(corrnum**2) * kI)
        I /= S_q

    return I + bkg

def load_dat_file(path):
    data = np.loadtxt(path, skiprows=1)
    q = data[:, 0]
    I = data[:, 1]
    return q, I

def fit_B1_highq(q, I, q_low=0.075, q_high_cap=0.15):
    """estimate B1 from high-q region by enforcing a q^-4 power law

    choose q range q >= q_low and q <= q_high_cap (if available);
    perform a linear fit in log-space with fixed slope -4:
        ln I = ln B1 - 4 ln q
    Returns B1 and estimated 1-sigma uncertainty
    """
    q = np.asarray(q)
    I = np.asarray(I)

    # Choose high-q window
    if q.max() >= q_high_cap:
        mask = (q >= q_low) & (q <= q_high_cap)
    else:
        mask = q >= q_low

    if not np.any(mask):
        # Fallback: use top 10% of q values
        idx = int(len(q) * 0.9)
        mask = np.arange(len(q)) >= idx

    qh = q[mask]
    Ih = I[mask]

    # Remove non-positive intensities
    valid = (Ih > 0) & np.isfinite(Ih) & np.isfinite(qh)
    qh = qh[valid]
    Ih = Ih[valid]

    if len(qh) < 3:
        raise RuntimeError("Not enough high-q points to estimate B1")

    y = np.log(Ih)
    x = np.log(qh)

    # log-linear fit for B1 (fixed slope -4), direct mean for bkg
    # ln I = ln B1 - 4 ln q + bkg
    # Estimate bkg as mean(I - B1*q^-4) after fitting B1
    y = np.log(Ih)
    x = np.log(qh)
    lnB1_vals = y + 4 * x
    lnB1 = np.mean(lnB1_vals)
    B1 = float(np.exp(lnB1))
    se_lnB1 = np.std(lnB1_vals, ddof=1) / np.sqrt(len(lnB1_vals))
    perr_B1 = B1 * se_lnB1
    # Estimate bkg as mean residual in high-q
    n=5
    last_points = Ih[-n:]
    bkg = 0.01*(np.mean(Ih - B1 * qh**(-4)))
    perr_bkg = np.std(Ih - B1 * qh**(-4)) / np.sqrt(len(Ih))
    return B1, perr_B1, bkg, perr_bkg

def fit_with_penalties(q, I, curvewt = 0.5, kI_floor=0.05, delt_floor=0.005, q_high_low=0.075, q_high_cap=0.15, q_low=0.01, q_high=0.11):
    """fit with least sqares plus penalties with a 2nd derivative for smoothness and penalties ok kI and delt to prevent unphysical values. Returns results dict.
    """
    
    results = {}
    # Step 1: fit B1 and bkg in high-q window
    try:
        B1_est, B1_err, bkg_est, bkg_err = fit_B1_highq(q, I, q_low=0.075, q_high_cap=0.15)
    except Exception as e:
        raise RuntimeError(f"Failed to estimate B1/bkg from high-q: {e}")

    results['B1'] = B1_est
    results['B1_err'] = B1_err
    results['bkg'] = bkg_est
    results['bkg_err'] = bkg_err
    qm = np.asarray(q)
    Im = np.asarray(I)
    mask_low = (qm >= 0.0145) & (qm < 0.11)
    #if not np.any(mask_low):# fallback to 0.016 if no points above 0.01
    #    mask_low = (qm >= 0.001) & (qm < 0.075) # if still no points, just use all q values below 0.075
    qm = qm[mask_low]
    Im = Im[mask_low]
    if len(qm) < 5:# if still too few points, just use all q values below 0.075
        raise RuntimeError("Not enough points in low-q range for fitting remaining parameters")

    def ubg_resid(qvals, s, pack, mesh, delt, kI):
        return ubg_model(qvals,
                         bkg_est,
                         s,
                         B1_est,
                         pack,
                         mesh,
                         delt,
                         kI) - Im
    p0 = [0.60, 200, 170, 1e-89, 0.08]
    lower = [0.50, 100, 50, 1e-90, 0.05]
    upper = [0.95, 400, 300, 1e-9, 1]
    param_names = ['s', 'pack', 'mesh', 'delt', 'kI']

    def residuals(params):
        s, pack, mesh, delt, kI = params
        model = ubg_model(qm,
                         bkg_est,
                         s,
                         B1_est,
                         pack,
                         mesh,
                         delt,
                         kI)
        resid = (Im - model) / (Im + 1e-12)  # Add small value to avoid division by zero
        second_diff= np.diff(model, n=2)
        curvpenalty = curvewt * second_diff/(np.mean(model)+1e-12) # curvature penalty normalized by mean intensity to be scale invariant
        kI_penalty = np.exp(-kI / kI_floor) # exponential penalty to keep kI above floor
        delt_penalty = 0.1*np.exp(((delt - 0.003) / 0.001) ** 2) # exponential penalty to keep delt between 0.009 and 1
        mesh_penalty = 0.1*np.exp(((mesh - 205) / 50) ** 2) # exponential penalty to keep mesh near 155
        # Strong quadratic penalty to keep s near 0.8, strongly discouraging s from reaching upper bound
        s_penalty = np.exp(((s - 0.75) / 0.25) ** 2)

        #gaussian penalties to keep parameters within bounds (with width of 15% of the allowed range)
        penalties = []
        for i, (param, low, high) in enumerate(zip(params, lower, upper)):
            width = 0.15 * (high - low)
            center = (high + low) / 2
            penalty = np.exp(-0.5 * ((param - center) / width) ** 2)
            penalties.append(penalty)
        all_penalties = penalties + [kI_penalty, s_penalty, mesh_penalty] 
        return np.concatenate([resid, curvpenalty, all_penalties])
        
    try:
        res = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=40000)
    except Exception as e:
        raise RuntimeError(f"least_squares optimization failed: {e}")
    popt = res.x

    

    # Estimate uncertainties from the covariance of the residuals at the solution
    residuals = res.fun
    dof = max(0, len(residuals) - len(popt))  # degrees of freedom
    if dof > 0:
        residual_variance = np.sum(residuals**2) / dof
        try:
            pcov = np.linalg.inv(res.jac.T @ res.jac) * residual_variance
        except np.linalg.LinAlgError:
            pcov = np.full((len(popt), len(popt)), np.nan)
        perr = np.sqrt(np.abs(np.diag(pcov)))
    else:
        pcov = np.full((len(popt), len(popt)), np.nan)
        perr = np.full(len(popt), np.nan)
    for i, name in enumerate(param_names):
        results[name] = popt[i]
        results[f"{name}_err"] = perr[i]
        results['pcov'] = pcov

    diag = np.sqrt(np.diag(pcov))
    with np.errstate(invalid='ignore'):
        corr = pcov / np.outer(diag, diag)
    print("\nParameter correlation matrix:")
    print(pd.DataFrame(corr, index=param_names, columns=param_names).round(2))


    # Ensure 'delt' and 'mesh' are always present
    if 'delt' not in results:
        results['delt'] = 0.0
        results['delt_err'] = 0.0
    #if 'mesh' not in results:
        #results['mesh'] = mesh_est 
        #results['mesh_err'] = mesh_err
    # Derived Rg1 and its uncertainty via propagation from s and mesh
    s_val = results.get('s', 0.0)
    mesh_val = results.get('mesh', 0.0)
    s_err = results.get('s_err', 0.0)
    mesh_err = results.get('mesh_err', 0.0)
    cov_s_mesh = pcov[param_names.index('s'), param_names.index('mesh')] if pcov is not None and 's' in param_names and 'mesh' in param_names else 0.0 # covariance between s and mesh from the fit, if available
    Rg1_val = 0.5 * mesh_val * s_val
    dRg1_ds = 0.5 * mesh_val
    dRg1_dmesh = 0.5 * s_val
    var_Rg1 = (dRg1_ds ** 2) * (s_err ** 2) + (dRg1_dmesh ** 2) * (mesh_err ** 2) + 2 * dRg1_ds * dRg1_dmesh * cov_s_mesh # propagation of uncertainty to Rg1, including covariance
    Rg1_err = np.sqrt(abs(var_Rg1))
    results['Rg1'] = Rg1_val
    results['Rg1_err'] = Rg1_err

    return results

def plot_fit(q, I, popt, title, show=True, save_path=None):
    fig = plt.figure(figsize=(7,5))
    plt.loglog(q, I, 'o', label="Data")
    plt.loglog(q, ubg_model(q, *popt), '-', label="UBG Fit")
    plt.xlabel("q (Å⁻¹)")
    plt.ylabel("I(q)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def residuals(params, q, I,bkg_est, B1_est, kI_lower=0.001, kI_preferred=0.02):
    s, pack, mesh, delt, kI = params
    model = ubg_model(q, bkg_est, s, B1_est, pack, mesh, delt, kI)
    resid = (I - model) / I
    second_diff= np.diff(model, n=2)

    penalty = 1e2 * np.sum(second_diff**2) # penalize non-smoothness in the fit
    
    return np.append(resid, penalty)


def plot_fit_with_residuals(q, I, results, title, show=True, save_path=None):
    # Assemble full fit parameters
    bkg = results.get('bkg', np.nan)
    s = results.get('s', np.nan)
    B1 = results.get('B1', np.nan)
    pack = results.get('pack', np.nan)
    mesh = results.get('mesh', np.nan)
    delt = results.get('delt', np.nan) 
    kI = results.get('kI', np.nan)

    # compute fitted curve
    Ifit = ubg_model(q, bkg, s, B1, pack, mesh, delt, kI)

    resid = (I - Ifit) / I

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax1.loglog(q, I, 'o', label='Data', markersize=4)
    ax1.loglog(q, Ifit, '-', label='UBG Fit')
    ax1.set_ylabel('I(q)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.semilogx(q, resid, 'k.', markersize=3)
    ax2.axhline(0, color='r', linestyle='--', linewidth=1)
    ax2.set_xlabel('q (Å⁻¹)')
    ax2.set_ylabel('Normalized Residual')
    ax2.set_ylim(-1.0, 1.0)
    ax2.grid(alpha=0.3)

    # Annotate parameters on the top plot
    def fmt(val, err):
        try:
            return f"{val:.3g} ± {err:.2g}"
        except Exception:
            return str(val)

    txt_lines = []
    txt_lines.append(f"bkg = {fmt(results.get('bkg',np.nan), results.get('bkg_err',np.nan))}")
    txt_lines.append(f"Rg1 = {fmt(results.get('Rg1',np.nan), results.get('Rg1_err',np.nan))}")
    txt_lines.append(f"B1  = {fmt(results.get('B1',np.nan), results.get('B1_err',np.nan))}")
    txt_lines.append(f"packing = {fmt(results.get('pack',np.nan), results.get('pack_err',np.nan))}")
    txt_lines.append(f"mesh = {fmt(results.get('mesh',np.nan), results.get('mesh_err',np.nan))}")
    txt_lines.append(f"del = {fmt(results.get('delt',np.nan), results.get('delt_err',np.nan))}")
    txt_lines.append(f"k1 = {fmt(results.get('kI',np.nan), results.get('kI_err',np.nan))}")

    textbox = "\n".join(txt_lines)
    ax1.text(0.02, 0.95, textbox, transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a single .dat SAXS file or a folder of .dat files using the UBG model")
    parser.add_argument("datfile", nargs='?', default=None, help="Path to the .dat file to fit (optional if --folder is used)")
    parser.add_argument("--folder", help="Path to a folder containing .dat files to process (batch mode)")
    parser.add_argument("--out-dir", help="Output directory for saved plots when using --folder")
    parser.add_argument("--show-plot", action="store_true", help="Show the fit plot after fitting")
    parser.add_argument("--save-plot", action="store_true", help="Save the fit plot to disk")
    parser.add_argument("--output", help="Output path for saved plot (defaults to same folder as datfile)")
    args= parser.parse_args()

    if args.folder:
        folder = args.folder
        if not os.path.isdir(folder):
            raise SystemExit(f"Folder not found: {folder}")

        dat_files = sorted(glob.glob(os.path.join(folder, "*.dat")))
        if not dat_files:
            raise SystemExit(f"No .dat files found in folder: {folder}")

        out_dir = args.out_dir or os.path.join(folder, "ubg_fits")
        os.makedirs(out_dir, exist_ok=True)

        results_list = []

        import time
        for f in dat_files:
            print(f"\nFitting: {f}")
            q, I = load_dat_file(f)

            
            try:
                #results = fit_with_steps(q, I, f, fix_delt=not args.free_delt,q_high_low=0.075, q_high_cap=0.16, q_low=0.075, q_high=0.16)
                results = fit_with_penalties(q, I, curvewt = 0.4, kI_floor=0.0001, delt_floor=0.009, q_high_low=0.075, q_high_cap=0.15, q_low=0.01, q_high=0.11)
            except Exception as e:
                print(f"  Fit failed for {os.path.basename(f)}: {e}")
                results_list.append({
                    'filename': os.path.basename(f),
                    'status': 'failed',
                    'error': str(e)
                })
                time.sleep(0.1)
                continue

            base_name = os.path.splitext(os.path.basename(f))[0]
            save_path = os.path.join(out_dir, base_name + "_ubg_fit.png")
            # ensure unique
            if os.path.exists(save_path):
                root, ext = os.path.splitext(save_path)
                i = 1
                candidate = f"{root}_{i}{ext}"
                while os.path.exists(candidate):
                    i += 1
                    candidate = f"{root}_{i}{ext}"
                save_path = candidate

            # Print concise results
            print(f"  pack = {results['pack']:.6g} ± {results['pack_err']:.2g}")
            print(f"  delt = {results['delt']:.6g} ± {results['delt_err']:.2g}")
            print(f"  kI = {results['kI']:.6g} ± {results['kI_err']:.2g}")
            print(f"  s = {results['s']:.6g} ± {results['s_err']:.2g}")
            print(f"  eta = {results['mesh']:.6g} ± {results['mesh_err']:.2g}")

            # Save and/or show
            plot_fit_with_residuals(q, I, results, os.path.basename(f), show=args.show_plot, save_path=save_path if args.save_plot else None)
            results_list.append({
                'filename': os.path.basename(f),
                'status': 'ok',
                'B1': results.get('B1'),
                'B1_err': results.get('B1_err'),
                'bkg': results.get('bkg'),
                'bkg_err': results.get('bkg_err'),
                'delt': results.get('delt'),
                'delt_err': results.get('delt_err'),
                's': results.get('s'),
                's_err': results.get('s_err'),
                'Rg1': results.get('Rg1'),
                'Rg1_err': results.get('Rg1_err'),
                'pack': results.get('pack'),
                'pack_err': results.get('pack_err'),
                'mesh': results.get('mesh'),
                'mesh_err': results.get('mesh_err'),
                'kI': results.get('kI'),
                'kI_err': results.get('kI_err')
            })
            time.sleep(0.1)
            #save results to csv after each fit to avoid data loss if something goes wrong
            df = pd.DataFrame(results_list)
            df.to_csv(os.path.join(out_dir, "fit_results.csv"), index=False)
            #after all fits are done and saved, clear up the memory by deleting the results list
        del results_list
    elif args.datfile:
        datfile = args.datfile
        if not os.path.isfile(datfile):
            raise SystemExit(f"File not found: {datfile}")

        q, I = load_dat_file(datfile)
        try:
            results = fit_with_penalties(q, I, curvewt = 0.4, kI_floor=0.0001, delt_floor=0.0001, q_high_low=0.075, q_high_cap=0.15, q_low=0.01, q_high=0.11)
        except Exception as e:
            raise SystemExit(f"Fit failed: {e}")

        print(f"Fit results for {datfile}:")
        print(f"  pack = {results['pack']:.6g} ± {results['pack_err']:.2g}")
        print(f" delt = {results['delt']:.6g} ± {results['delt_err']:.2g}")
        print(f"  kI = {results['kI']:.6g} ± {results['kI_err']:.2g}")
        print(f"  eta = {results['mesh']:.6g} ± {results['mesh_err']:.2g}")

        save_path = args.output or os.path.splitext(datfile)[0] + "_ubg_fit.png"
        plot_fit_with_residuals(q, I, results, os.path.basename(datfile), show=args.show_plot, save_path=save_path if args.save_plot else None)