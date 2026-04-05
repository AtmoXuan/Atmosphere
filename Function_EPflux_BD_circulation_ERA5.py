import xarray as xr
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed

# Reference:
# G.J., B. (2020). Low latitude dynamical response to vortex split sudden stratospheric warming: An Eliassen Palm Flux perspective. Dynamics of Atmospheres and Oceans, 91, 101146. https://doi.org/10.1016/j.dynatmoce.2020.101146
# David G. Andrews et al. (1987), Middle Atmosphere Dynamics. 

# ========== Batch computing EPflux，Residual circulation and Heat Flux ==========
data_dir = r"F:\Data\ERA5_1-1000hPa"
pattern_T  = os.path.join(data_dir, "ERA5_T_daily",  "ERA5_Temperature_daily_{y}.nc")
pattern_Z  = os.path.join(data_dir, "ERA5_GPH_daily", "ERA5_GPH_daily_{y}.nc")
pattern_U  = os.path.join(data_dir, "ERA5_UVW_daily", "U_ERA5_daily",           "ERA5_U_wind_daily_{y}.nc")
pattern_V  = os.path.join(data_dir, "ERA5_UVW_daily", "V_ERA5_daily",           "ERA5_V_wind_daily_{y}.nc")
pattern_W  = os.path.join(data_dir, "ERA5_UVW_daily", "W_ERA5_daily",           "ERA5_W_wind_daily_{y}.nc")

out_dir = r"D:\Data\ERA5_EPFLUX"
os.makedirs(out_dir, exist_ok=True)

# physical constants
r = 6371000.0          # Earth radius (m)
Omega = 7.292116e-5    # rad/s
Rd = 287.05            # gas constant for dry air J·kg-1·K-1 
cp = 1004.0            # specific heat at constant pressure for dry air J·kg-1·K-1
p0 = 100000.0          # Pa
g  = 9.80665           # m/s²

use_dask = True
chunk_time = 'auto'
small = 1e-12

# ========== Utility functions ==========
def vertical_gradient(var, z):
    """
    calculate d(var)/dz
    var: (time, lev, lat)
    z  : (time, lev, lat)
    """
    dvar_mid = var[:, 2:, :] - var[:, :-2, :]
    dz_mid = z[:, 2:, :] - z[:, :-2, :]
    grad_mid = dvar_mid / dz_mid

    grad = np.zeros_like(var)
    grad[:, 1:-1, :] = grad_mid
    # bottom boundary
    grad[:, 0, :] = (var[:, 1, :] - var[:, 0, :]) / (z[:, 1, :] - z[:, 0, :])
    # top boundary
    grad[:, -1, :] = (var[:, -1, :] - var[:, -2, :]) / (z[:, -1, :] - z[:, -2, :])
    return grad
    
def mask_poles(da, lat_name="latitude", pole_lat=89.5):
    """
    Mask polar points (|lat| >= pole_lat) by NaN
    """
    lat = da[lat_name]
    return da.where(np.abs(lat) < pole_lat)
    
# ========== single year processing ==========
def process_year(y):
    print(f"[INFO] Processing year {y}", flush=True)

    fnT, fnZ, fnU, fnV, fnW = [p.format(y=y) for p in [pattern_T, pattern_Z, pattern_U, pattern_V, pattern_W]]
    chunks = {'valid_time': chunk_time} if use_dask else None 
    ds = xr.merge([xr.open_dataset(f, chunks=chunks) for f in [fnT, fnZ, fnU, fnV, fnW]], compat='override')
    ds = ds.sel(pressure_level=slice(1000, 1)).dropna(dim='pressure_level', how='all')   
    p_pa = ds['pressure_level'] * 100.0
    ds['theta'] = ds['t'] * (p0 / p_pa)**(Rd/cp)
    ds_zm = ds.mean(dim='longitude')
    
    common_coords = ds_zm.coords
    common_dims = ds_zm.dims
    lat_rad_val = np.deg2rad(ds_zm['latitude'].values)
    
    ntime, nlev, nlat = ds_zm['t'].shape
    cosphi = np.cos(lat_rad_val)
    f = 2 * Omega * np.sin(lat_rad_val)
    cos3d = np.broadcast_to(cosphi[np.newaxis, np.newaxis, :], (ntime, nlev, nlat))
    f3d   = np.broadcast_to(f[np.newaxis, np.newaxis, :], (ntime, nlev, nlat))

    u_zm = ds_zm['u'].values
    v_zm = ds_zm['v'].values
    w_pa_s = ds_zm['w'].values
    t_zm = ds_zm['t'].values
    theta_zm = ds_zm['theta'].values
    z_vals = (ds_zm['z'] / g).values    # GPH
    plev_pa = ds_zm['pressure_level'].values * 100.0
    rho_vals = plev_pa[None, :, None] / (Rd * t_zm)

    #  Pa/s => m/s
    # omega = -rho * g * w => w = -omega / (rho * g)
    w_zm = - w_pa_s / (rho_vals * g)

    dtheta_dz_vals = vertical_gradient(theta_zm, z_vals)
    dtheta_dz_vals = np.where(np.abs(dtheta_dz_vals) < small, np.sign(dtheta_dz_vals) * small + small, dtheta_dz_vals)
    du_dz_vals     = vertical_gradient(u_zm, z_vals)
    
    ducos3d_dphi_vals = np.gradient(u_zm * cos3d, lat_rad_val, axis=2, edge_order=1)
    dtheta_dphi_vals  = np.gradient(theta_zm, lat_rad_val, axis=2, edge_order=1)

    u_eddy      = ds['u'] - ds_zm['u']
    v_eddy      = ds['v'] - ds_zm['v']
    theta_eddy  = ds['theta'] - ds_zm['theta']

    rho_4d  = p_pa / (Rd * ds['t'])      
    w_m_s_4d = - ds['w'] / (rho_4d * g)  
    w_eddy = w_m_s_4d - w_m_s_4d.mean(dim='longitude')

    uv_vals      = (u_eddy * v_eddy).mean(dim='longitude').values
    vtheta_vals  = (v_eddy * theta_eddy).mean(dim='longitude').values
    uw_vals      = (u_eddy * w_eddy).mean(dim='longitude').values

    # EP Flux & Refractive Index 
    F_phi_vals = rho_vals * r * cos3d * (du_dz_vals * vtheta_vals / dtheta_dz_vals - uv_vals)
    F_z_vals   = rho_vals * r * cos3d * ((f3d - ducos3d_dphi_vals / (r * cos3d)) * vtheta_vals / dtheta_dz_vals - uw_vals)

    dFphi_cos_dphi = np.gradient(F_phi_vals * cos3d, lat_rad_val, axis=2, edge_order=1)
    dFz_dz = vertical_gradient(F_z_vals, z_vals)
    divF = dFphi_cos_dphi / (r * cos3d) + dFz_dz
    wave_forcing = divF / (r * cos3d * rho_vals)

    # wave refractive index
    N_squre_vals = np.maximum((g / theta_zm) * dtheta_dz_vals, 1e-10)  
    Long1 = np.gradient(ducos3d_dphi_vals / cos3d, lat_rad_val, axis=2, edge_order=1)
    Long2 = vertical_gradient((rho_vals * du_dz_vals) / N_squre_vals, z_vals)
    q_phi = 2 * Omega * cos3d / r - Long1 / r**2 - (f3d**2 / rho_vals) * Long2 
    
    u_stable = np.where(np.abs(u_zm) < 0.1, 0.1, u_zm)
    H_scale = Rd * t_zm / g
    def get_nk2(k_val):
        return q_phi / u_stable - (k_val / (r * cos3d))**2 - f3d**2 / (4 * N_squre_vals * H_scale**2)
    
    n1_squre = get_nk2(1)
    n2_squre = get_nk2(2)

    # TEM & Heating 
    A = vtheta_vals / dtheta_dz_vals
    v_resi = v_zm - vertical_gradient((rho_vals * A), z_vals) / rho_vals
    w_resi = w_zm + np.gradient(cos3d * A, lat_rad_val, axis=2) / (r * cos3d)   
    
    # Pa/s 
    omega_res = - rho_vals * g * w_resi
    hor_adv   = - (v_resi / r) * dtheta_dphi_vals
    ad_heating = - w_resi * dtheta_dz_vals

    # K/s
    dyn_heating = (t_zm / theta_zm) * (hor_adv + ad_heating)

    # save
    ds_save_eddy = xr.Dataset({
        'uv_eddy':(common_dims, uv_vals),
        'vtheta_eddy':(common_dims, vtheta_vals),
        'uw_eddy':(common_dims, uw_vals)
    }, coords=common_coords)
    
    ds_save_air = xr.Dataset({'rho': (common_dims, rho_vals)}, coords=common_coords)

    ds_save_ep = xr.Dataset({
        'F_phi': (common_dims, F_phi_vals),
        'F_z': (common_dims, F_z_vals),
        'divF': (common_dims, divF),
        'wave_forcing': (common_dims, wave_forcing),
        'n1_squre': (common_dims, n1_squre),
        'n2_squre': (common_dims, n2_squre)
    }, coords=common_coords)

    ds_save_tem = xr.Dataset({
        'v_resi': (common_dims, v_resi),
        'w_resi': (common_dims, w_resi),
        'omega_res': (common_dims, omega_res),
        'ad_heating': (common_dims, ad_heating),
        'dyn_heating': (common_dims, dyn_heating)
    }, coords=common_coords)

    # rho
    ds_save_air['rho'].attrs = {'long_name': 'Air density', 'units': 'kg m-3'}
    
    # EP Flux 
    ds_save_ep['F_phi'].attrs = {'long_name': 'Meridional component of EP flux', 'units': 'kg m-1 s-2'}
    ds_save_ep['F_z'].attrs = {'long_name': 'Vertical component of EP flux', 'units': 'kg m-1 s-2'}
    ds_save_ep['divF'].attrs = {'long_name': 'Divergence of EP flux', 'units': 'kg m-2 s-2'}
    ds_save_ep['wave_forcing'].attrs = {'long_name': 'Wave forcing (divF / rho*r*cosphi)', 'units': 'm s-2'}
    ds_save_ep['n1_squre'].attrs = {'long_name': 'Squared refractive index for wavenumber 1', 'units': 'None'}
    ds_save_ep['n2_squre'].attrs = {'long_name': 'Squared refractive index for wavenumber 2', 'units': 'None'}

    # TEM 
    ds_save_tem['v_resi'].attrs = {'long_name': 'Residual meridional velocity (v*)', 'units': 'm s-1'}
    ds_save_tem['w_resi'].attrs = {'long_name': 'Residual vertical velocity (w*)', 'units': 'm s-1'}
    ds_save_tem['omega_res'].attrs = {'long_name': 'Residual vertical velocity in pressure coord', 'units': 'Pa s-1'}
    ds_save_tem['ad_heating'].attrs = {'long_name': 'Adiabatic heating term', 'units': 'K s-1'}
    ds_save_tem['dyn_heating'].attrs = {'long_name': 'Dynamical heating rate', 'units': 'K s-1'}
    
    for ds_final, name in zip([ds_save_eddy, ds_save_air, ds_save_ep, ds_save_tem], ['Eddies', 'Air_Density', 'EPFlux', 'TEM']):
        # Mask poles
        for var in ds_final.data_vars:
            ds_final[var] = mask_poles(ds_final[var])
        
        # ds_final = ds_final.astype(np.float32)

        out_name = f'ERA5_{name}_{y}.nc'
        ds_final.to_netcdf(os.path.join(out_dir, out_name))

    print(f"[INFO] Year {y} saved successfully.")

# ========== Batch processing ==========
Parallel(n_jobs=4)(
    delayed(process_year)(y) for y in range(1986, 2025)
)

# more details:
# Generally, the wave-forcing (dudt) is drawn in the pressure-latitude figure. 
# G.J., B. (2020): 
# For the clear view of E–P flux vectors throughout the stratosphere
# E–P flux vectors are multiplied by (exp z/H) (is approximately equal to density)  (Mechoso et al., 1985)
# The vertical component of E–P flux is magnified by a factor of 150 with respect to the horizontal component (Randel and Boville, 1987)

import matplotlib.pyplot as plt
def plot_epflux(ds, dudt, F_phi_vals, F_z_vals, rho_vals):
    lat = ds['latitude'].values
    plev = ds['pressure_level'].values

    EPFD_plot = dudt[:, :, :]*86400     # m/s/day -> m/day
    
    Fphi_plot  = F_phi_vals[:, :, :] /rho_vals
    F_z = F_z_vals / rho_vals     
    vert_factor = 150.0                
    Fz_plot = F_z * vert_factor

    fig, ax = plt.subplots(figsize=(8,6),dpi=300)

    levels = np.linspace(-20, 20, 9)  
    cf = ax.contourf(lat, plev, EPFD_plot[0], levels=levels, cmap='RdBu_r',  extend='both') 
    
    # zero contour
    ax.contour(lat, plev, EPFD_plot[0], levels=[0], colors='k', linewidths=0.8)
    
    # quiver plot
    Q = ax.quiver(lat, plev, Fphi_plot[0], Fz_plot[0],
                width=0.002, headwidth=2, headlength=4, headaxislength=2, color = 'black')
    
    # log pressure axis
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel('Latitude (°N)')
    ax.set_title('EP Flux and EPFD - 19890205')

    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('EPFD (m/s/day)')

    plt.show()

