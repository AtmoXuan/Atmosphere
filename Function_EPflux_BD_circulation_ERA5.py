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
def compute_theta(T_da, plevel):
    """ Calculate potential temperature: theta"""
    p_pa = plevel * 100.0
    exponent = Rd / cp
    theta = T_da * (p0 / p_pa) ** exponent
    theta.name = 'theta'
    return theta

def compute_density(T_da, plevel):
    """ According to ERA5 T and pressure_level, calculate density: rho"""
    rho_vals = np.zeros_like(T_da.values)
    for i, p_hPa in enumerate(plevel.values):
        p_Pa = p_hPa * 100.0
        rho_vals[:, i, :] = p_Pa / (Rd * T_da.isel(pressure_level=i).values)
    return xr.DataArray(rho_vals, coords=T_da.coords, dims=T_da.dims, name='rho')

def pressure_to_height(GP):
    """ According to ERA5 GP and g, calculate GPH to represent height """
    return GP / g

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

# ========== single year processing ==========
def process_year(y):
    print(f"[INFO] Processing year {y}", flush=True)

    fnT = pattern_T.format(y=y)
    fnZ = pattern_Z.format(y=y)
    fnU = pattern_U.format(y=y)
    fnV = pattern_V.format(y=y)
    fnW = pattern_W.format(y=y)

    for fn in (fnT, fnZ, fnU, fnV, fnW):
        if not os.path.exists(fn):
            raise FileNotFoundError(f"缺少文件: {fn}")
    
    chunks = {'valid_time': chunk_time} if use_dask else None  # 不使用 dask  # 
    dsT = xr.open_dataset(fnT, chunks=chunks)
    dsZ = xr.open_dataset(fnZ, chunks=chunks)
    dsU = xr.open_dataset(fnU, chunks=chunks)
    dsV = xr.open_dataset(fnV, chunks=chunks)
    dsW = xr.open_dataset(fnW, chunks=chunks)
    ds = xr.merge([dsT, dsZ, dsU, dsV, dsW], compat='override')
    # ds = ds.sel(latitude=slice(90,-90), pressure_level=slice(1000,1))
    lat = ds['latitude']
    lat_rad = np.deg2rad(lat)
    f = 2 * Omega * np.sin(lat_rad)
    cosphi = np.cos(lat_rad)

    # potential temperature, density, height 
    theta = compute_theta(ds['t'], ds['pressure_level'])
    ds['theta'] = theta
    ds_zm = ds.mean(dim='longitude')
    z_da = pressure_to_height(ds_zm['z'])
    z_vals = z_da.values
    rho_da = compute_density(ds_zm['t'], ds['pressure_level'])
    rho_vals = rho_da.values

    ntime, nlev, nlat = ds_zm['theta'].shape
    dtheta_dz_vals = vertical_gradient(ds_zm['theta'].values, z_vals)
    du_dz_vals     = vertical_gradient(ds_zm['u'].values, z_vals)
    dtheta_dz_vals = np.where(abs(dtheta_dz_vals) < small, np.sign(dtheta_dz_vals) * small + small, dtheta_dz_vals)

    cos3d = np.broadcast_to(cosphi.values[np.newaxis, np.newaxis, :], (ntime,nlev,nlat))
    f3d   = np.broadcast_to(f.values[np.newaxis, np.newaxis, :], (ntime,nlev,nlat))
    ducos3d_dphi_vals = np.gradient(ds_zm['u'].values * cos3d, lat_rad.values, axis=2, edge_order=1)

    # ---- eddies ----
    u_eddy      = ds['u'] - ds_zm['u']
    v_eddy      = ds['v'] - ds_zm['v']
    w_eddy      = ds['w'] - ds_zm['w']
    theta_eddy  = ds['theta'] - ds_zm['theta']

    uv_eddy      = (u_eddy * v_eddy).mean(dim='longitude')
    vtheta_eddy  = (v_eddy * theta_eddy).mean(dim='longitude')
    uw_eddy      = (u_eddy * w_eddy).mean(dim='longitude')

    # ---- EP Flux ----
    uv_vals     = uv_eddy.values
    vtheta_vals = vtheta_eddy.values
    uw_vals     = uw_eddy.values

    F_phi_vals = rho_vals * r * cos3d * (du_dz_vals * vtheta_vals / dtheta_dz_vals - uv_vals)
    F_z_vals   = rho_vals * r * cos3d * ((f3d - ducos3d_dphi_vals / (r * cos3d)) * vtheta_vals / dtheta_dz_vals - uw_vals)

    # divF_y： meridional divergence ∂(Fφ cosφ)/∂φ
    dFphi_cos_dphi = np.gradient(F_phi_vals * cos3d, lat_rad.values, axis=2, edge_order=1)
    # divF_z： vertical divergence ∂Fz/∂z 
    dFz_dz = vertical_gradient(F_z_vals, z_vals)
    # EP Flux divergence
    divF = dFphi_cos_dphi/(r*cos3d) + dFz_dz
    # wave-induced acceleration
    dudt = divF / (r * cos3d * rho_vals)

    # ---- TEM residual circulation ----
    A = vtheta_vals / dtheta_dz_vals
    v_resi = ds_zm['v'].values - vertical_gradient((rho_vals * A), z_vals) / rho_vals
    w_resi = ds_zm['w'].values + np.gradient(cos3d * A, lat_rad.values, axis=2) / (r * cos3d)
    w_resi_da = xr.DataArray(w_resi, coords=ds_zm['w'].coords, dims=ds_zm['w'].dims, name='w_resi')
    if 90.0 in w_resi_da.latitude.values:
        w_resi_da.loc[:, :, 90.0] = np.nan
    if -90.0 in w_resi_da.latitude.values:
        w_resi_da.loc[:, :, -90.0] = np.nan
    omega_res = - rho_vals * g * w_resi_da
    ad_heating = - w_resi_da * dtheta_dz_vals
    
    # ========== Save ==========
    xr.Dataset({'uv_eddy': uv_eddy, 'vtheta_eddy': vtheta_eddy, 'uw_eddy': uw_eddy}).to_netcdf(os.path.join(out_dir, f'ERA5_eddies_{y}.nc'))
    xr.Dataset({'rho': rho_da}).to_netcdf(os.path.join(out_dir, f'ERA5_air_density_{y}.nc'))
    ds_ep = xr.Dataset(
    {
        'F_phi': xr.DataArray(
            F_phi_vals,
            coords={
                'valid_time': ds_zm.valid_time,
                'pressure_level': ds_zm.pressure_level,
                'latitude': ds_zm.latitude,
            },
            dims=('valid_time','pressure_level','latitude')
        ),
        'F_z': xr.DataArray(
            F_z_vals,
            coords={
                'valid_time': ds_zm.valid_time,
                'pressure_level': ds_zm.pressure_level,
                'latitude': ds_zm.latitude,
            },
            dims=('valid_time','pressure_level','latitude')
        ),
        'divF': xr.DataArray(
            divF,
            coords={
                'valid_time': ds_zm.valid_time,
                'pressure_level': ds_zm.pressure_level,
                'latitude': ds_zm.latitude,
            },
            dims=('valid_time','pressure_level','latitude')
        ),
        'dudt': xr.DataArray(
            dudt,
            coords={
                'valid_time': ds_zm.valid_time,
                'pressure_level': ds_zm.pressure_level,
                'latitude': ds_zm.latitude,
            },
            dims=('valid_time','pressure_level','latitude')
        )
    },
    attrs={'description': 'EP Flux and tendency fields'}
)
    ds_ep['F_phi'].attrs.update({'long_name':'Meridional EP Flux', 'units':'kg m s^-2'})
    ds_ep['F_z'].attrs.update({'long_name':'Vertical EP Flux', 'units':'kg m s^-2'})
    ds_ep['divF'].attrs.update({'long_name':'EP Flux Divergence', 'units':'kg m^-2 s^-2'})
    ds_ep['dudt'].attrs.update({'long_name':'Wave-forcing', 'units':'m s^-2'})
    ds_ep.to_netcdf(os.path.join(out_dir, f'ERA5_EPFlux_{y}.nc'))

    xr.Dataset({'v_resi': (('valid_time','pressure_level','latitude'), v_resi),
                'w_resi': w_resi_da,
                'omega_res': omega_res,
                'ad_heating': ad_heating
               }).to_netcdf(os.path.join(out_dir, f'ERA5_TEM_{y}.nc'))
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

