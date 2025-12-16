import xarray as xr
import numpy as np
import pandas as pd
import os
# U6090: the zonal-mean zonal winds, cosine weighted and averaged from 60° to 90°N.
# This function is used to calculate the daily U6090 climatology of 60-90°N latitude and the corresponding anomaly.
# reference: 
# Butler, A. H., Seidel, D. J., Hardiman, S. C., Butchart, N., Birner, T., & Match, A. (2015). Defining sudden stratospheric warmings. Bulletin of the American Meteorological Society, 96(11), 1913–1928. https://doi.org/10.1175/BAMS-D-13-00173.1
# Li, H., Li, Y., Wang, Y., Sun, J., & Wang, C. (2025). Impact of solar proton events on the stratospheric polar vortex in the northern hemisphere: A quantitative analysis. Journal of Geophysical Research: Space Physics, 130(4), e2024JA033068. https://doi.org/10.1029/2024JA033068

def weighted_lat_mean(file_path, var_name, level_bounds, lat_range):
    """
    file_path: str, ERA5 file path
    var_name: str, variable name, e.g.'u'
    ds_var: xarray.DataArray (valid_time, pressure_level, latitude, longitude)
    level_bounds = (1000, 1)  # hPa
    lat_range: tuple, (lat_min, lat_max)
    return: (valid_time, pressure_level) 加权纬向平均
    """
    ds = xr.open_dataset(file_path)
    ds_var = ds[var_name].sel(pressure_level=slice(level_bounds[0], level_bounds[1]), latitude=slice(lat_range[0], lat_range[1]))
    lonave_sel = ds_var.mean(dim="longitude")
    weights = np.cos(np.deg2rad(ds_var.latitude))
    weights.name = "weights"
    u_weighted = lonave_sel.weighted(weights).mean(dim="latitude")
    return u_weighted

def lat6090_climatology(data_dir, data_name, output_file, output_anom, years, var_name, level_bounds, lat_range, df_varname):
    """
    lat6090_climatology 的 Docstring
    :param data_dir: data content directory               data_dir = r"F:\Data\ERA5_1-1000hPa\ERA5_UVW_daily\U_ERA5_daily"
    :data_name: data name                                 e.g. "ERA5_U_wind_daily"
    :param output_file: daily data of 60-90° climatology  output_file = r"F:\Data\ERA5_1-1000hPa\ERA5_UVW_daily\U6090_climatology_1986-2020.parquet"
    :param output_anom: daily-climatology anomaly         output_anom = r"F:\Data\ERA5_1-1000hPa\ERA5_UVW_daily\U6090_clim_anomaly_1986-2020.parquet"
    :param years: e.g.                                    years = range(1986, 2021) 
    :param var_name: e.g.                                 var_name = 'u'
    :param level_bounds: e.g.                             level_bounds = (1000, 1)   # hPa
    :param lat_range: e.g.                                lat_range = (90, 60)  # ERA5 lat order
    :param df_varname: e.g.                               df_varname = df_varname
    usage:
    data_dir = r"F:\Data\ERA5_1-1000hPa\ERA5_UVW_daily\U_ERA5_daily"
    data_name = "ERA5_U_wind_daily"
    start_year = 1986
    end_year = 2020
    years = np.arange(start_year, end_year+1)
    output_file = rf"F:\Data\ERA5_1-1000hPa\ERA5_UVW_daily\U6090_climatology_{start_year}-{end_year}.parquet"
    output_anom = rf"F:\Data\ERA5_1-1000hPa\ERA5_UVW_daily\U6090_clim_anomaly_{start_year}-{end_year}.parquet"
    var_name = "u"
    level_bounds = (1000, 1)
    df_varname = "U6090"
    U_df_all, U_clim_stats = lat6090_climatology(data_dir, data_name, output_file, output_anom, years, var_name, level_bounds, lat_range, df_varname)
    """ 

    all_data = []
    for year in years:
        print(f"Processing {year} ...")
        file_path = os.path.join(data_dir, f"{data_name}_{year}.nc")

        if not os.path.exists(file_path):
            print(f"mising file: {file_path}")
            continue

        # ----- NH 60-90° -----
        var_NH = weighted_lat_mean(file_path, var_name, level_bounds, lat_range)
        df_NH = var_NH.to_dataframe(name=df_varname).reset_index()
        df_NH['hemi'] = 'NH'

        # ----- SH 60~-90° -----
        var_SH = weighted_lat_mean(file_path, var_name, level_bounds, lat_range)
        df_SH = var_SH.to_dataframe(name=df_varname).reset_index()
        df_SH['hemi'] = 'SH'

        # NH + SH
        df_year = pd.concat([df_NH, df_SH], ignore_index=True)
        df_year["month"] = df_year["valid_time"].dt.month
        df_year["day"]   = df_year["valid_time"].dt.day
        # if you want to delete the 29th Feb:
        # df_year = df_year[~((df_year["month"] == 2) & (df_year["day"] == 29))]
        all_data.append(df_year)

    df_all = pd.concat(all_data, ignore_index=True)
    df_daily_6090 = df_all.sort_values(['hemi','pressure_level','valid_time']).reset_index(drop=True)

    # ---------------------------- caluclate climatology (including 29th Feb) ----------------------------
    clim_stats = df_daily_6090.groupby(
        ['hemi', 'pressure_level', 'month', 'day']
    ).agg(
        mean=(df_varname,'mean'),
        median=(df_varname,'median'),
        q25=(df_varname, lambda x: np.percentile(x,25)),
        q75=(df_varname, lambda x: np.percentile(x,75))
    ).reset_index()

    clim_stats.to_parquet(output_file, index=False)
    print(f"has saved daily data of 60-90° climatology: {output_file}")

    # ---------------------------- aluclate daily climatology anomaly (including 29th Feb) ----------------------------
    df_all = df_daily_6090.merge(
        clim_stats[['hemi','pressure_level','month','day','mean']],
        on=['hemi','pressure_level','month','day'], 
        how='left'
    )

    df_all[f'{df_varname}_anom'] = df_all[df_varname] - df_all['mean']
    df_all.to_parquet(output_anom, index=False)
    print(f"has saved daily-climatology anomaly: {output_anom}")
    return df_all, clim_stats

