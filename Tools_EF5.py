# ======================================
# Tools_EF5.py - EF5 Hydrological Model Tools
# A collection of tools for downloading and processing data for the EF5 hydrological model
# Includes functions for:
# - MRMS precipitation data download and extraction
# - PET data download and extraction
# - Raster data processing and conversion
# - Basin boundary clipping
# - USGS discharge data download
# ======================================

import os
import requests
import gzip
import shutil
import tarfile
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import rasterio
from osgeo import gdal
import geopandas as gpd
from rasterio.windows import from_bounds
import io
import matplotlib.pyplot as plt

def download_mrms_precipitation(start_date, end_date, download_folder='../MRMS_precipitation'):
    """
    Downloads and extracts MRMS hourly precipitation data for a given date range.
    
    Parameters:
    -----------
    start_date : datetime
        The start date and time for data download
    end_date : datetime
        The end date and time for data download
    download_folder : str, optional
        Directory to save the downloaded and extracted files (default: '../MRMS_precipitation')
    """
    # Create directories if they do not exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Base URL for data
    base_url = "https://mtarchive.geol.iastate.edu/"

    # Calculate total number of hours for progress bar
    total_hours = int((end_date - start_date).total_seconds() / 3600) + 1
    
    print(f"Downloading MRMS precipitation data from {start_date} to {end_date}")
    print(f"Files will be saved to: {os.path.abspath(download_folder)}")
    
    # Loop through each hourly timestamp in the date range
    current_time = start_date
    failed_downloads = 0
    
    with tqdm(total=total_hours, desc="Downloading MRMS data") as pbar:
        while current_time <= end_date:
            # Format year, month, day, and hour as strings
            year_str = current_time.strftime('%Y')
            month_str = current_time.strftime('%m')
            day_str = current_time.strftime('%d')
            hour_str = current_time.strftime('%H')

            # Construct the file name
            file_name = f"MultiSensor_QPE_01H_Pass2_00.00_{year_str}{month_str}{day_str}-{hour_str}0000.grib2.gz"
            
            # Construct the full download URL
            file_url = f"{base_url}{year_str}/{month_str}/{day_str}/mrms/ncep/MultiSensor_QPE_01H_Pass2/{file_name}"
                        
            # Define the full local file path
            output_file = os.path.join(download_folder, file_name)
            
            # Download the file
            try:
                response = requests.get(file_url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Save the compressed file
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                # Extract the file
                extracted_file = output_file[:-3]  # Remove .gz extension
                with gzip.open(output_file, 'rb') as f_in:
                    with open(extracted_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Delete the original .gz file
                os.remove(output_file)
            except Exception as e:
                failed_downloads += 1
                tqdm.write(f"Failed to download or extract: {file_url} - Error: {str(e)[:100]}...")

            # Move to the next hour
            current_time += timedelta(hours=1)
            pbar.update(1)
    
    print(f"MRMS download complete. Files saved in: {os.path.abspath(download_folder)}")
    if failed_downloads > 0:
        print(f"Note: {failed_downloads} files failed to download")

def download_pet_data(start_date, end_date, download_folder='../PET_data'):
    """
    Downloads and extracts USGS PET data for a given date range.
    
    Parameters:
    -----------
    start_date : datetime
        The start date for downloading data
    end_date : datetime
        The end date for downloading data
    save_folder : str, optional
        Directory to save the downloaded and extracted files (default: '../PET_data')
    """
    # Create directory if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    # USGS PET data base URL
    base_url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"
    
    # Calculate total number of days for progress bar
    total_days = (end_date - start_date).days + 1
    
    print(f"Downloading PET data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Files will be saved to: {os.path.abspath(download_folder)}")
    
    # Loop through each date in the range
    current_date = start_date
    failed_downloads = 0
    
    with tqdm(total=total_days, desc="Downloading PET data") as pbar:
        while current_date <= end_date:
            # Generate the filename in "yymmdd" format
            file_name = f"et{current_date.strftime('%y%m%d')}.tar.gz"
            
            # Construct the full download URL
            file_url = f"{base_url}{file_name}"
            
            # Define the local save path
            save_path = os.path.join(download_folder, file_name)
            
            # Download the file
            try:
                response = requests.get(file_url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Save the file
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                # Extract the .tar.gz file
                with tarfile.open(save_path, 'r:gz') as tar:
                    tar.extractall(path=download_folder)
                
                # Delete the original .tar.gz file to save space
                os.remove(save_path)
                
            except Exception as e:
                failed_downloads += 1
                tqdm.write(f"Failed to download: {file_url} - Error: {str(e)[:100]}...")
            
            # Move to the next day
            current_date += timedelta(days=1)
            pbar.update(1)
    
    print(f"PET data download complete. Files saved in: {os.path.abspath(download_folder)}")
    if failed_downloads > 0:
        print(f"Note: {failed_downloads} files failed to download")

# def process_mrms_grib2_to_tif(input_folder='../MRMS_precipitation', output_folder='../CREST_input/MRMS/', basin_shp_path='shpFile/Basin_selected_5.shp'):
#     """
#     Process MRMS grib2 files to GeoTIFF format and clip to basin boundary.
    
#     Args:
#         input_folder (str): Path to the folder containing MRMS grib2 files
#         output_folder (str): Path to save the output GeoTIFF files
#         basin_shp_path (str): Path to the basin shapefile for clipping
#     """
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     else:
#         # Clear all files in the output folder
#         for file_path in glob.glob(os.path.join(output_folder, '*')):
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#         print(f"Cleared all existing files in {output_folder}")
    
#     # Load the basin shapefile
#     try:
#         basin_gdf = gpd.read_file(basin_shp_path)
#         basin_bounds = basin_gdf.total_bounds  # (minx, miny, maxx, maxy)
        
#         # Expand the bounds by 5% to make the clipping area slightly larger than the basin
#         width = basin_bounds[2] - basin_bounds[0]
#         height = basin_bounds[3] - basin_bounds[1]
#         buffer_x = width * 1
#         buffer_y = height * 1
        
#         expanded_bounds = (
#             basin_bounds[0] - buffer_x,  # minx
#             basin_bounds[1] - buffer_y,  # miny
#             basin_bounds[2] + buffer_x,  # maxx
#             basin_bounds[3] + buffer_y   # maxy
#         )
        
#         print(f"Loaded basin shapefile: {basin_shp_path}")
#         print(f"Original bounds: ({basin_bounds[0]:.3f}, {basin_bounds[1]:.3f}, {basin_bounds[2]:.3f}, {basin_bounds[3]:.3f})")
#         print(f"Expanded bounds: ({expanded_bounds[0]:.3f}, {expanded_bounds[1]:.3f}, {expanded_bounds[2]:.3f}, {expanded_bounds[3]:.3f})")
#     except Exception as e:
#         print(f"Error loading basin shapefile: {str(e)}")
#         print("Processing will continue without clipping to basin boundary")
#         basin_gdf = None
#         expanded_bounds = None
    
#     # Find all grib2 files in the input folder
#     grib2_files = glob.glob(os.path.join(input_folder, '*.grib2'))
    
#     if not grib2_files:
#         print(f"No grib2 files found in {input_folder}")
#         return
    
#     print(f"Processing {len(grib2_files)} MRMS grib2 files to GeoTIFF format")
#     print(f"Input folder: {os.path.abspath(input_folder)}")
#     print(f"Output folder: {os.path.abspath(output_folder)}")
    
#     failed_files = 0
    
#     # Process each grib2 file
#     with tqdm(total=len(grib2_files), desc="Converting MRMS to GeoTIFF") as pbar:
#         for grib_file in grib2_files:
#             # Get the base filename without extension
#             base_name = os.path.basename(grib_file)
#             output_name = os.path.splitext(base_name)[0] + '.tif'
#             output_path = os.path.join(output_folder, output_name)
            
#             try:
#                 # Open the grib2 file with GDAL
#                 src_ds = gdal.Open(grib_file)
#                 if src_ds is None:
#                     failed_files += 1
#                     tqdm.write(f"Could not open {grib_file}")
#                     pbar.update(1)
#                     continue
                
#                 if basin_gdf is not None:
#                     # Process with basin clipping
#                     with rasterio.open(grib_file) as src:
#                         data = src.read(1)
#                         # Set values >1000 or <0 to -9999
#                         data = np.where((data > 1000) | (data < 0), -9999, data)
#                         # Explicitly convert to float32
#                         data_float32 = data.astype(np.float32)
                        
#                         # Get the window for the expanded basin bounds
#                         window = from_bounds(expanded_bounds[0], expanded_bounds[1], 
#                                             expanded_bounds[2], expanded_bounds[3], 
#                                             src.transform)
                        
#                         # Read only the data within the expanded basin bounds
#                         clipped_data = src.read(1, window=window)
#                         clipped_data = np.where((clipped_data > 1000) | (clipped_data < 0), -9999, clipped_data)
#                         # Explicitly convert to float32
#                         clipped_data = clipped_data.astype(np.float32)
                        
#                         # Get the transform for the clipped data
#                         clipped_transform = rasterio.windows.transform(window, src.transform)
                        
#                         # Update metadata for the clipped output
#                         new_meta = {
#                             'driver': 'GTiff',
#                             'height': clipped_data.shape[0],
#                             'width': clipped_data.shape[1],
#                             'count': 1,
#                             'dtype': 'float32',
#                             'crs': src.crs,
#                             'transform': clipped_transform,
#                             'nodata': -9999,
#                             'compress': 'none'
#                         }
#                         with rasterio.open(output_path, 'w', **new_meta) as dst:
#                             dst.write(clipped_data, 1)
#                 else:
#                     # Create output GeoTIFF without clipping
#                     driver = gdal.GetDriverByName('GTiff')
#                     dst_ds = driver.CreateCopy(output_path, src_ds, 0)
                    
#                     # Get the band
#                     band = dst_ds.GetRasterBand(1)
#                     data = band.ReadAsArray()
#                     # Explicitly convert to float32
#                     data = data.astype(np.float32)
                    
#                     # Set values >1000 or <0 to -9999
#                     data = np.where((data > 1000) | (data < 0), -9999, data)
                    
#                     # Set nodata value
#                     band.SetNoDataValue(-9999)
                    
#                     # Write the data as float32
#                     band.WriteArray(data)
                    
#                     # Close datasets
#                     dst_ds = None
                
#                 src_ds = None
                
#             except Exception as e:
#                 failed_files += 1
#                 tqdm.write(f"Error processing {base_name}: {str(e)[:100]}...")
            
#             pbar.update(1)
    
#     # Process files in output folder to ensure correct format
#     tif_files = glob.glob(os.path.join(output_folder, '*.tif'))
#     print(f"Ensuring proper format for {len(tif_files)} output files")
    
#     with tqdm(total=len(tif_files), desc="Checking output formats") as pbar:
#         for file_path in tif_files:
#             try:
#                 with rasterio.open(file_path) as src:
#                     data = src.read(1)
#                     # Set values >1000 or <0 to -9999
#                     data = np.where((data > 1000) | (data < 0), -9999, data)
#                     data_float32 = data.astype('float32')
#                     meta = src.meta.copy()
#                     meta.update({'dtype': 'float32', 'nodata': -9999, 'compress': 'none'})
                    
#                 with rasterio.open(file_path, 'w', **meta) as dst:
#                     dst.write(data_float32, 1)
#             except Exception as e:
#                 tqdm.write(f"Error formatting {os.path.basename(file_path)}: {str(e)[:100]}...")
            
#             pbar.update(1)
    
#     print(f"MRMS conversion completed. Output files saved to {os.path.abspath(output_folder)}")
#     if failed_files > 0:
#         print(f"Note: {failed_files} files failed to process")

import os
import glob
import numpy as np
import geopandas as gpd
from osgeo import gdal
import rasterio
from rasterio.windows import from_bounds
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def _process_single_file(grib_file, output_folder, basin_clipping, expanded_bounds):
    """
    Helper function to process a single GRIB2 file.
    It converts the file to GeoTIFF format and clips it if basin clipping is enabled.
    
    Args:
        grib_file (str): Path to the GRIB2 file.
        output_folder (str): Folder to save the output GeoTIFF file.
        basin_clipping (bool): Whether to perform clipping to basin bounds.
        expanded_bounds (tuple or None): Expanded bounds for clipping (minx, miny, maxx, maxy).
        
    Returns:
        tuple: (base filename, error message or None if successful)
    """
    base_name = os.path.basename(grib_file)
    output_name = os.path.splitext(base_name)[0] + '.tif'
    output_path = os.path.join(output_folder, output_name)
    try:
        # Open the grib2 file using GDAL
        src_ds = gdal.Open(grib_file)
        if src_ds is None:
            return (base_name, "Could not open file")
        
        if basin_clipping:
            # Process with basin clipping using rasterio
            with rasterio.open(grib_file) as src:
                # Read the data and apply nodata filter
                data = src.read(1)
                data = np.where((data > 1000) | (data < 0), -9999, data)
                data_float32 = data.astype(np.float32)
                
                # Determine window corresponding to the expanded basin bounds
                window = from_bounds(expanded_bounds[0], expanded_bounds[1],
                                     expanded_bounds[2], expanded_bounds[3],
                                     src.transform)
                # Read only the data within the window
                clipped_data = src.read(1, window=window)
                clipped_data = np.where((clipped_data > 1000) | (clipped_data < 0), -9999, clipped_data)
                clipped_data = clipped_data.astype(np.float32)
                
                # Get the transform for the clipped window
                clipped_transform = rasterio.windows.transform(window, src.transform)
                
                # Prepare metadata for output GeoTIFF
                new_meta = {
                    'driver': 'GTiff',
                    'height': clipped_data.shape[0],
                    'width': clipped_data.shape[1],
                    'count': 1,
                    'dtype': 'float32',
                    'crs': src.crs,
                    'transform': clipped_transform,
                    'nodata': -9999,
                    'compress': 'none'
                }
                with rasterio.open(output_path, 'w', **new_meta) as dst:
                    dst.write(clipped_data, 1)
        else:
            # Process without basin clipping
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.CreateCopy(output_path, src_ds, 0)
            band = dst_ds.GetRasterBand(1)
            data = band.ReadAsArray()
            data = data.astype(np.float32)
            data = np.where((data > 1000) | (data < 0), -9999, data)
            band.SetNoDataValue(-9999)
            band.WriteArray(data)
            dst_ds = None
        
        src_ds = None
        return (base_name, None)
    except Exception as e:
        return (base_name, str(e)[:100])

def process_mrms_grib2_to_tif(input_folder='../MRMS_precipitation', 
                              output_folder='../CREST_input/MRMS/', 
                              basin_shp_path='shpFile/Basin_selected_5.shp',
                              num_processes=1):
    """
    Process MRMS grib2 files to GeoTIFF format and clip to basin boundary.
    The function now supports parallel processing using multiple processes.
    
    Args:
        input_folder (str): Path to the folder containing MRMS grib2 files.
        output_folder (str): Path to save the output GeoTIFF files.
        basin_shp_path (str): Path to the basin shapefile for clipping.
        num_processes (int): Number of processes to use (use 1 for sequential processing).
    """
    # Create or clear the output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for file_path in glob.glob(os.path.join(output_folder, '*')):
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared all existing files in {output_folder}")
    
    # Load the basin shapefile and compute expanded bounds for clipping
    try:
        basin_gdf = gpd.read_file(basin_shp_path)
        basin_bounds = basin_gdf.total_bounds  # (minx, miny, maxx, maxy)
        
        # Expand the bounds by 100% of the width/height (adjust factor as needed)
        width = basin_bounds[2] - basin_bounds[0]
        height = basin_bounds[3] - basin_bounds[1]
        buffer_x = width * 1
        buffer_y = height * 1
        
        expanded_bounds = (
            basin_bounds[0] - buffer_x,  # minx
            basin_bounds[1] - buffer_y,  # miny
            basin_bounds[2] + buffer_x,  # maxx
            basin_bounds[3] + buffer_y   # maxy
        )
        
        print(f"Loaded basin shapefile: {basin_shp_path}")
        print(f"Original bounds: ({basin_bounds[0]:.3f}, {basin_bounds[1]:.3f}, "
              f"{basin_bounds[2]:.3f}, {basin_bounds[3]:.3f})")
        print(f"Expanded bounds: ({expanded_bounds[0]:.3f}, {expanded_bounds[1]:.3f}, "
              f"{expanded_bounds[2]:.3f}, {expanded_bounds[3]:.3f})")
        basin_clipping = True
    except Exception as e:
        print(f"Error loading basin shapefile: {str(e)}")
        print("Processing will continue without clipping to basin boundary")
        basin_gdf = None
        expanded_bounds = None
        basin_clipping = False
    
    # Retrieve list of GRIB2 files from the input folder
    grib2_files = glob.glob(os.path.join(input_folder, '*.grib2'))
    if not grib2_files:
        print(f"No grib2 files found in {input_folder}")
        return
    
    print(f"Processing {len(grib2_files)} MRMS grib2 files to GeoTIFF format")
    print(f"Input folder: {os.path.abspath(input_folder)}")
    print(f"Output folder: {os.path.abspath(output_folder)}")
    
    failed_files = 0

    if num_processes == 1:
        # Sequential processing if num_processes is 1
        for grib_file in tqdm(grib2_files, desc="Converting MRMS to GeoTIFF"):
            base_name, error = _process_single_file(grib_file, output_folder, basin_clipping, expanded_bounds)
            if error:
                tqdm.write(f"Error processing {base_name}: {error}")
                failed_files += 1
    else:
        # Parallel processing using multiple processes
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(_process_single_file, grib_file, output_folder,
                                         basin_clipping, expanded_bounds): grib_file 
                       for grib_file in grib2_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting MRMS to GeoTIFF"):
                base_name, error = future.result()
                if error:
                    tqdm.write(f"Error processing {base_name}: {error}")
                    failed_files += 1
    
    # Post-process the generated TIFF files to ensure proper format (sequentially)
    tif_files = glob.glob(os.path.join(output_folder, '*.tif'))
    print(f"Ensuring proper format for {len(tif_files)} output files")
    
    for file_path in tqdm(tif_files, desc="Checking output formats"):
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                data = np.where((data > 1000) | (data < 0), -9999, data)
                data_float32 = data.astype('float32')
                meta = src.meta.copy()
                meta.update({'dtype': 'float32', 'nodata': -9999, 'compress': 'none'})
            with rasterio.open(file_path, 'w', **meta) as dst:
                dst.write(data_float32, 1)
        except Exception as e:
            print(f"Error formatting {os.path.basename(file_path)}: {str(e)[:100]}...")
    
    print(f"MRMS conversion completed. Output files saved to {os.path.abspath(output_folder)}")
    if failed_files > 0:
        print(f"Note: {failed_files} files failed to process")

def process_pet_bil_to_tif(input_folder='../PET_data', output_folder='../CREST_input/PET/'):
    """
    Process PET BIL files to GeoTIFF format.
    
    Args:
        input_folder (str): Path to the folder containing PET BIL files
        output_folder (str): Path to save the output GeoTIFF files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all BIL files in the input folder
    bil_files = glob.glob(os.path.join(input_folder, '*.bil'))
    
    if not bil_files:
        print(f"No BIL files found in {input_folder}")
        return
    
    print(f"Processing {len(bil_files)} PET BIL files to GeoTIFF format")
    print(f"Input folder: {os.path.abspath(input_folder)}")
    print(f"Output folder: {os.path.abspath(output_folder)}")
    
    failed_files = 0
    
    # Process each BIL file
    with tqdm(total=len(bil_files), desc="Converting PET to GeoTIFF") as pbar:
        for bil_file in bil_files:
            # Get the base filename without extension
            base_name = os.path.basename(bil_file)
            output_name = os.path.splitext(base_name)[0] + '.tif'
            output_path = os.path.join(output_folder, output_name)
            
            try:
                # Open the BIL file with rasterio
                with rasterio.open(bil_file) as src:
                    # Read the data
                    data = src.read(1)
                    
                    # Set NaN values to -9999
                    data = np.where(np.isnan(data), -9999, data)
                    data_float32 = data.astype(np.float32)
                    
                    # Create a new GeoTIFF with the same metadata but float32 dtype
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'driver': 'GTiff',
                        'dtype': 'float32',
                        'nodata': -9999
                    })
                    
                    # Write the output file
                    with rasterio.open(output_path, 'w', **kwargs) as dst:
                        dst.write(data_float32, 1)
                
            except Exception as e:
                failed_files += 1
                tqdm.write(f"Error processing {base_name}: {str(e)[:100]}...")
            
            pbar.update(1)
    
    print(f"PET conversion completed. Output files saved to {os.path.abspath(output_folder)}")
    if failed_files > 0:
        print(f"Note: {failed_files} files failed to process")

def clip_tif_by_shapefile(tif_path, output_path, shp_path):
    """
    Clip a GeoTIFF file to the bounding box of a shapefile.
    
    Parameters:
    -----------
    tif_path : str
        Path to the input GeoTIFF file
    output_path : str
        Path where the clipped GeoTIFF will be saved
    shp_path : str
        Path to the shapefile
    

    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read the shapefile and get its bounding box
        gdf = gpd.read_file(shp_path)
        
        # Get the bounding box (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = gdf.total_bounds
        
        # Open the GeoTIFF file
        with rasterio.open(tif_path) as src:
            # Create a window from the bounding box
            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            
            # Read the data within the window (all bands)
            data = src.read(window=window)
            
            # Calculate the new transform for the clipped raster
            out_transform = src.window_transform(window)
            
            # Copy the metadata and update with new dimensions and transform
            out_meta = src.meta.copy()
            out_meta.update({
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": out_transform,
                "dtype": 'float32',  # Ensure output is float32
                "compress": 'deflate'
            })
            
            # Convert data to float32 if it's not already
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Write the clipped data to a new GeoTIFF
            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(data)
                
        
    except Exception as e:
        tqdm.write(f"Error clipping {os.path.basename(tif_path)}: {str(e)[:100]}...")

def batch_clip_tifs_by_shapefile(input_dir='../BasicData', output_dir='../BasicData_Clip', shp_path='../shpFile/WBDHU12_CobbFort_sub2.shp'):
    """
    Batch process all TIF files in the input directory,
    clipping them to the bounding box of the specified shapefile.
    
    Parameters:
    -----------
    input_dir : str, optional
        Directory containing TIF files to clip
    output_dir : str, optional
        Directory to save clipped files
    shp_path : str, optional
        Path to the shapefile
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TIF files in the input directory
    tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return
    
    print(f"Clipping {len(tif_files)} TIF files to basin boundary defined in {os.path.basename(shp_path)}")
    print(f"Input folder: {os.path.abspath(input_dir)}")
    print(f"Output folder: {os.path.abspath(output_dir)}")
    print(f"Shapefile: {os.path.abspath(shp_path)}")
    
    # Process each TIF file with a progress bar
    successful = 0
    
    with tqdm(total=len(tif_files), desc="Clipping TIF files") as pbar:
        for tif_file in tif_files:
            # Get the base filename
            base_name = os.path.basename(tif_file)
            if '_' in base_name:
                base_name = base_name.split('_', 1)[1]  # Split at first '_' and keep the second part
            # Add '_clip' suffix to the filename before the extension
            base_name_without_ext, ext = os.path.splitext(base_name)
            output_name = f"{base_name_without_ext}_clip{ext}"
            output_path = os.path.join(output_dir, output_name)
            
            # Clip the TIF file
            if clip_tif_by_shapefile(tif_file, output_path, shp_path):
                successful += 1
            
            pbar.update(1)
    
    print(f"Clipping completed. {successful} of {len(tif_files)} files processed successfully.")
    print(f"Output files saved to {os.path.abspath(output_dir)}")

def download_usgs_data(site_code='07325850', start_date=None, end_date=None, output_dir='../USGS_gauge/'):
    """
    Download discharge data from USGS station and save as CSV file
    
    Parameters:
        site_code (str): USGS station ID
        start_date (datetime): Start date and time
        end_date (datetime): End date and time
        output_dir (str): Output directory path
        
    Returns:
        DataFrame or None: The downloaded discharge data, or None if download failed
    """
    try:
        import dataretrieval.nwis as nwis
    except ImportError:
        print("Error: dataretrieval package not found. Please install with 'pip install dataretrieval'")
        return None
    
    if start_date is None or end_date is None:
        print("Error: start_date and end_date must be provided as datetime objects")
        return None
    
    print(f"Downloading USGS discharge data for station {site_code} from {start_date} to {end_date}")
    print(f"Files will be saved to: {os.path.abspath(output_dir)}")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Format dates for NWIS service
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Call NWIS service to download discharge data
    df = nwis.get_record(
        sites=site_code,
        service='iv',             # 'iv' -> instantaneous values
        start=start_date_str,
        end=end_date_str,
        parameterCd='00060'
    )
    # Convert time to UTC
    df = df.tz_convert('UTC')

    if not df.empty:
        # Extract discharge data column and convert units (from cfs to m³/s, conversion factor: 0.0283)
        discharge_cols = [col for col in df.columns if '00060' in col and 'cd' not in col]
        if discharge_cols:
            discharge_col = discharge_cols[0]
            
            # Create result DataFrame - use copy instead of reference
            result_df = pd.DataFrame()
            result_df['datetime'] = df.index.copy()
            
            # Use values directly instead of column reference
            discharge_values = df[discharge_col].values * 0.0283  # Convert cfs to m³/s
            result_df['discharge'] = discharge_values
            
            # If NaN values still exist, try direct loop assignment
            if result_df['discharge'].isna().any():
                print("NaN values detected, trying direct loop assignment...")
                new_discharge = []
                for val in df[discharge_col].values:
                    new_discharge.append(val * 0.0283 if pd.notna(val) else val)
                result_df['discharge'] = new_discharge
            
            # Save as CSV file
            output_file = os.path.join(output_dir, f'USGS_{site_code}_UTC_m3s.csv')
            result_df.to_csv(output_file, index=False, float_format='%.6f')  # Specify float format
            print(f'Successfully downloaded data for station {site_code} and saved to {output_file}')
                
            return result_df
        else:
            print(f"Error: No discharge column found for station {site_code}")
    else:
        print(f'No data available for station {site_code}')
    
    return None



def visualize_mrms(date_for_visualization, data_path):
    """
    Visualize MRMS precipitation data for a specific date.
    
    Parameters:
    -----------
    date_for_visualization : datetime
        The date and time to visualize
    data_path : str
        Path to the MRMS data folder
        
    Returns:
    --------
    bool: True if visualization was successful, False otherwise
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from rasterio.plot import show
    import matplotlib.colors as colors
    import matplotlib.ticker as mticker

    # Format the MRMS filename based on the visualization date
    formatted_date = date_for_visualization.strftime('%Y%m%d-%H%M%S')
    hour_str = date_for_visualization.strftime('%H')
    mrms_tif_name = f'MultiSensor_QPE_01H_Pass2_00.00_{formatted_date[:8]}-{hour_str}0000.tif'
    
    mrms_tif_path = os.path.join(data_path, mrms_tif_name)
    
    if os.path.exists(mrms_tif_path):
        print(f"Visualizing MRMS precipitation data from: {mrms_tif_path}")
        # Open the raster file
        with rasterio.open(mrms_tif_path) as src:
            mrms_data = src.read(1)  # Read the first band
            transform = src.transform  # Get the transform from the source
            
            # Get the center coordinates of the raster
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            
            # Create a figure for visualization with geographic features
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Set extent to 10° x 10° around the center of the raster
            ax.set_extent([center_lon - 5, center_lon + 5, center_lat - 5, center_lat + 5], crs=ccrs.PlateCarree())
            
            # Add geographic features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.STATES, linestyle=':')
            
            # Create a custom colormap with light gray for zero values
            cmap = plt.cm.Blues.copy()
            cmap.set_under('lightgray')  # Set color for values below vmin
            
            # Show the raster data with custom colormap
            # Use vmin slightly above zero to ensure zero values are colored light gray
            show(mrms_data, ax=ax, cmap=cmap, transform=transform, vmin=0.01)
            
            # Add gridlines with latitude and longitude labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = mticker.MultipleLocator(2)
            gl.ylocator = mticker.MultipleLocator(2)
            gl.xformatter = mticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x == int(x) else "")
            gl.yformatter = mticker.FuncFormatter(lambda y, pos: f"{int(y)}" if y == int(y) else "")
            
            plt.colorbar(ax.images[0], label='Precipitation (mm)', shrink=0.5, extend='min')
            plt.title(f'MRMS Precipitation - {formatted_date}')
            plt.show()
            # return True
    else:
        print(f"MRMS file not found: {mrms_tif_path}")
        # return False


def visualize_pet(date_for_visualization, data_path):
    """
    Visualize PET (Potential Evapotranspiration) data for a specific date.
    
    Parameters:
    -----------
    date_for_visualization : datetime
        The date to visualize
    data_path : str
        Path to the PET data folder
        
    Returns:
    --------
    bool: True if visualization was successful, False otherwise
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from rasterio.plot import show
    # Format the PET filename based on the visualization date
    pet_tif_name = f'et{date_for_visualization.strftime("%y%m%d")}.tif'
    pet_tif_path = os.path.join(data_path, pet_tif_name)
    
    if os.path.exists(pet_tif_path):
        print(f"Visualizing PET data from: {pet_tif_path}")
        # Open the raster file
        with rasterio.open(pet_tif_path) as src:
            pet_data = src.read(1)  # Read the first band
            # Divide by 100 as the source file contains values that are 100x the actual values
            pet_data = pet_data / 100.0
            transform = src.transform  # Get the transform from the source
            
            # Create a figure for visualization with geographic features
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Add geographic features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.STATES, linestyle=':')
            
            # Show the raster data - use the transform from the source file
            show(pet_data, ax=ax, cmap='YlOrRd', transform=transform)
            
            plt.colorbar(ax.images[0], label='PET (mm/day)', shrink=0.5)
            plt.title(f'Potential Evapotranspiration - {date_for_visualization.strftime("%Y-%m-%d")}')
            plt.show()
    else:
        print(f"PET file not found: {pet_tif_path}")

# Visualize the clipped data with basin boundary overlay
def visualize_clipped_data_with_basin(clip_data_folder, basin_shp_path):
    """
    Visualize clipped raster data with basin boundary overlay.
    
    Parameters:
    -----------
    clip_data_folder : str
        Path to the folder containing clipped raster files
    basin_shp_path : str
        Path to the basin shapefile
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from rasterio.plot import show
    import matplotlib.colors as colors
    import matplotlib.patches as mpatches
    
    # Read the basin shapefile
    basin_gdf = gpd.read_file(basin_shp_path)
    
    # Get all TIF files in the clipped data folder
    tif_files = glob.glob(os.path.join(clip_data_folder, '*.tif'))
    
    if not tif_files:
        print(f"No TIF files found in {clip_data_folder}")
        return False
    
    print(f"Found {len(tif_files)} TIF files in {clip_data_folder}")
    
    # Map file names to descriptive titles and custom colormaps
    file_info = {
        'facc_clip.tif': {
            'title': 'Flow Accumulation Map (FAM)',
            'cmap': 'Blues',
            'norm': colors.LogNorm()  # Logarithmic scale for flow accumulation
        },
        'dem_clip.tif': {
            'title': 'Digital Elevation Model (DEM)',
            'cmap': 'terrain',
            'norm': None  # Linear scale for elevation
        },
        'fdir_clip.tif': {
            'title': 'Drainage Direction Map (DDM)',
            'cmap': None,  # Will be set to custom colormap for directions
            'norm': None
        }
    }
    
    # Display up to 3 TIF files with basin boundary in a single row
    if len(tif_files) > 0:
        # Create a figure with subplots in a single row with reduced spacing
        fig, axes = plt.subplots(1, min(3, len(tif_files)), figsize=(15, 5), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Reduce horizontal spacing between subplots
        plt.subplots_adjust(wspace=0)
        
        # If only one file, axes won't be an array
        if len(tif_files) == 1:
            axes = [axes]
        
        # Process each file and plot in the corresponding subplot
        for i, (tif_file, ax) in enumerate(zip(tif_files[:3], axes)):
            file_name = os.path.basename(tif_file)
            print(f"Visualizing: {file_name}")
            
            with rasterio.open(tif_file) as src:
                data = src.read(1)
                transform = src.transform
                
                # Add geographic features
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.STATES, linestyle=':')
                
                # Get custom visualization settings for this file
                file_settings = file_info.get(file_name, {
                    'title': file_name,
                    'cmap': 'viridis',
                    'norm': None
                })
                
                # Special handling for flow direction map to use discrete colors
                if file_name == 'fdir_clip.tif':
                    # Get unique values in the direction data
                    unique_values = np.unique(data)
                    unique_values = unique_values[~np.isnan(unique_values)]
                    
                    # Create a custom colormap for the unique direction values
                    n_values = len(unique_values)
                    colors_list = plt.cm.tab10(np.linspace(0, 1, n_values))
                    
                    # Create a custom discrete colormap
                    cmap = colors.ListedColormap(colors_list)
                    bounds = np.concatenate([unique_values - 0.5, [unique_values[-1] + 0.5]])
                    norm = colors.BoundaryNorm(bounds, cmap.N)
                    
                    # Show the raster with discrete colors
                    img = show(data, ax=ax, transform=transform, cmap=cmap, norm=norm)
                    
                    # Create a legend for direction values
                    legend_patches = []
                    for i, val in enumerate(unique_values):
                        patch = mpatches.Patch(color=colors_list[i], label=f'Direction {int(val)}')
                        legend_patches.append(patch)
                    
                    # Add the legend
                    ax.legend(handles=legend_patches, loc='lower right', fontsize='small')
                else:
                    # Show other raster data with standard colormap
                    img = show(data, ax=ax, transform=transform, 
                              cmap=file_settings['cmap'], norm=file_settings['norm'])
                    
                    # Fix colorbar issue by directly creating it from the image
                    if hasattr(img, 'get_images') and img.get_images():
                        # For matplotlib 3.5+
                        cbar = plt.colorbar(img.get_images()[0], ax=ax, shrink=0.7)
                    elif hasattr(img, 'images') and img.images:
                        # For older matplotlib versions
                        cbar = plt.colorbar(img.images[0], ax=ax, shrink=0.7)
                    
                    # Set appropriate colorbar label
                    if file_name == 'facc_clip.tif':
                        cbar.set_label('Flow Accumulation (log scale)')
                    elif file_name == 'dem_clip.tif':
                        cbar.set_label('Elevation (m)')
                
                # Add basin boundary with black color
                basin_gdf.boundary.plot(ax=ax, color='black', linewidth=2)
                
                # Add title using the mapping
                ax.set_title(file_settings['title'], fontsize=10)
        
        plt.tight_layout()
        plt.show()
# Visualize the USGS data
def visualize_usgs_data(site_code, df_usgs):
    """
    Visualize USGS gauge discharge data.
    
    Parameters:
    -----------
    site_code : str
        USGS gauge site code
    df_usgs : pandas.DataFrame
        DataFrame containing USGS discharge data with 'datetime' and 'discharge' columns
        
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df_usgs['datetime'], df_usgs['discharge'], 'b-', linewidth=1.5)
    plt.title(f'USGS Gauge {site_code} Discharge', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Discharge (m³/s)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()

def visualize_model_results(ts_file='../Output/ts.07325850.crest.csv', show_plot=True, save_path=None):
    """
    Visualize hydrological model results comparing simulated vs observed discharge.
    
    Parameters:
    -----------
    ts_file : str, optional
        Path to the time series CSV file with model results
    show_plot : bool, optional
        Whether to display the plot (default: True)
    save_path : str, optional
        Path to save the plot image (default: None, plot is not saved)
    """
    
    
    # Check if file exists
    if not os.path.exists(ts_file):
        print(f"Error: Results file not found at {ts_file}")
        return False
        
    # Read the CSV file
    try:
        df = pd.read_csv(ts_file)
    except Exception as e:
        print(f"Error reading results file: {str(e)}")
        return False
    
    print(f"Visualizing model results from: {os.path.abspath(ts_file)}")
    
    # Create the figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot discharge on left y-axis
    ax1.plot(df['Time'], df['Discharge(m^3 s^-1)'], label='Simulated', linewidth=1)
    ax1.plot(df['Time'], df['Observed(m^3 s^-1)'], label='Observed', linewidth=1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Discharge (m³/s)')
    # Set y-axis limit to 2.1 times the maximum observed discharge value
    max_observed = df['Observed(m^3 s^-1)'].max()
    ax1.set_ylim(0, max_observed * 2.1)

    # Plot precipitation on right y-axis (inverted)
    ax2.plot(df['Time'], df['Precip(mm h^-1)'], color='blue', alpha=0.3, label='Precipitation')
    ax2.set_ylabel('Precipitation (mm/h)')
    ax2.invert_yaxis()  # Invert the y-axis so 0 is at top
    # Set y-axis limit to 2.1 times the maximum precipitation value (inverted)
    max_precip = df['Precip(mm h^-1)'].max()
    ax2.set_ylim(max_precip * 2.1, 0)  # Set inverted y-axis limits

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Set title
    plt.title('Simulated vs Observed Discharge with Precipitation')

    # Set x-axis limits to first and last time points
    ax1.set_xlim(df['Time'].iloc[0], df['Time'].iloc[-1])

    # Reduce x-axis density and rotate labels
    step = 24  # Show every 24th tick
    ax1.set_xticks(range(0, len(df), step))
    ax1.set_xticklabels([t.split()[0] for t in df['Time'][::step]], rotation=45, ha='right')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.abspath(save_path)}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
def evaluate_model_performance(ts_file='../Output/ts.07325850.crest.csv'):
    """
    Evaluate hydrological model performance by calculating statistical metrics
    between simulated and observed discharge.
    
    Parameters:
    -----------
    ts_file : str, optional
        Path to the time series CSV file with model results
        
    Returns:
    --------
    dict: Dictionary containing the calculated performance metrics
    """
    # Check if file exists
    if not os.path.exists(ts_file):
        print(f"Error: Results file not found at {ts_file}")
        return None
        
    # Read the CSV file
    try:
        df = pd.read_csv(ts_file)
    except Exception as e:
        print(f"Error reading results file: {str(e)}")
        return None
    
    print(f"Evaluating model performance from: {os.path.abspath(ts_file)}")
    
    # Extract simulated and observed discharge
    sim = df['Discharge(m^3 s^-1)'].values
    obs = df['Observed(m^3 s^-1)'].values
    
    # Remove any rows where either simulated or observed values are NaN
    valid_indices = ~(np.isnan(sim) | np.isnan(obs))
    sim = sim[valid_indices]
    obs = obs[valid_indices]
    
    if len(sim) == 0 or len(obs) == 0:
        print("Error: No valid data points after removing NaN values")
        return None
    
    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    
    # Calculate Bias (as percentage)
    bias = np.mean(sim - obs)
    bias_percent = (bias / np.mean(obs)) * 100
    
    # Calculate Correlation Coefficient (CC)
    cc = np.corrcoef(sim, obs)[0, 1]
    
    # Calculate Nash-Sutcliffe Coefficient of Efficiency (NSCE)
    mean_obs = np.mean(obs)
    nsce = 1 - (np.sum((sim - obs) ** 2) / np.sum((obs - mean_obs) ** 2))
    
    # Create a dictionary with the metrics
    metrics = {
        'RMSE': rmse,
        'Bias': bias,
        'Bias_percent': bias_percent,
        'CC': cc,
        'NSCE': nsce
    }
    
    # Print the metrics
    print("\nModel Performance Metrics:")
    print(f"RMSE: {rmse:.4f} m³/s")
    print(f"Bias: {bias:.4f} m³/s ({bias_percent:.2f}%)")
    print(f"CC: {cc:.4f}")
    print(f"NSCE: {nsce:.4f}")
    
    return metrics

def create_output_directory(output_dir='../Output'):
    """
    Create the output directory for model results if it doesn't exist.
    
    Parameters:
    -----------
    output_dir : str, optional
        Path to the output directory (default: '../Output')
        
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {os.path.abspath(output_dir)}")
        else:
            print(f"Output directory already exists: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")


# ======================================
# Basin Selection and Processing Functions
# ======================================

import requests
import geopandas as gpd
import folium
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import os

def download_watershed_shp(latitude, longitude, output_path, level=5):
    """
    Download watershed boundary data for a given latitude and longitude.
    
    Parameters:
    -----------
    latitude : float
        Latitude of the point of interest
    longitude : float
        Longitude of the point of interest
    output_path : str
        Directory path where output files will be saved
    level : int, default=5
        WBD level to query (5=HUC8, 4=HUC6, etc.)
    
    Returns:
    --------
    Basin_Area : float
        Area of the watershed in square kilometers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Step 1: Query WBD API to find the watershed containing this point
    wbd_url = f"https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/{level}/query"
    params = {
        'geometry': f'{longitude},{latitude}', # Note: longitude comes first
        'geometryType': 'esriGeometryPoint',
        'inSR': '4326',
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': '*',
        'f': 'geojson'
    }

    response = requests.get(wbd_url, params=params)
    data = response.json()

    # Step 2: Save as local GeoJSON and read with GeoPandas
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    # Set CRS to match the input data (WGS 84 / EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    gdf['shape_Area'] = gdf['shape_Area'] / 1000000
    print('Basin Area (km2):')
    Basin_Area = round(gdf['shape_Area'][0], 2)
    print(Basin_Area)

    # Rename columns to fit 10-character limit
    column_rename_dict = {
        'shape_Length': 'shp_length',
        'metasourceid': 'metasource',
        'sourcedatadesc': 'sourcedata',
        'sourceoriginator': 'sourceorig',
        'sourcefeatureid': 'sourcefeat',
        'referencegnis_ids': 'ref_gnis'
    }

    # Apply the renaming
    gdf = gdf.rename(columns=column_rename_dict)

    # Save to shapefile
    shp_path = os.path.join(output_path, f'Basin_selected_{level}.shp')
    gdf.to_file(shp_path)

    # Step 3: Visualize using both folium (interactive) and matplotlib (static)
    # Create folium map centered on input point
    m = folium.Map(location=[latitude, longitude], zoom_start=8)

    # Add watershed polygon with style
    folium.GeoJson(
        gdf,
        style_function=lambda x: {
            'fillColor': '#ffaf00',
            'color': 'red',
            'weight': 2,
            'fillOpacity': 0.3
        }
    ).add_to(m)

    # Add input point marker
    folium.Marker(location=[latitude, longitude], popup="Input Location").add_to(m)

    # Save interactive map
    html_path = os.path.join(output_path, f'Basin_selected_{level}.html')
    m.save(html_path)

    # Create static map with matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))

    # Reproject to Web Mercator for basemap compatibility
    gdf_web = gdf.to_crs(epsg=3857)

    # Plot watershed boundary first
    gdf_web.plot(ax=ax, alpha=0.5, edgecolor='red', facecolor='yellow', linewidth=2)

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Add point location
    point = Point(longitude, latitude)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs='EPSG:4326').to_crs(epsg=3857)
    point_gdf.plot(ax=ax, color='red', marker='*', markersize=100)

    # Set map extent to focus on the watershed
    ax.set_xlim(gdf_web.total_bounds[[0, 2]])
    ax.set_ylim(gdf_web.total_bounds[[1, 3]])

    # Remove axes
    ax.set_axis_off()

    plt.title(f'Watershed Boundary (Level {level}) with Basemap')
    plt.show()

    print(f"Done. Check {html_path} for interactive view and see the static map above!")
    
    return Basin_Area

def plot_watershed_with_gauges(basin_shp_path, gauge_meta_path):
    """
    Plot watershed with USGS gauge stations.
    
    Parameters:
    -----------
    basin_shp_path : str
        Path to the watershed shapefile
    gauge_meta_path : str
        Path to the USGS gauge metadata CSV file
    
    Returns:
    --------
    None
    """
    # Load the watershed shapefile
    watershed_shp = basin_shp_path
    gdf_web = gpd.read_file(watershed_shp).to_crs(epsg=3857)
    
    # Calculate centroid in a projected CRS first, then convert to WGS84 (EPSG:4326)
    # This avoids the warning about calculating centroids in geographic CRS
    centroid = gdf_web.geometry.union_all().centroid
    center_point = gpd.GeoDataFrame(geometry=[centroid], crs='EPSG:3857').to_crs(epsg=4326)
    center_lat = center_point.geometry.y[0]
    center_lng = center_point.geometry.x[0]
    
    # Create interactive map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=8)
    
    # Add watershed boundary to the map
    folium.GeoJson(
        gdf_web.to_crs(epsg=4326),
        name='Watershed Boundary',
        style_function=lambda x: {'fillColor': 'yellow', 'color': 'red', 'weight': 2, 'fillOpacity': 0.5}
    ).add_to(m)
    
    # Load USGS gauge information
    gauge_info = pd.read_csv(gauge_meta_path)

    # Convert gauge locations to GeoDataFrame
    gauge_points = gpd.GeoDataFrame(
        gauge_info,
        geometry=gpd.points_from_xy(gauge_info.LNG_GAGE, gauge_info.LAT_GAGE),
        crs='EPSG:4326'
    )

    # Reproject to match the watershed CRS (Web Mercator)
    gauge_points = gauge_points.to_crs(epsg=3857)

    # Spatial join to find gauges within the watershed
    gauges_in_basin = gpd.sjoin(gauge_points, gdf_web, how='inner', predicate='within')

    # Create a new figure for this plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Replot the watershed boundary
    gdf_web.plot(ax=ax, alpha=0.5, edgecolor='red', facecolor='yellow', linewidth=2)

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Get the point from the watershed centroid
    point = Point(center_lng, center_lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs='EPSG:4326').to_crs(epsg=3857)
    point_gdf.plot(ax=ax, color='red', marker='*', markersize=100)

    if len(gauges_in_basin) > 0:
        # Plot gauge locations on the static map
        gauge_plot = gauges_in_basin.plot(
            ax=ax,
            color='blue',
            marker='^',
            markersize=100,
            label='USGS Gauges'
        )
        
        # Add station IDs as labels
        for idx, row in gauges_in_basin.iterrows():
            # Pad station ID with leading zeros to ensure 8 digits
            padded_staid = str(row['STAID']).zfill(8)
            ax.annotate(
                padded_staid,
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, 5),
                textcoords='offset points',
                color='blue',
                fontsize=10,
                fontweight='bold'
            )
        
        # Add gauges to the interactive map
        for idx, row in gauges_in_basin.iterrows():
            # Pad station ID with leading zeros to ensure 8 digits
            padded_staid = str(row['STAID']).zfill(8)
            folium.Marker(
                location=[row['LAT_GAGE'], row['LNG_GAGE']],
                popup=f"Station ID: {padded_staid}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
        
        # Update the map save
        html_path = 'basin_map_with_gauges.html'
        m.save(html_path)
        
        # Set map extent to focus on the watershed
        ax.set_xlim(gdf_web.total_bounds[[0, 2]])
        ax.set_ylim(gdf_web.total_bounds[[1, 3]])
        
        # Remove axes
        ax.set_axis_off()
        
        plt.title('Watershed Boundary with USGS Gauges')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Make sure to display the plot
        plt.tight_layout()
        plt.show()
        
        # Print all station IDs and names
        print("\nUSGS Gauge Stations in the Watershed:")
        print("--------------------------------------")
        for idx, row in gauges_in_basin.iterrows():
            # Pad station ID with leading zeros to ensure 8 digits
            padded_staid = str(row['STAID']).zfill(8)
            print(f"Station ID: {padded_staid}, Name: {row['STANAME']}, Latitude: {row['LAT_GAGE']:.2f}, Longitude: {row['LNG_GAGE']:.2f}")
    else:
        # Set map extent to focus on the watershed
        ax.set_xlim(gdf_web.total_bounds[[0, 2]])
        ax.set_ylim(gdf_web.total_bounds[[1, 3]])
        
        # Remove axes
        ax.set_axis_off()
        
        plt.title('Watershed Boundary (No USGS Gauges Found)')
        plt.show()
        
        # Save the map even if no gauges found
        html_path = 'basin_map_with_gauges.html'
        m.save(html_path)
        
        print("No USGS gauge stations found within the watershed boundary.")
        print(f"Interactive map saved to {html_path}")


def get_gauge_coordinates(gauge_meta_path, station_id):
    """
    Get the latitude and longitude of a USGS gauge station by its ID.
    
    Parameters:
    -----------
    gauge_meta_path : str
        Path to the CSV file containing gauge metadata
    station_id : str or int
        The station ID (will be converted to integer to remove leading zeros)
    
    Returns:
    --------
    tuple
        (latitude, longitude) of the gauge station, or None if not found
    """
    import pandas as pd
    
    # Convert station_id to integer to handle leading zeros
    try:
        station_id_int = int(station_id)
    except ValueError:
        print(f"Error: Station ID '{station_id}' is not a valid number")
        return None
    
    # Read the gauge metadata
    try:
        gauge_meta = pd.read_csv(gauge_meta_path)
    except Exception as e:
        print(f"Error reading gauge metadata file: {e}")
        return None
    
    # Find the station in the metadata
    station = gauge_meta[gauge_meta['STAID'] == station_id_int]
    
    if len(station) == 0:
        print(f"Station ID {station_id} not found in metadata")
        return None
    
    # Return the coordinates
    lat = station['LAT_GAGE'].values[0]
    lon = station['LNG_GAGE'].values[0]
    
    return (lat, lon)

def download_hydrosheds_data(latitude, longitude, dest_folder="../BasicData"):
    """
    Download and process HydroSHEDS data based on coordinates.
    
    Parameters:
    -----------
    latitude : float
        Latitude of the gauge station
    longitude : float
        Longitude of the gauge station
    dest_folder : str, optional
        Destination folder for downloaded data (default: "../BasicData")
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    import os
    import requests
    from zipfile import ZipFile
    import glob
    import shutil
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created directory: {dest_folder}")
    else:
        # Clear all files in the destination folder before downloading
        for file_path in glob.glob(os.path.join(dest_folder, '*')):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"Cleared all existing files in {dest_folder}")
    
    # Determine the tile prefix based on coordinates
    def get_tile_prefix(lat, lon):
        # For latitude: N20 means 20N-30N
        lat_prefix = f"N{int(lat) // 10 * 10}"
        
        # For longitude: W100 means 100W-90W
        # We need absolute value and then round down to nearest 10
        lon_abs = abs(lon)
        # For longitude: W100 means 100W-90W
        # Need to round up to the nearest 10 for the correct tile
        lon_prefix = f"W{(int(lon_abs) // 10 + 1) * 10}"
        
        return f"{lat_prefix}{lon_prefix}"
    
    # Get the tile prefix based on gauge coordinates
    tile_prefix = get_tile_prefix(latitude, longitude)
    print(f"Determined tile prefix: {tile_prefix}")
    
    # Define the URLs for the three datasets
    urls = [
        f"https://data.hydrosheds.org/file/hydrosheds-v1-acc/na_acc_3s/{tile_prefix}_acc.zip",
        f"https://data.hydrosheds.org/file/hydrosheds-v1-con/na_con_3s/{tile_prefix.lower()}_con.zip",
        f"https://data.hydrosheds.org/file/hydrosheds-v1-dir/na_dir_3s/{tile_prefix.lower()}_dir.zip"
    ]
    
    # Download and process each file
    for url in urls:
        print(f"Downloading from {url}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            # Extract the filename from the URL
            filename = url.split('/')[-1]
            filepath = os.path.join(dest_folder, filename)
            
            # Save the zip file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename} to {dest_folder}")
            
            # Extract the contents
            with ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(dest_folder)
            print(f"Extracted contents to {dest_folder}")
            
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
            return False
    
    # Clean up by removing the zip files
    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(dest_folder, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Removed zip file: {filepath}")
    
    # Rename the extracted files according to a more standardized naming convention
    # Find all extracted files in the destination folder
    extracted_files = [f for f in os.listdir(dest_folder) if f.endswith('.tif')]
    
    for file in extracted_files:
        # Determine the file type (acc, con, or dir)
        if '_acc' in file.lower():
            new_name = "facc.tif"
        elif '_con' in file.lower():
            new_name = "dem.tif"
        elif '_dir' in file.lower():
            new_name = "fdir.tif"
        else:
            continue  # Skip if not one of the expected file types
        
        # Create the full paths
        old_path = os.path.join(dest_folder, file)
        new_path = os.path.join(dest_folder, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file} → {new_name}")
    
    return True

def generate_control_file(
    time_begin,
    time_end,
    basic_data_path,
    mrms_path,
    pet_path,
    gauge_id,
    gauge_lon,
    gauge_lat,
    usgs_data_path,
    basin_area,
    output_dir,
    wm,
    b,
    im,
    ke,
    fc,
    iwu,
    under,
    leaki,
    th,
    isu,
    alpha,
    beta,
    alpha0,
    
    control_file_path='control.txt',
    grid_on=False,

):
    """
    Generate a control.txt file for CREST model with variable parameters.
    
    Args:
        time_begin (datetime): Simulation start time
        time_end (datetime): Simulation end time
        basic_data_path (str): Path to basic data directory containing DEM, flow direction and flow accumulation
        mrms_path (str): Path to MRMS precipitation data directory
        pet_path (str): Path to PET data directory
        gauge_id (str): USGS gauge ID
        gauge_lon (float): Gauge longitude
        gauge_lat (float): Gauge latitude
        usgs_data_path (str): Path to USGS data directory
        basin_area (float): Basin area in square kilometers
        output_dir (str): Output directory path for model results
        wm (float): CREST parameter - Maximum water capacity
        b (float): CREST parameter - Exponent of the variable infiltration curve
        im (float): CREST parameter - Impervious area ratio
        ke (float): CREST parameter - Potential evapotranspiration adjustment factor
        fc (float): CREST parameter - Soil saturated hydraulic conductivity
        iwu (float): CREST parameter - Initial soil water content
        under (float): KW parameter - Overland runoff velocity multiplier
        leaki (float): KW parameter - Interflow reservoir discharge rate
        th (float): KW parameter - Overland flow velocity exponent
        isu (float): KW parameter - Initial value of overland reservoir
        alpha (float): KW parameter - Multiplier in channel velocity equation
        beta (float): KW parameter - Exponent in channel velocity equation
        alpha0 (float): KW parameter - Base flow velocity
        control_file_path (str): Path to save the control file
        grid_on (bool): Whether to output grid files for streamflow
        
    Returns:
        str: Absolute path to the generated control file
    """
    # Convert all paths to absolute paths
    basic_data_path = os.path.abspath(basic_data_path)
    mrms_path = os.path.abspath(mrms_path)
    pet_path = os.path.abspath(pet_path)
    usgs_data_path = os.path.abspath(usgs_data_path)
    output_dir = os.path.abspath(output_dir)
    
    # Prepare the Task Simu section with optional output_grids parameter
    task_simu = """[Task Simu]
STYLE=SIMU
MODEL=CREST
ROUTING=KW
BASIN=0
PRECIP=MRMS
PET=PET
OUTPUT={output_dir}
PARAM_SET=CrestParam
ROUTING_PARAM_Set=KWParam
TIMESTEP=1h
"""
    
    # Add OUTPUT_GRIDS parameter if grid_on is True
    if grid_on:
        task_simu += "OUTPUT_GRIDS=STREAMFLOW\n"
    
    task_simu += """
TIME_BEGIN={time_begin}
TIME_END={time_end}
"""
    
    control_content = f"""[Basic]
DEM={basic_data_path}/dem_clip.tif
DDM={basic_data_path}/fdir_clip.tif
FAM={basic_data_path}/facc_clip.tif

PROJ=geographic
ESRIDDM=true
SelfFAM=true

[PrecipForcing MRMS]
TYPE=TIF
UNIT=mm/h
FREQ=1h
LOC={mrms_path}
NAME=MultiSensor_QPE_01H_Pass2_00.00_YYYYMMDD-HH0000.tif

[PETForcing PET]
TYPE=TIF
UNIT=mm/100d
FREQ=d
LOC={pet_path}
NAME=et{time_begin.strftime('%y')}MMDD.tif

[Gauge {gauge_id}] 
LON={gauge_lon}
LAT={gauge_lat}
OBS={usgs_data_path}/USGS_{gauge_id}_UTC_m3s.csv
OUTPUTTS=TRUE
BASINAREA={basin_area}
WANTCO=TRUE

[Basin 0]
GAUGE={gauge_id}

[CrestParamSet CrestParam]
gauge={gauge_id}
wm={wm}
b={b}
im={im}
ke={ke}
fc={fc}
iwu={iwu}

[kwparamset KWParam]
gauge={gauge_id}
under={under}
leaki={leaki}
th={th}
isu={isu}
alpha={alpha}
beta={beta}
alpha0={alpha0}

{task_simu.format(
    output_dir=output_dir,
    time_begin=time_begin.strftime('%Y%m%d%H%M'),
    time_end=time_end.strftime('%Y%m%d%H%M')
)}

[Execute]
TASK=Simu
"""

    # Write the content to the control file
    with open(control_file_path, 'w') as f:
        f.write(control_content)
    
    # Return the absolute path of the control file
    return os.path.abspath(control_file_path)
