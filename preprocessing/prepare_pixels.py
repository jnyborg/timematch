from argparse import ArgumentParser
from glob import glob
from itertools import product
import os
from pathlib import Path
import pickle
from typing import List

import cv2
import geopandas as gpd
import mahotas as mh
import numpy as np
import rasterio
from rasterio.features import rasterize
import rasterio.mask
import rasterio.mask
from shapely.geometry import box
from tqdm import tqdm
import zarr

from preprocessing.downloader import S2Image, s2_download
from utils import label_utils



def prepare_pixels(config):
    s2_images: List[S2Image] = s2_download(
        config.tile,
        config.start_date,
        config.end_date,
        config.download_dir,
        min_coverage=config.min_coverage,
        max_cloudy_pct=config.max_cloudy_percentage,
        sort_by_date=True,
        bands=config.bands,
        data_collection=config.data_collection,
        # previews=True,
    )
    # Load bounding box and transform information for chosen S2 tile
    with rasterio.open(s2_images[0].bands_10m[0]) as f:
        tile_bbox = gpd.GeoSeries(box(*f.bounds))
        tile_bbox.crs = f.crs
        tile_transform = f.transform
        tile_crs = f.crs

    cloudy_percentages = [s2_image.cloudy_pct for s2_image in s2_images]

    # Group parcels by spatial blocks to ensure independence across years
    blocks = create_blocks(tile_bbox, config)

    # Load parcels from shapefile
    parcels = get_ground_truth_parcels(tile_bbox, config, blocks)
    # top_100_labels(parcels, config)

    # Create bounding boxes and masks for each parcel
    bboxes, masks = compute_bboxes_and_masks(parcels, tile_transform)
    n_pixels = [mask.sum().astype(np.int32) for mask in masks]

    # Remove polygons with 0 pixels, if any
    valid_indices = [idx for idx, S in enumerate(n_pixels) if S != 0]
    print(f'Remaining parcels after removing parcels with no pixels: {len(valid_indices)}/{len(masks)}')
    parcels = parcels.iloc[valid_indices]
    parcels = parcels.reset_index()
    bboxes = [bboxes[i] for i in valid_indices]
    masks = [masks[i] for i in valid_indices]
    n_pixels = [n_pixels[i] for i in valid_indices]
    assert len(parcels) == len(bboxes) == len(masks) == len(n_pixels)

    # Write various metadata
    dates = [int(tile.yyyymmdd) for tile in s2_images]
    write_metadata(parcels, bboxes, masks, dates, n_pixels, config, tile_transform, tile_crs, cloudy_percentages)

    # Setup zarr files
    print('Initializing .zarr files...')
    zarr_files = []
    is_l2a = config.data_collection == 'l2a'
    for parcel_idx in range(len(parcels)):
        parcel_file = str(config.data_dir / f"{parcel_idx}.zarr")
        n_bands = len(config.bands)
        if is_l2a:  # append cloud mask as last band
            n_bands += 1
        pixel_set_shape = (len(dates), n_bands, n_pixels[parcel_idx])  # (T, C, S)
        z = zarr.open(parcel_file, mode='a', shape=pixel_set_shape, dtype=np.uint16)
        zarr_files.append(z)

    # Load step_size S2 samples at a time, extract pixels for each parcel and write to zarr files.
    step_size = config.buffer_size
    for img_idx in tqdm(range(0, len(s2_images), step_size), desc='processing s2 image'):
        t0, t1 = img_idx, img_idx + step_size
        image = np.stack([read_s2_image(s2_image, tile_transform, is_l2a=is_l2a) for s2_image in tqdm(s2_images[t0:t1], desc='reading S2 images to memory', leave=False)])

        for parcel_idx in tqdm(range(len(parcels)), desc='writing parcel pixels sets', leave=False):
            bbox, mask = bboxes[parcel_idx], masks[parcel_idx]
            pixels = image[..., bbox[0], bbox[1]][..., mask]  # (C, S)
            zarr_files[parcel_idx][t0:t1] = pixels
        del image


def compute_bboxes_and_masks(parcels, transform):
    print('Rasterizing polygons...')
    polygon_index = [(polygon, idx) for idx, polygon in enumerate(parcels.geometry, start=1)]  # 0 = background
    labeled = rasterize(polygon_index, out_shape=(10980, 10980), transform=transform)
    print('Computing bounding boxes and masks for each polygon...')
    bboxes = mh.labeled.bbox(labeled, as_slice=True)[1:]  # remove background bbox
    masks = [(labeled[bbox] == idx).astype(bool) for idx, bbox in enumerate(bboxes, start=1)]
    return bboxes, masks


def write_metadata(parcels, bboxes, masks, dates, n_pixels, config, tile_transform, tile_crs, cloudy_pct, write_geometry=False):
    print('Writing metadata...')
    metadata = {
        'start_date': config.start_date,
        'end_date': config.end_date,
        'dates': dates,
        'cloudy_pct': cloudy_pct,
        'parcels': [
            {
                'id': int(p[config.id_column]),
                'label': p[config.crop_code_column],
                'n_pixels': n_pixels[idx],
                'block': p['block'],
                'geometric_features': get_geometric_features(p, masks[idx]),
            } for idx, p in parcels.iterrows()
        ],
    }
    pickle.dump(metadata, open(config.meta_dir / 'metadata.pkl', 'wb'))

    if write_geometry:
        geometry = {
            'tile': config.tile,
            'tile_transform': tile_transform,
            'tile_crs': tile_crs,
            'parcels': [
                {
                    'id': p[config.id_column],
                    'bbox': bboxes[idx],
                    'mask': masks[idx],
                } for idx, p in parcels.iterrows()
            ],
        }
        pickle.dump(geometry, open(config.meta_dir / 'geometry.pkl', 'wb'))
    print('Done writing metadata.')


def get_geometric_features(parcel, mask):
    # Compute geometric features
    perimeter = parcel.geometry.length  # Parcel Perimeter (meters)
    area = parcel.geometry.area  # Parcel Area (squared meters)
    perimeter_area_ratio = perimeter / area  # Perimeter / Area
    cover_ratio = mask.sum() / mask.size  # n pixels in parcel / n pixels in bounding box
    return [perimeter, area, perimeter_area_ratio, cover_ratio]


def read_s2_image(s2_image: S2Image, tile_transform=None, is_l2a=False):
    x10 = np.empty(shape=(10980, 10980, len(s2_image.bands_10m)), dtype=np.uint16)
    x20 = np.empty(shape=(10980 // 2, 10980 // 2, len(s2_image.bands_20m)), dtype=np.uint16)

    if is_l2a:
        try:
            cloud_mask = gpd.read_file(s2_image.cloud_mask)
            cloud_mask = rasterize(cloud_mask.geometry, out_shape=(10980, 10980), transform=tile_transform)  # clouds will be set to 1
        except:  # there sometimes is no cloud mask if no clouds
            assert s2_image.cloudy_pct == 0.0
            cloud_mask = np.zeros((10980, 10980), dtype=np.uint16)
        cloud_mask = cloud_mask * (2**16-1)  # set clouds as max pixel value
        cloud_mask = cloud_mask[:, :, np.newaxis].astype(np.uint16)
    else:
        cloud_mask = None

    for band_idx, path in enumerate(s2_image.bands_10m):
        with rasterio.open(path) as f:
            x10[..., band_idx] = np.squeeze(f.read())

    for band_idx, path in enumerate(s2_image.bands_20m):
        with rasterio.open(path) as f:
            x20[..., band_idx] = np.squeeze(f.read())

    x20 = cv2.resize(x20, x10.shape[:2], interpolation=cv2.INTER_LINEAR)
    if is_l2a:
        image = np.concatenate([x10, x20, cloud_mask], axis=-1)
    else:
        image = np.concatenate([x10, x20], axis=-1)

    image = np.moveaxis(image, -1, 0)  # channels first

    return image


def top_100_labels(shapefile, config):
    df = shapefile.copy()
    if config.country == 'france':
        crop_codes = label_utils.get_codification_table('france')
        df['label'] = [crop_codes[code] for code in df[config.crop_code_column.lower()]]
        print(df['label'].value_counts().head(100).to_string())
    elif config.country == 'austria':
        crop_codes = label_utils.get_codification_table('austria')
        df['label'] = [crop_codes[str(int(code))] for code in df[config.crop_code_column.lower()]]
        print(df.columns)
        print(df[['snar_code', 'snar_bezei']].value_counts().head(1000).to_string())
        exit()
    else:
        crop_codes = label_utils.get_codification_table('denmark')
        df['label'] = [crop_codes[str(int(code))] for code in df[config.crop_code_column.lower()]]
        print(df['label'].value_counts().head(100).to_string())


def get_ground_truth_parcels(tile_bbox, config, blocks=None, erosion_m=20, min_area_sqm=10000):
    output_dir = config.meta_dir / 'parcels'
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Creating new label shape file...')
    gt_path = os.path.join(config.ground_truth_dir, config.country, config.year)
    gt_shape_file = None
    output_file = output_dir / f'parcels_{config.country}_{config.tile}_{config.year}.shp'
    if output_file.exists():
        print('Loading cached label shapefile')
        return gpd.read_file(output_file)

    for filetype in ['shp', 'gpkg']:
        res = glob(os.path.join(gt_path, f"*.{filetype}"))
        if len(res)>0:
            gt_shape_file = res[0]
            out_filetype = filetype
            break

    print('reading ground truth file...', gt_shape_file)
    df = gpd.read_file(gt_shape_file, bbox=tile_bbox)
    df = df.to_crs(tile_bbox.crs)
    # remove binary column in austria dataset
    if config.country == 'austria':
        df = df.drop(columns='GML_GEOM', errors='ignore')
    df.columns = df.columns.str.lower()
    print(df.columns)
    available_fields = len(df)



    if blocks is not None:
        # Assign fields to blocks
        df['block'] = -1
        for block_idx, block in tqdm(blocks.iterrows(), total=len(blocks), desc='Locating fields within blocks'):
            fields_within_bbox = df.geometry.within(block.geometry)
            df.loc[fields_within_bbox, 'block'] = block_idx

        df = df[df.block != -1]
        print(f'Remaining parcels after removing those not contained in blocks: {len(df)}/{available_fields}')
    else:
        df['block'] = 0  # assign all to single block

    df.geometry = df.geometry.buffer(-erosion_m)
    df = df[~df.geometry.is_empty]
    print(f'Remaining parcels after {erosion_m}m erosion: {len(df)}/{available_fields}')

    df = df[df.geometry.area >= min_area_sqm]
    print(f'Remaining parcels after removing parcels less than {min_area_sqm} m2: {len(df)}/{available_fields}')

    print('Saving filtered shapefile...')
    df.to_file(output_file)

    return df


def create_blocks(tile_bbox, config, s2_tile_size=10980):
    output_dir = config.meta_dir / 'blocks'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f'blocks_{config.country}_{config.tile}_{config.year}.shp'

    crs = tile_bbox.crs
    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = tile_bbox.bounds.values[0]

    block_size, margin = config.block_size, config.margin

    fixed_raster_size = s2_tile_size - ((s2_tile_size - block_size) % (block_size + margin))
    blocks = list(product(range(0, fixed_raster_size, block_size + margin), repeat=2))
    blocks = [(minx, miny, block_size, block_size) for minx, miny in blocks]
    blocks = [{'idx': idx, 'bbox': bbox} for idx, bbox in enumerate(blocks)]

    print(
        f'Dividing rasters with block_size={block_size} and margin={margin}, leaving {s2_tile_size - fixed_raster_size}px unused')

    def pixel_to_world(x, y, ulx, uly, res=10):
        return res * x + ulx, uly - y * res

    blocks_world = []
    for block in blocks:
        minx, miny, width, height = block['bbox']
        world_minx, world_maxy = pixel_to_world(minx, miny, ulx=bbox_minx, uly=bbox_maxy)

        blocks_world.append(box(
            minx=world_minx,
            miny=world_maxy - height * 10,
            maxx=world_minx + width * 10,
            maxy=world_maxy,
        ))

    gdf = gpd.GeoDataFrame({'id': list(range(len(blocks)))}, geometry=blocks_world, crs=crs)
    gdf.to_file(output_file)

    return gdf

def fix_crs(s2_images, epsg=32632):
    from osgeo import gdal, osr
    for s2_image in s2_images:
        for band in s2_image.bands_10m + s2_image.bands_20m:
            ds = gdal.Open(str(band), 1)
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(32632)
            ds.SetProjection(proj.ExportToWkt())
            ds = None
    print('updated crs')

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--tile', type=str, default='32VNH')
    args.add_argument('--country', type=str, choices=['denmark', 'france', 'austria'], default='denmark', help='ground truth file to use')
    args.add_argument('--years', nargs='+', required=True)
    args.add_argument('--start', default='0101', help='start date in format mmdd')
    args.add_argument('--end', default='1231', help='end date in format mmdd')
    args.add_argument('--max_cloudy_percentage', type=float, default=80.0)
    args.add_argument('--min_coverage', type=float, default=50.0)
    args.add_argument('--download_dir', default='/media/data/s2')
    args.add_argument('--ground_truth_dir', default='/media/data/parcels')
    args.add_argument('--output_dir', default='data')
    args.add_argument('--data_collection', type=str, default='l1c', choices=['l1c', 'l2a'])
    args.add_argument('--block_size', type=int, default=1098)
    args.add_argument('--margin', type=int, default=0)  # default 0, as we remove parcels that overlap between blocks
    args.add_argument('--buffer_size', type=int, default=10, help='number of S2 images to load to memory at once')

    config = args.parse_args()

    if config.data_collection == 'l1c':
        config.bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    else:
        config.bands = ['R10m/B02', 'R10m/B03', 'R10m/B04', 'R20m/B05', 'R20m/B06', 'R20m/B07', 'R10m/B08', 'R20m/B8A', 'R20m/B11', 'R20m/B12']

    config.download_dir = os.path.join(config.download_dir, config.data_collection)

    config.id_column, config.crop_code_column = label_utils.get_shapefile_columns(config.country)
    for year in config.years:
        config.year = year
        output_dir = Path(config.output_dir) / config.country / config.tile / config.year
        data_dir = output_dir / 'data'
        data_dir.mkdir(exist_ok=True, parents=True)
        meta_dir = output_dir / 'meta'
        meta_dir.mkdir(exist_ok=True, parents=True)
        config.meta_dir = meta_dir
        config.data_dir = data_dir

        config.end_date = config.year + config.end
        if int(config.end) < int(config.start):
            config.start_date = str(int(config.year) - 1) + config.start
        else:
            config.start_date = config.year + config.start
        print(f'start_date={config.start_date}, end_date={config.end_date}')

        print(config)
        prepare_pixels(config)
