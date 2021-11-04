import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List

import requests
import sentinelhub
from dateutil.rrule import rrule, MONTHLY
from sentinelhub import DataCollection
from tqdm import tqdm
import boto3

# Call roda API for listing, its free
L1C_API = "https://roda.sentinel-hub.com/sentinel-s2-l1c"
L2A_API = "https://roda.sentinel-hub.com/sentinel-s2-l2a"


class S2Image:
    def __init__(self, name, yyyymmdd, cloudy_pct, coverage, aws_path, local_path, data_collection):
        self.name = name
        self.yyyymmdd = yyyymmdd
        self.cloudy_pct = cloudy_pct
        self.coverage = coverage
        self.aws_path = aws_path
        self.local_path = local_path
        if data_collection == 'l1c':
            self.bands_10m = [local_path / "{}.jp2".format(x) for x in ['B02', 'B03', 'B04', 'B08']]
            self.bands_20m = [local_path / "{}.jp2".format(x) for x in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']]
            self.bands_60m = [local_path / "{}.jp2".format(x) for x in ['B01', 'B09', 'B10']]
            self.cloud_mask = None
        else:
            self.bands_10m = [local_path / 'R10m' / "{}.jp2".format(x) for x in ['B02', 'B03', 'B04', 'B08']]
            self.bands_20m = [local_path / 'R20m' / "{}.jp2".format(x) for x in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']]
            self.bands_60m = [local_path / 'R60m' / "{}.jp2".format(x) for x in ['B01', 'B09', 'B10']]
            self.cloud_mask = local_path / 'qi' / 'MSK_CLOUDS_B00.gml'


    def get_date(self):
        return datetime.strptime(self.yyyymmdd, '%Y%m%d')

    def __repr__(self):
        return f"S2Image(tile={self.name}, date={self.yyyymmdd}, clouds={self.cloudy_pct}, coverage={self.coverage})"


def s2_download(tile_name, date_start, date_end, download_dir, min_coverage=100.0, max_cloudy_pct=100.0,
                previews=False, sort_by_date=True, bands=None, data_collection='l1c'):
    if type(date_start) == str:
        date_start, date_end = tuple(map(_yyyymmdd_to_date, (date_start, date_end)))

    assert data_collection in ['l1c', 'l2a']

    if bands is None:
        if data_collection == 'l1c':
            bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        else:
            bands = ['R10m/B02', 'R10m/B03', 'R10m/B04', 'R20m/B05', 'R20m/B06', 'R20m/B07', 'R10m/B08', 'R20m/B8A', 'R20m/B11', 'R20m/B12']

    cache_file = "/tmp/{}_{}_{}_{}.pkl".format(data_collection, tile_name, date_start.date(), date_end.date())
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            s2_images = pickle.load(f)
    else:
        s2_images = get_available_s2_images(tile_name, (date_start, date_end), download_dir, data_collection)
        with open(cache_file, "wb") as f:
            pickle.dump(s2_images, f)

    print("Found {} images from tile {} between {} and {}".format(len(s2_images), tile_name, date_start.date(),
                                                                  date_end.date()))
    n_before_filter = len(s2_images)
    if sort_by_date:
        s2_images = sorted(s2_images, key=lambda x: x.get_date())
    s2_images = list(filter(lambda x: x.coverage >= min_coverage, s2_images))
    s2_images = list(filter(lambda x: x.cloudy_pct <= max_cloudy_pct, s2_images))
    print("Number of images with coverage >= {} and cloudy pct. <= {}: {}/{}".format(min_coverage, max_cloudy_pct,
                                                                                     len(s2_images),
                                                                                     n_before_filter))

    tile_dir = os.path.join(download_dir, tile_name)
    os.makedirs(tile_dir, exist_ok=True)
    existing = os.listdir(tile_dir)
    not_downloaded = [x for x in s2_images if x.local_path.parts[-1] not in existing]
    print("Number of images not downloaded: {}/{}".format(len(not_downloaded), len(s2_images)))
    if len(not_downloaded) > 0:
        print(not_downloaded)

    if previews:
        preview_folder = "data/previews_{}_{}_{}".format(tile_name, date_start.date(), date_end.date())
        if not os.path.exists(preview_folder):
            os.makedirs(preview_folder)
            download_previews(s2_images, preview_folder)
        else:
            print('Preview folder {} already exists, skipping'.format(preview_folder))

    if len(not_downloaded) != 0:
        input("Will download {} S2 images. OK?".format(len(not_downloaded)))
        for s2_tile in tqdm(not_downloaded, 'Downloading S2 tiles'):
            download_tile(s2_tile, data_folder=tile_dir, bands=bands, data_collection=data_collection)

    return s2_images


def _yyyymmdd_to_date(yyyymmdd):
    return datetime.strptime(yyyymmdd, "%Y%m%d")


def download_previews(s2_images: List[S2Image], preview_folder):
    s3 = boto3.client('s3')
    for s2_tile in tqdm(s2_images, desc='downloading previews'):
        key = s2_tile.aws_path + '/preview.jpg'
        file_name = "{}_{}pct.jpg".format(s2_tile.yyyymmdd, int(s2_tile.cloudy_pct))
        file_name = os.path.join(preview_folder, file_name)
        if not os.path.exists(file_name):
            try:
                s3.download_file('sentinel-s2-l1c', key, file_name, ExtraArgs={'RequestPayer': 'requester'})
            except Exception:
                print('preview not found', key)


def download_tile(s2_tile: S2Image, data_folder, bands=None, data_collection='l1c'):
    data_collection = DataCollection.SENTINEL2_L1C if data_collection == 'l1c' else DataCollection.SENTINEL2_L2A
    if data_collection == DataCollection.SENTINEL2_L1C:
        metafiles = ['metadata', 'tileInfo']
    else:
        metafiles = ['metadata', 'tileInfo', 'qi/MSK_CLOUDS_B00']  # also download cloud masks for L2A data
    req = sentinelhub.AwsTileRequest(
        tile=s2_tile.name,
        time=s2_tile.yyyymmdd,
        aws_index=0,
        data_folder=data_folder,
        metafiles=metafiles,
        bands=bands,
        data_collection=data_collection,
    )
    req.save_data()


def get_available_s2_images(tile_name, date_interval, download_dir, data_collection):
    # api = L1C_API if data_collection == DataCollection.SENTINEL2_L1C else L2A_API
    api = L1C_API  # use l1c api for l2a data as l2a cloudy percentage is buggy
    start_date, end_date = date_interval
    months_between = [x for x in rrule(MONTHLY, dtstart=start_date, until=end_date)]
    result = []
    for dt in tqdm(months_between, desc=f'Browsing S2 {data_collection} inventory for available images'):
        tile_path = "{}/{}/{}".format(tile_name[:2], tile_name[2], tile_name[3:])
        url = "/".join([api, "tiles", tile_path, str(dt.year), str(dt.month), ""])
        js = requests.get(url).json()
        date_prefixes = [api + "/" + x['Prefix'] for x in js['CommonPrefixes']]
        for date_prefix in date_prefixes:
            tileinfo = requests.get(date_prefix + "0/tileInfo.json").json()
            aws_path = date_prefix.replace(api, "")[1:] + '0'
            date = get_date_from_prefix(date_prefix)
            if start_date <= _yyyymmdd_to_date(date) <= end_date:
                try:
                    cloudy_pct = tileinfo['cloudyPixelPercentage']
                    coverage = tileinfo['dataCoveragePercentage']
                except KeyError as e:
                    print(e)
                    continue

                local_path = Path(download_dir) / tile_name / "{},{}-{}-{},0".format(tile_name, date[:4], date[4:6],
                                                                                     date[6:])
                result.append(S2Image(tile_name, date, cloudy_pct, coverage, aws_path, local_path, data_collection))

    return result


def get_date_from_prefix(prefix):
    parts = prefix.split('/')
    parts = list(filter(None, parts))
    yyyy = parts[-3]
    mm = "0" + parts[-2] if len(parts[-2]) == 1 else parts[-2]
    dd = "0" + parts[-1] if len(parts[-1]) == 1 else parts[-1]
    return yyyy + mm + dd


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tile', type=str, help='For example 32VNH')
    parser.add_argument('--start', type=str, help='Start date in yyyymmdd format to download from (inclusive)')
    parser.add_argument('--end', type=str, help='End date in yyyymmdd format to download to (inclusive)')
    parser.add_argument('--dir', type=str, default='/media/data/s2/l1c')
    parser.add_argument('--bands', default=['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'], type=str, nargs='+')
    parser.add_argument('--collection', type=str, default='l1c', choices=['l1c', 'l2a'])
    parser.add_argument('--max_cloudy_percentage', type=float, default=80.0)
    parser.add_argument('--min_coverage', type=float, default=100.0)
    config = parser.parse_args()

    s2_download(tile_name=config.tile, date_start=config.start, date_end=config.end, download_dir=config.dir,
                bands=config.bands, min_coverage=config.min_coverage, max_cloudy_pct=config.max_cloudy_percentage,
                data_collection=config.collection)
