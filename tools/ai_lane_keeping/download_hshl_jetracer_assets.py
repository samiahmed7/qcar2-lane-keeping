#!/usr/bin/env python3
"""Download public HSHL/Tasawar JetRacer lane-following assets.

The pure PyTorch ResNet files are the ones to use for ROS inference. The TRT
files are TensorRT-serialized Jetson artifacts and are not portable to a normal
desktop ROS environment.
"""
import argparse
import urllib.request
from pathlib import Path


ASSETS = {
    'resnet18': (
        'models/ai_lane_keeping/road_following_model_resnet18.pth',
        'https://raw.githubusercontent.com/TasawarSiddiquy/'
        'Automated-lane-following-Waveshare-JetRacer-with-artificial-intelligence/'
        'main/Resnet18%20Model/road_following_model%20Resnet18.pth',
    ),
    'resnet34': (
        'models/ai_lane_keeping/road_following_model_resnet34.pth',
        'https://raw.githubusercontent.com/TasawarSiddiquy/'
        'Automated-lane-following-Waveshare-JetRacer-with-artificial-intelligence/'
        'main/Resnet%2034%20Model/road_following_model%20Resnet34.pth',
    ),
    'video': (
        'data/ai_lane_keeping/hshl_jetracer_video.mp4',
        'https://raw.githubusercontent.com/TasawarSiddiquy/'
        'Automated-lane-following-Waveshare-JetRacer-with-artificial-intelligence/'
        'main/Video/video_2022-10-31_21-21-51.mp4',
    ),
    'track': (
        'data/ai_lane_keeping/hshl_track.jpg',
        'https://raw.githubusercontent.com/TasawarSiddiquy/'
        'Automated-lane-following-Waveshare-JetRacer-with-artificial-intelligence/'
        'main/Track.jpg',
    ),
}


def download(name, workspace):
    relative_path, url = ASSETS[name]
    destination = workspace / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {name} -> {destination}')
    urllib.request.urlretrieve(url, destination)
    print(f'OK: {destination.stat().st_size} bytes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'assets',
        nargs='*',
        choices=sorted(ASSETS),
        default=['resnet18', 'track'],
        help='Assets to download. Default: resnet18 track',
    )
    parser.add_argument(
        '--workspace',
        default='.',
        help='Workspace root where models/ and data/ will be created.',
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    for name in args.assets:
        download(name, workspace)


if __name__ == '__main__':
    main()

