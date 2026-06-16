#!/usr/bin/env python3
import os

import pandas as pd
import requests
import zarr


def inspect_remote_csv(url, timeout=10):
    print("=== Remote CSV Pathway ===")
    print(f"Loading data from {url}...")
    try:
        # Check if URL is accessible
        response = requests.head(url, timeout=timeout)
        response.raise_for_status()

        # Load the CSV
        df = pd.read_csv(url)
        print("Successfully loaded remote CSV dataset.")

        # Inspect columns, shape, unique track_id count
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        if 'track_id' in df.columns:
            unique_tracks = df['track_id'].nunique()
            print(f"Unique track_id count: {unique_tracks}")
        else:
            print("Warning: 'track_id' column not found in the dataset.")

    except requests.exceptions.Timeout:
        print(f"Error: Network timeout occurred while trying to access {url}.")
    except requests.exceptions.RequestException as e:
        print(f"Error: Network request failed. {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
    print()

def inspect_local_zarr(zarr_path):
    print("=== Local Zarr Bundle Pathway ===")
    print(f"Opening Zarr bundle at {zarr_path}...")

    if not os.path.exists(zarr_path):
        print(f"Error: Zarr bundle not found at '{zarr_path}'.")
        return

    """
    Zarr Group Layout Explanation:
    The Zarr bundle is structured as a hierarchical storage format, often used to store large multi-dimensional arrays and metadata.
    In the context of cell tracking data, the bundle typically contains the following groups:
    - 'points': Stores the spatial coordinates (e.g., x, y, z, t) of detected cells or objects across time.
    - 'points_to_tracks': A sparse mapping or index array linking individual points (cells) to their corresponding tracks.
    - 'tracks_to_points': The inverse mapping, linking each track ID to the set of point indices that belong to it.
    - 'tracks_to_tracks': Represents the lineage or graph structure, mapping parent tracks to child tracks, essential for understanding cell division events.
    Metadata attributes at the root or group level provide context such as scaling, units, or experimental details.
    """
    try:
        # Open the zarr bundle
        bundle = zarr.open(zarr_path, mode='r')

        print("Root metadata attributes:")
        for key, value in bundle.attrs.items():
            print(f"  {key}: {value}")

        # Target groups to inspect
        target_groups = ['points', 'points_to_tracks', 'tracks_to_points', 'tracks_to_tracks']

        for group_name in target_groups:
            if group_name in bundle:
                group = bundle[group_name]
                print(f"\nGroup: '{group_name}'")

                # Check if it's an array or a group containing arrays
                if isinstance(group, zarr.Array):
                    print("  Type: Array")
                    print(f"  Shape: {group.shape}")
                    print(f"  Dtype: {group.dtype}")
                elif isinstance(group, zarr.Group):
                    print("  Type: Group")
                    print(f"  Sub-arrays/groups: {list(group.array_keys())}")
                    for key in group.array_keys():
                        print(f"    - {key} shape: {group[key].shape}")
                else:
                    print(f"  Type: {type(group)}")

                # Print group specific attributes
                if group.attrs:
                    print("  Attributes:")
                    for k, v in group.attrs.items():
                        print(f"    {k}: {v}")
            else:
                print(f"\nWarning: '{group_name}' not found in the Zarr bundle.")

    except Exception as e:
        print(f"Error: Failed to read Zarr bundle. {e}")
    print()

def main():
    print("=== inTRACKtive Data Inspection Script ===\n")

    # Pathway 1: Remote CSV
    remote_url = 'https://public.czbiohub.org/royerlab/zoo/C_elegans/tracks.csv'
    inspect_remote_csv(remote_url)

    # Pathway 2: Local Zarr
    local_zarr_path = 'gt_data_bundle.zarr'
    inspect_local_zarr(local_zarr_path)

if __name__ == "__main__":
    main()
