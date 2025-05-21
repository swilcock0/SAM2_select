# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Define the base URL for SAM 2.1 checkpoints
$SAM2p1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824"

# Define the URLs for the checkpoints
$checkpoints = @{
    "sam2_hiera_tiny.pt"       = "$SAM2p1_BASE_URL/sam2_hiera_tiny.pt"
    "sam2_hiera_small.pt"      = "$SAM2p1_BASE_URL/sam2_hiera_small.pt"
    "sam2_hiera_base_plus.pt"  = "$SAM2p1_BASE_URL/sam2_hiera_base_plus.pt"
    "sam2_hiera_large.pt"      = "$SAM2p1_BASE_URL/sam2_hiera_large.pt"
}

# Download each checkpoint
foreach ($name in $checkpoints.Keys) {
    $url = $checkpoints[$name]
    Write-Host "Downloading $name from $url ..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $name -UseBasicParsing
        Write-Host "$name downloaded successfully.`n"
    } catch {
        Write-Error "Failed to download $name from $url"
        exit 1
    }
}

Write-Host "All checkpoints are downloaded successfully."
