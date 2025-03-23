# Occlusion-Robust Detection Project

## Overview

This project evaluates edge-deployable models for vehicle detection under occlusion. We will be working with the MS-COCO dataset, filtering it for occluded vehicles, and assessing the performance of four models: MobileNet-SSD V2, YOLOv5-Nano, EfficientDet-Lite0, and SSDLite-MobileDet.

## Setup

Follow these steps to set up your environment and prepare the data:

### Step 1: Download the MS-COCO Dataset

The MS-COCO dataset needs to be downloaded from the official website. `pycocotools` will be used to process the annotations, but it does not provide the raw data itself.

#### Actions:

1.  **Download MS-COCO:**
    * Navigate to the [cocodataset.org/#download](http://cocodataset.org/#download) website.
    * Download the following files:
        * **Test Images:** `test2017.zip` (approximately 6GB).
        * **Annotations:** `annotations_trainval2017.zip` (approximately 241MB; includes test annotations if using the `test-dev` subset).
    * Extract the downloaded files into the `data/raw/coco/` directory:
        * Images should be placed in `data/raw/coco/test2017/`.
        * Annotations should be placed in `data/raw/coco/annotations/` (e.g., `instances_test2017.json`).

2.  **Update `scripts/setup_env.sh`:**
    * Ensure that `pycocotools` is installed correctly. Modify the `scripts/setup_env.sh` script to include a check and installation if it's missing:

    ```bash
    #!/bin/bash
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    # Install pycocotools if not already included
    pip install pycocotools
    echo "Environment setup complete."
    ```

    * This script will create a virtual environment, install the required Python packages from `requirements.txt`, and ensure `pycocotools` is installed.

**Next Steps:**

After completing these setup steps, you will be ready to proceed with filtering the MS-COCO dataset for occluded vehicles and setting up the inference for the four models.