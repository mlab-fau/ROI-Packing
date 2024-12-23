## ROI-Packing: A Novel Approach to Image Compression for Machine Vision

![License](https://img.shields.io/badge/license-BSD--3--Clause--Clear-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue)

ROI-Packing is a novel approach to image compression for **machine vision**. By selecting important regions and packing them together, it eliminates underutilized space to improve **compression** and **computational** efficiency of traditional intra codecs. This project contains scripts to run the ROI region processing pipeline on images with optional packing.

![Pipeline](.github/figure.svg)

## Setup

ROI-Packing is supported on Linux and Python 3.8+.

```sh
pip install -r requirements.txt
```

## Getting Started

Start [here](notebook/pipeline.ipynb) with an interactive Jupyter notebook for running the ROI pipeline. It is recommended to test out functions in the 
notebook to get an understanding of how the pipeline works and to visualize the output of each of the steps.

## Advanced Usage

Make sure you are running the pipeline from the [scripts/](scripts/) directory.

```sh
cd scripts/
```

Then run the the following in order:

1. Processing Pipeline
2. (optional) Your own encoding + decoding
3. Unpacking Pipeline

### Processing Pipeline

```sh
bash run_pipeline.sh [input] [predictions] [output] [sequence] [packing] [padding] [size] [scale] [alignCTU] [reducedParameters]
```

| Parameter           | Description                                                                            | Value           |
|---------------------|----------------------------------------------------------------------------------------|-----------------|
| `input`             | A dataset folder with PNGs or JPGs for processing                                      | Path            |
| `predictions`       | A CSV, should contain the columns: img, box_x1, box_y1, box_x2, box_y2, label, score   | Path            |
| `output`            | Output folder                                                                          | Path            |
| `sequence`          | Determines if the images are to be processed as a sequence or set of individual images | true/false      |
| `packing`           | Turn packing on or off                                                                 | true/false      |
| `padding`           | Padding size                                                                           | int             |
| `size`              | The size that the image will be encoded at                                             | 100, 75, 50, 25 |
| `scale`             | Apply class based region scaling                                                       | true/false      |
| `alignCTU`          | Align regions to CTU with multiples of 16                                              | true/false      |
| `reducedParameters` | Reduce parameters to multiples of 16                                                   | true/false      |

### Unpacking Pipeline

```sh
bash run_unpacking.sh python [input] [csv] [output] [sequence] [reducedParameters] [encoded] [rescaleSize]
```

| Parameter           | Description                                                                            | Value           |
|---------------------|----------------------------------------------------------------------------------------|-----------------|
| `input`             | Folder of images to unpack                                                             | Path            |
| `csv`               | Folder of CSV packed parameters for images                                             | Path            |
| `output`            | Output folder                                                                          | Path            |
| `sequence`          | Determines if the images are to be processed as a sequence or set of individual images | true/false      |
| `reducedParameters` | Are the CSV parameters reduced to multiples of 16                                      | true/false      |
| `encoded`           | Were the CSV files decoded from bitstreams (necessary to identify naming conventions)  | true/false      |
| `rescaleSize`       | Rescale size for evaluation                                                            | 100, 75, 50, 25 |

### Examples

See [test/](test/), a sample folder with Bash files showing ideal workflow setup and output.

## Repository Contents

Most work is kept in the [scripts/](scripts/) folder, which contains Python and Bash scripts for running the pipeline. It includes:
* [`dicts.py`](scripts/dicts.py): Contains dictionaries necessary for running pipeline (image sizes for datasets)
* [`packer.py`](scripts/packer.py): Contains methods for running packing algorithm
* [`region_boxes.py`](scripts/region_boxes.py): Contains methods for creating region boxes using top-down approach
* [`run_pipeline.py`](scripts/run_pipeline.py): Script to run entire pipeline (processing and packing)
* [`run_pipeline.sh`](scripts/run_pipeline.sh): Bash script which calls run_pipeline.py
* [`run_unpacking.py`](scripts/run_unpacking.py): Script to run entire unpacking pipeline (additionally fills background for non-packed images)
* [`run_unpacking.sh`](scripts/run_unpacking.sh): Bash script which calls run_unpacking.py

## License

ROI-Packing is licensed under the BSD 3-Clause Clear License.
