# Dataset guide

## Scope

The article describes 495 facade images:

- 289 field images from Longhua District and Longgang District in Shenzhen, Luhe County in Shanwei, and Yuexiu District in Guangzhou;
- 206 images obtained through web keyword searches; and
- two annotation classes: `cracks` and `exterior_wall_damage`.

The code expects YOLO-format bounding-box labels. The repository does not redistribute the images because image-level permissions and source licences must be verified before public release.

## Expected local layout

```text
data/raw/
|-- images/
|   |-- example_001.jpg
|   `-- example_002.png
`-- labels/
    |-- example_001.txt
    `-- example_002.txt
```

Each label line contains:

```text
class_id x_center y_center width height
```

All coordinates must be normalised to `[0, 1]`. Width and height must be greater than zero. Supported class IDs are:

| ID | Class |
| ---: | --- |
| 0 | cracks |
| 1 | exterior_wall_damage |

Images without a same-stem label file are rejected by default. Empty label files are allowed for verified negative images.

## Split protocol

Run:

```bash
python scripts/prepare_dataset.py \
  --images-dir data/raw/images \
  --labels-dir data/raw/labels \
  --output-dir data/processed \
  --seed 42
```

The script:

1. validates each annotation row;
2. shuffles paired image-label records with a fixed seed;
3. creates train, validation, and test subsets at 80%, 10%, and 10%;
4. copies files into the YOLO directory layout; and
5. writes `manifest.csv` with the source filename, split, and object count.

For 495 valid image-label pairs, integer allocation produces 396 training, 49 validation, and 50 test images.

## Access and responsible use

The article reports that relevant data accompany the Supporting Information and that further information may be requested from the first author subject to Universiti Sains Malaysia consent. Refer to the [version of record](https://doi.org/10.1038/s41598-025-92112-7) for the authoritative data-availability statement.

Before publishing images, confirm:

- permission from property owners or the relevant data custodian;
- licence compatibility for every web-sourced image;
- removal of faces, vehicle plates, addresses, and other unnecessary identifiers; and
- that train/test near-duplicates have been removed to prevent leakage.

## Recommended future release metadata

If an authorised dataset is released, add an image-level metadata table containing source category, city, capture device, material, lighting, defect class, annotator, quality-control status, and licence. A building- or scene-level group identifier should be included so future evaluations can prevent images from the same facade appearing across different splits.
