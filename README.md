# DAIR-V2X-C Preprocessing and Visualization

This repository contains preprocessing and visualization scripts for the DAIR-V2X-C dataset.

## Project structure

- `config.py` - dataset root and output path configuration
- `scripts/` - preprocessing and visualization scripts
- `outputs/` - generated outputs
  - `visual_check/` - generated visualizations (ignored by git)
  - `preprocessed/` - generated CSV statistics (ignored by git)

## Usage

1. Set your dataset path in `config.py`:
   ```python
   DATA_ROOT = Path(r"D:/Python/study/item3.0/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure")
   ```

2. Activate your Python environment:
   ```powershell
   conda activate dair-preprocess
   ```

3. Run preprocessing and statistics scripts:
   ```powershell
   python scripts/01_check_structure.py
   python scripts/02_inspect_data_info.py
   python scripts/03_read_image.py
   python scripts/04_read_pointcloud.py
   python scripts/05_read_label.py
   python scripts/06_read_calib.py
   python scripts/07_build_pair_index.py
   python scripts/08_file_size_stats.py
   python scripts/09_visual_check_batch.py
   python scripts/11_label_statistics.py
   python scripts/12_build_device_modality_index.py
   ```

4. Generated visualizations and CSV files are stored under `outputs/visual_check/` and `outputs/preprocessed/`.

## Notes

- The repository does not track generated visualization results or CSV files unless explicitly requested.
- Keep `outputs/visual_check/` and `outputs/preprocessed/*.csv` out of git history to avoid large binary and temporary files.
