# D2KLab @ European Statistics Awards Deduplication Challenge 2023

This repository contains the source code used for reproducing the experiments conducted by D2KLab for the [European Statistics Awards Deduplication Challenge 2023](https://statistics-awards.eu/competitions/4#).

The source code on this repository was based on a Jupyter Notebook which is available on Google Colab at [this link](https://colab.research.google.com/drive/1VLIkrP552ZQh6Qk-erVu-OmDU4gtV33k).

## Requirements

* Python >=3.9

## How to use

1. Install required packages using `pip`:
    ```sh
    pip install -r requirements.txt
    ```
1. Copy the dataset file `wi_dataset.csv` into the same directory as this source code.
1. Run `main.py`:
    ```sh
    python main.py
    ```
    After processing, this should create a new file named `duplicates.csv`.

## Results

During the submission phase, our latest experiment obtained the following scores:

| Full F1 | Semantic F1 | Temporal F1 | Partial F1 | Non-Duplicate F1 | Macro F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.99 | 0.81 | 0.68 | 0.00 | 1.00 | 0.70 |

## References

* [SentenceTransformer](https://www.sbert.net/index.html)
* [LaBSE](https://huggingface.co/sentence-transformers/LaBSE)