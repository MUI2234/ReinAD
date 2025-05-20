# ReinAD
## 1. Training
### 1.1 Prepare data
Download [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad), [BTAD](https://avires.dimi.uniud.it/papers/btad/btad.zip), [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar), [ReinAD](https://kaggle.com/datasets/595fda0f2e4a97cc5955c102727fab09faa65a0a34493fc2879f733c0c3e05af) and other datasets. Unzip and move them to `./data`.
The data directory of ReinAD should be as follows.
dataset/
├── train/
│   ├── category1/
│   │   └── Data/
│   │       ├── Images/
│   │       │   ├── Anomaly/       # Anomaly images
│   │       │   ├── Normal/        # Normal images
│   │       │   └── Prompt/        # Prompt for few-shot settings
│   │       └── Masks/
│   │           └── Anomaly/        # Pixel-level annotations for anomaly images
│   ├── category2/
│   │   └── Data/                   # Same structure as category 1
│   ├── category3/
│   │   └── Data/                   # Same structure as category 1
│   └── ...                          # Additional categories
└── test/
    ├── category1/                   # Same structure as train/category1
    ├── category2/
    ├── category3/
    └── ...                          # Additional categories

### 1.2 Extract prompt features
extract_ref_features
### 1.3 Training ReinAD
