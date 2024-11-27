# Express-Rail Point Cloud Dataset

[中文版](README_cn.md)  

## 🎈 1. Introduction  

1. **Dataset Overview**:  
   This dataset contains high-quality point cloud data of railway tracks and their surrounding environments. It aims to support research in fields such as railway inspection, 3D modeling, and digitalization. The data was collected using LiDAR sensors and includes high-precision labels.
   
2. **Benchmark Methods**:  
   We provide benchmark results of popular methods evaluated on this dataset. The corresponding code is available in the `code_benchmark` directory.

3. **Research Contribution**:  
   In our paper (under review), we propose a weakly supervised method based on active learning. This method achieves higher semantic segmentation accuracy with significantly fewer labels. The relevant code is available in the `code_paper` directory.

## 🎈 2. Dataset  

### 2.1 Data Format  

The dataset is stored in **LAS** format with the following folder structure:  

```bash
/dataset-root
├── README.md                 # Dataset description file
├── /train                    # Directory for training data
│   ├── track_segment1.las    # Point cloud of the first railway segment
│   ├── track_segment2.las    # Point cloud of the second railway segment
│   └── ...
└── /test                     # Directory for testing data
    ├── track_segment1.las    
    ├── track_segment2.las    
    └── ...
```

### 2.2 Label Categories  

The dataset includes the following labeled categories:  

| Label ID | Name (CN)         | Name (EN)       | Description                                      |
|---------|-------------------|-----------------|--------------------------------------------------|
| 0       | 铁路              | Rail            | Steel rails of the railway track                 |
| 1       | 支撑结构          | Support         | Supporting structures like cable brackets        |
| 2       | 支柱              | Pillar          | Pillars along the railway, e.g., utility poles   |
| 3       | 接触网            | Overhead Lines  | Overhead cables above the railway                |
| 4       | 围栏/立面设施     | Fence           | Fences, sound barriers, and other vertical structures |
| 5       | 轨道床            | Track Bed       | Ballast or subgrade structure supporting the track |
| 6       | 植被              | Vegetation      | Vegetation along the railway, such as trees and shrubs |
| 7       | 地面              | Ground          | Ground surfaces, including leveled roads and slopes |
| 8       | 未分类的点集合     | Others          | Unclassified points or points not belonging to other categories |



### 2.3 Dataset Overview  

We provide sample images from the dataset to give a quick overview of the data content:  

![dataset.png](dataset/dataset.png)  

_Figure. Example of railway point cloud data._

### 2.4 Download and Usage  

* **Download**: Please refer to `dataset/README.md` for download links or methods.
* **Usage**:  
  1. Use the provided code in this repository. Refer to the instructions in the code directories.  
  2. Alternatively, adapt the dataset for your own tasks as needed.

## 🎈 3. Benchmark Methods  

### 3.1 Experimental Results  

We evaluated several popular methods on this dataset, including `PointNet++`, `DGCNN`, `KPConv`, and `RandLA-Net`. The results are summarized in the table below:

| Method       | Rail | Support | Pillar | Overhead | Fence | Bed  | Veget. | Ground | Others | mIoU (%) | OA (%) |
|--------------|------|---------|--------|----------|-------|------|--------|--------|--------|----------|--------|
| PointNet++   | 82.0 | 78.9    | 84.1   | 92.9     | 95.6  | 94.5 | 90.1   | 83.1   | 73.8   | 86.1     | 95.0   |
| DGCNN        | 83.4 | 82.0    | 84.2   | 96.7     | 92.7  | 95.0 | 91.9   | 81.4   | 63.6   | 85.6     | 94.8   |
| KPConv       | 86.7 | 79.2    | 87.0   | 95.2     | 95.7  | 95.0 | 92.1   | 83.9   | 75.3   | 87.8     | 95.6   |
| RandLA-Net   | 75.4 | 85.0    | 91.0   | 97.6     | 97.2  | 93.7 | 92.5   | 85.3   | 76.0   | 88.2     | 95.2   |
| Transformer   |  |     |    |      |   |  |    |    |    |      |    |

### 3.2 Code  

* **Details**:  
  For benchmarking, we integrated the above methods into a unified codebase using the `Open3D-ML` library. This ensures consistent preprocessing, data input strategies, and hyperparameter settings across methods, allowing fair comparisons.  

* **Usage**:  
  Please refer to the documentation in the `code_benchmark` directory.

## 🎈 4. Our Method  

### 4.1 Experimental Results  

Compared to popular weakly supervised methods, our method uses approximately **0.1‰** of labeled data and achieves superior accuracy, even outperforming some fully supervised approaches. The results are shown below:

| Weak Supervision   | Rail | Support | Pillar | Overhead | Fence | Bed  | Veget. | Ground | Others | mIoU (%) | OA (%) |
|--------------------|------|---------|--------|----------|-------|------|--------|--------|--------|----------|--------|
| SQN (0.1%)         | 55.8 | 60.8    | 71.1   | 92.1     | 93.3  | 89.6 | 88.0   | 76.4   | 61.1   | 76.4     | 91.6   |
| SQN (1%)           | 71.2 | 69.9    | 78.6   | 92.3     | 93.7  | 93.1 | 90.7   | 82.4   | 68.9   | 82.3     | 94.1   |
| PSD (1%)           | 83.5 | 84.0    | 89.1   | 97.7     | 96.5  | 94.8 | 92.2   | 83.2   | 76.9   | 88.6     | 95.6   |
| OCOC (1pt)         | 83.0 | 86.4    | 88.0   | 97.8     | 97.0  | 93.3 | 92.7   | 79.6   | 76.9   | 88.3     | 94.9   |
| **Ours (~0.1‰)**   | **88.1** | **88.3** | **92.2** | **98.3** | 96.9  | **95.5** | 92.1 | **85.4** | **78.8** | **90.6** | **96.2** |

### 4.2 Code  

Please refer to the documentation in the `code_paper` directory.

## 🤝 License  

This dataset is available for academic research and non-commercial use. Please cite the following paper when using this dataset:  

> Author Name, Paper Title, Published Journal/Conference, Year.  

## 🤝 Contact  

For any questions or suggestions, please contact the dataset maintainers:  

Name: Leung  
Email: gisleung@whu.edu.cn  

