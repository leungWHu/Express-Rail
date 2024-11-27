# Project Description

This is the English version of the project description.

[中文版](README.md)

## Overview
This dataset contains point cloud data of railway tracks and their surrounding environments, aiming to support research in fields such as railway inspection, 3D modeling, and digitalization. The point cloud data is collected using LiDAR equipment and includes high-precision labels.


## Dataset Structure
The dataset is stored in LAS format, with the following folder structure:
```bash
/dataset-root
├── README.txt                # Dataset description file
├── /train                    # Directory for point cloud data
│   ├── track_segment1.las    # Point cloud data for the first track segment
│   ├── track_segment2.las    # Point cloud data for the second track segment
│   └── ...
└── /test             
    ├── track_segment1.las   # Point cloud data for the first track segment
    ├── track_segment2.las   # Point cloud data for the second track segment
    └── ...
```


## Data Preview
| Label | Chinese Name       | English Name | Description                                                            |
|-------|---------------------|--------------|------------------------------------------------------------------------|
| 0     | 铁路               | Rail         | Represents the steel rail part of the railway track.                   |
| 1     | 支撑结构           | Support      | Represents supporting structures related to the railway, such as cable racks. |
| 2     | 支柱               | Pillar       | Represents pillar structures along the railway, such as utility poles. |
| 3     | 接触网             | Overhead     | Represents the overhead cables above the railway.                      |
| 4     | 围栏/立面设施      | Fence        | Represents fences, sound barriers, and other vertical facilities around the railway. |
| 5     | 轨道床             | Track Bed    | Represents the ballast or track bed structure supporting the railway track. |
| 6     | 植被               | Vegetation   | Represents vegetation along the railway, such as trees and shrubs.     |
| 7     | 地面               | Ground       | Represents the ground around the railway, including leveled surfaces and slopes. |
| 8     | 未分类的点集合     | Others       | Represents unclassified points or points that do not belong to other categories. |

Sample images from the dataset are provided to help quickly understand the dataset contents:

![dataset.png](dataset/dataset.png)  

_Figure 1. Sample visualization of railway point cloud data_


## Instructions
Download and extract the dataset locally.

## License
This dataset is available for academic research and non-commercial use. Please cite the following reference when using this dataset:
> Author, Paper Title, Journal/Conference, Year.

## Contact
For any questions or suggestions, please contact the dataset maintainer:

Name: leung
Email: gisleung@whu.edu.cn