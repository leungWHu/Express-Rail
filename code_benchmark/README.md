# Benchmark Methods

[Chinese Version](README_cn.md)

## Introduction
* When testing benchmark methods, we leverage the [Open3D-ML](https://github.com/isl-org/Open3D-ML) library to integrate common benchmark methods into a unified code framework.
* The benefit of this approach is that all methods use the same data preprocessing, input strategies, and hyperparameter settings, ensuring a fair comparison.
* **Please note:**  The code has not been fully organized yet. We will upload a more standardized and readable version in the near future. Stay tuned!

## 1 Install the Runtime Environment
We have referred to the Open3D-ML usage documentation, and the final configured runtime environment is as follows:
* Pytorch = 2.0.0
* Pytorch-cuda = 11.7
* Python = 3.10
* Open3d = 0.18.0

When creating the Python virtual environment, follow these steps:
```bash
# Create virtual environment
conda create -n py310_open3d-ml python=3.10
conda activate py310_open3d-ml

# Install required dependencies for code execution. You can install additional dependencies based on program prompts.
conda install ...

# Test if Open3D is installed correctly
pip install open3d
python -c "import open3d.ml.torch as ml3d"
```

## 2 Railway Point Cloud Data Preprocessing
Before using the data in the dataset for formal training, we perform chunking on each scene.  
The goal of chunking is to control the number of points per chunk to approximately 100,000, instead of simply dividing the point cloud using a uniform size.  
The preprocessing code is located at [prepare_ExpressRail.py](code_benchmark/dataset/ExpressRail/prepare/prepare_ExpressRail.py), and the processing logic includes the following steps:
1. Read each railway point cloud data file.
2. Perform down-sampling on the point cloud data according to the desired grid size (e.g., grid_size=0.05m).
3. Rotate the scene point cloud to a horizontal position automatically.
4. Chunk the data first along the track direction, and then along the vertical direction based on the expected number of points and chunk size.
5. Save each chunk’s data as the basic unit input for the network.

## 3 Configuration Files
For the benchmark testing of the ExpressRail dataset, the configuration files are located in the `code_benchmark/dataset/ExpressRail` directory.  
These configuration files include `*.yaml` files tailored for different methods.

## 4 Running the Code
1. **Training**:   
 Adjust the parameters in the `config_{model}.yaml` configuration file as needed, then run the following command:
```bash
python train_Fit_PointNet2.py --data=ExpressRail
```
2. **Test**:  
Change `run_mode='test'` in the `train_Fit_{model}.py` script to enable testing mode,  
and modify the `test_ckpt_path: path/to/your/checkpoint/*.pth` in the `config_{model}.yaml` file to point to the appropriate checkpoint file.  
Then run the following command:
```bash
python train_Fit_PointNet2.py --data=ExpressRail

