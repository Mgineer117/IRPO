# Intrinsic Reward Policy Optimization
[![arXiv](https://img.shields.io/badge/arXiv-2601.21391-b31b1b.svg)](https://arxiv.org/abs/2601.21391)
<img width="1918" height="626" alt="IRPO" src="https://github.com/user-attachments/assets/ac129e10-304a-4b40-8361-8e82844f5f11" />


## Authors
* **Minjae Cho** (Correspondance) - _The Grainger College of Engineering, University of Illinois Urbana-Champaign_ 
* **Huy T. Tran** - _The Grainger College of Engineering, University of Illinois Urbana-Champaign_

## Citation
Please cite our paper if you use this code or algorithm for any part of your research or work:
```
@misc{cho2026intrinsicrewardpolicyoptimization,
      title={Intrinsic Reward Policy Optimization for Sparse-Reward Environments}, 
      author={Minjae Cho and Huy Trong Tran},
      year={2026},
      eprint={2601.21391},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.21391}, 
}
```

## Prerequisites
In any local folder, open a terminal and run the following command to download our package into the current directory:
```
git clone https://github.com/Mgineer117/IRPO/
cd IRPO
```

We assume that you have Conda installed. If not, please refer to the [Anaconda installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install). Python 3.11.11 was used for our code. 

We recommend creating a dedicated virtual environment as follows:
```
conda create -n IRPO python==3.11.*
conda activate IRPO
```

Then, install the required Python packages using:
```
pip install -r requirements.txt
```

## Training
Our code uses the following command to train algorithms:
```
python3 main.py --env-name pointmaze-v1 --algo-name irpo
```
where all arguments should be written all lowercase.

## Logging
We support three logging options—Weights & Biases (WandB), TensorBoard, and CSV—to accommodate different user preferences. Specifically, when WandB is properly configured on your local machine, all algorithmic and parameter settings, along with real-time training metrics, are automatically logged to your WandB dashboard. Simultaneously, training results are saved locally in TensorBoard format for visualization, and evaluation metrics are exported as CSV files for easy analysis. In addition, the model parameters are saved along with the best-performing one.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
