# Kidney Blood Vessel Segmentation

This project focuses on the segmentation of blood vessels in kidney images using various deep learning models. Our proposed model, DeepSegPro, achieves state-of-the-art performance on this task.

## Dataset

The dataset for this project is available on Kaggle. To download the dataset, follow these steps:

1. Ensure you have the Kaggle CLI installed and configured with your API credentials.
2. Run the following command:

```
kaggle competitions download -c blood-vessel-segmentation
```

3. Unzip the downloaded file to access the dataset.

## Environment Setup

To set up the conda environment for this project, follow these steps:

1. Ensure you have Anaconda or Miniconda installed on your system.
2. Create a new conda environment:

```
conda create -n kidney_seg python=3.8
```

3. Activate the environment:

```
conda activate kidney_seg
```

4. Install the required packages:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge opencv matplotlib scikit-learn
pip install segmentation-models-pytorch
```

## Running the Code

(Note: The actual commands to run the code are not provided in the given information. Here are some example commands you might use. Please adjust these based on your actual project structure and requirements.)

1. To train the model:

```
python train.py --model deepsegpro --data_path /path/to/dataset --epochs 100
```

2. To evaluate the model:

```
python evaluate.py --model deepsegpro --weights /path/to/trained/weights --data_path /path/to/test/data
```

3. To run inference on new images:

```
python inference.py --model deepsegpro --weights /path/to/trained/weights --image /path/to/input/image
```

## Results

Below is a comparison of various models' performance on the kidney blood vessel segmentation task:

| Model          | Dice Score | Accuracy | Sensitivity | Specificity |
|----------------|------------|----------|-------------|-------------|
| U-Net (19)     | 0.621      | 95.01    | 95.38       | 92.64       |
| V-Net (22)     | 0.586      | 91.01    | 95.15       | 71.59       |
| SegNet (39)    | 0.522      | 89.02    | 95.09       | 62.36       |
| DeepLab (40)   | 0.593      | 91.01    | 95.32       | 65.75       |
| AttenU-net (41)| 0.642      | 93.11    | 96.53       | 70.58       |
| ResDO-UNet (42)| 65.13      | 94.41    | 98.12       | 73.33       |
| SegFormer (38) | 69.27      | 97.61    | 98.19       | 93.22       |
| DeepLabV3+ (9) | 73.66      | 98.60    | 99.31       | 93.75       |
| DeepSegPro (Ours) | 91.18   | 99.20    | 99.54       | 96.72       |

As shown in the table, our proposed model DeepSegPro outperforms other state-of-the-art models across all metrics.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to the Kaggle community for providing the dataset.
- We acknowledge the authors of the baseline models used in our comparison.