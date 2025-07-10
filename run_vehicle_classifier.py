"""

run_vehicle_classifer.py

Inference script for PyTorch Lightning models trained with train_vehicle_classifier.py.
Runs inference on a folder, producing a .json file in the MegaDetector batch output format
(https://lila.science/megadetector-output-format).

Relies on train_vehicle_classifier.py for core classes.

"""

#%% Imports and constants

import os
import argparse
import json
import sys

from pathlib import Path

import torch
import pandas as pd
import pytorch_lightning as pl

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from train_vehicle_classifier import VehicleClassifier, VehicleDataModule # type: ignore


#%% Classes

class InferenceDataset(Dataset):
    """
    Dataset for inference on images
    """

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return {
                'pixel_values': image,
                'path': str(image_path)
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy tensor if image fails to load
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            else:
                dummy_image = transforms.ToTensor()(dummy_image)
            return {
                'pixel_values': dummy_image,
                'path': str(image_path)
            }


#%% Support functions

def find_images(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """
    Recursively find all image files in directory.
    """

    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(directory).rglob(f'*{ext}'))
        image_paths.extend(Path(directory).rglob(f'*{ext.upper()}'))
    return sorted(image_paths)


def get_transforms_from_checkpoint(checkpoint_path, input_size=None):
    """
    Get the appropriate transforms for inference from a checkpoint.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint
        input_size (tuple): Override input size (height, width) if specified

    Returns:
        transform: Transform pipeline for inference
    """

    try:
        # Load checkpoint to get preprocessing info
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'hyper_parameters' not in checkpoint:
            raise Exception("No hyperparameters found in checkpoint")

        hparams = checkpoint['hyper_parameters']

        # Extract preprocessing parameters (these should be present in new checkpoints)
        if input_size is not None:
            size = input_size
            print(f"Using provided input size: {size}")
        else:
            size = hparams.get('image_size')
            if size is None:
                raise Exception("image_size not found in checkpoint hyperparameters. This " + \
                                "model may have been trained with an older script.")
            print(f"Using image size from checkpoint: {size}")

        mean = hparams.get('normalization_mean')
        std = hparams.get('normalization_std')

        if mean is None or std is None:
            raise Exception("normalization_mean or normalization_std not found in checkpoint " + \
                            "hyperparameters. This model may have been trained with an older script.")

        print(f"Using normalization from checkpoint: mean {mean}, std {std}")

    except Exception as e:
        raise Exception(f"Failed to extract preprocessing parameters from checkpoint: {e}")

    # Create transform pipeline
    transform_list = [
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    transform = transforms.Compose(transform_list)
    print(f"Created transform pipeline: {transform}")

    return transform


def load_model_from_checkpoint(checkpoint_path, class_names=None):
    """
    Load model from checkpoint, extracting necessary parameters.

    Args:
        checkpoint_path: Path to the checkpoint file
        class_names: List of class names (if not provided, will use generic names)

    Returns:
        tuple: (model, class_to_idx, idx_to_class, model_name)
    """

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint to extract metadata
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint loaded successfully")
    except Exception as e:
        raise Exception(f"Failed to load checkpoint: {e}")

    # Extract hyperparameters
    if 'hyper_parameters' not in checkpoint:
        raise Exception("No hyperparameters found in checkpoint. This checkpoint may be from an older training script.")

    hparams = checkpoint['hyper_parameters']
    print(f"Found hyperparameters: {list(hparams.keys())}")

    # Extract required parameters
    model_name = hparams.get('model_name', 'unknown_model')
    num_classes = hparams.get('num_classes')
    saved_class_to_idx = hparams.get('class_to_idx', {})

    if num_classes is None:
        raise Exception("num_classes not found in checkpoint hyperparameters")

    print(f"Model: {model_name}")
    print(f"Number of classes: {num_classes}")

    # Create class mappings
    if class_names:
        if len(class_names) != num_classes:
            raise Exception(f"Provided class names ({len(class_names)}) don't match model classes ({num_classes})")

        # Use provided class names
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        idx_to_class = {idx: cls for idx, cls in enumerate(class_names)}
        print(f"Using provided class names: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")

    elif saved_class_to_idx:
        # Use class mapping from checkpoint
        class_to_idx = saved_class_to_idx
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        print(f"Using class names from checkpoint: {list(class_to_idx.keys())[:5]}{'...' if len(class_to_idx) > 5 else ''}")

    else:
        # Fallback to generic class names
        class_to_idx = {f"class_{i}": i for i in range(num_classes)}
        idx_to_class = {i: f"class_{i}" for i in range(num_classes)}
        print(f"Using generic class names: class_0, class_1, ..., class_{num_classes-1}")

    # Load the model using Lightning's load_from_checkpoint
    try:
        model = VehicleClassifier.load_from_checkpoint(
            checkpoint_path,
            model_name=model_name,
            num_classes=num_classes,
            class_to_idx=class_to_idx
        )
        print("Model loaded successfully")
        return model, class_to_idx, idx_to_class, model_name

    except Exception as e:
        raise Exception(f"Failed to load model from checkpoint: {e}")


def load_class_names(class_file):
    """
    Load class names from file
    """

    try:
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(class_names)} class names from {class_file}")
        return class_names
    except Exception as e:
        raise Exception(f"Failed to load class names from {class_file}: {e}")


#%% Main inference function

def run_inference(checkpoint_path,
                  input_dir,
                  output_file,
                  batch_size=32,
                  num_workers=4,
                  input_size=None,
                  class_file=None,
                  output_absolute_filenames=False):
    """
    Run inference on a folder of images (recursively) using a trained model

    Args:
        checkpoint_path (str): path to the trained model checkpoint
        input_dir (str): directory containing images to process
        output_file (str): path for output JSON file
        batch_size (int, optional): batch size for inference
        num_workers (int, optional): number of data loader workers
        input_size (tuple, optional): override input size as (height, width) tuple
        class_file (str, optional): path to file containing class names (one per line)
        output_absolute_filenames (bool, optional): output absolute filenames instead of
            relative paths
    """

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Load class names if provided
    class_names = None
    if class_file:
        class_names = load_class_names(class_file)

    # Load model from checkpoint
    model, class_to_idx, idx_to_class, model_name = load_model_from_checkpoint(checkpoint_path, class_names)
    model = model.to(device)
    model.eval()

    num_classes = len(class_to_idx)

    # Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Get transforms
    transform = get_transforms_from_checkpoint(checkpoint_path, input_size)

    # Find all images
    print(f"Scanning for images in: {input_dir}")
    image_paths = find_images(input_dir)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found!")
        return

    # Create dataset and dataloader
    dataset = InferenceDataset(image_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Prepare MegaDetector format output structure
    megadetector_output = {
        "info": {
            "format_version": "1.4"
        },
        "detection_categories": {
            "0": "object"
        },
        "classification_categories": {
            str(i): idx_to_class[i] for i in range(num_classes)
        },
        "images": []
    }

    # Run inference
    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Processing images"):
            pixel_values = batch['pixel_values'].to(device)
            paths = batch['path']

            # Forward pass
            logits = model(pixel_values)
            predictions = torch.nn.functional.softmax(logits, dim=-1)

            # Get top predictions for each image
            top_preds = torch.topk(predictions, k=min(5, num_classes), dim=-1)

            for i, path in enumerate(paths):
                top_classes = top_preds.indices[i].cpu().numpy()
                top_scores = top_preds.values[i].cpu().numpy()

                # Create classifications list in MegaDetector format
                classifications = []
                for class_idx, score in zip(top_classes, top_scores):
                    classifications.append([
                        str(int(class_idx)),  # category ID as string
                        float(score.item() if hasattr(score, 'item') else score)  # confidence as float
                    ])

                # Create single dummy detection covering entire image
                detection = {
                    "category": "0",  # "object" category
                    "conf": 1.0,      # detection confidence
                    "bbox": [0, 0, 1, 1],  # entire image
                    "classifications": classifications
                }

                # Convert path to appropriate format
                if output_absolute_filenames:
                    # Keep absolute path but use forward slashes
                    file_path = Path(path).as_posix()
                else:
                    # Make relative to input_dir and use forward slashes
                    file_path = Path(path).relative_to(Path(input_dir)).as_posix()

                # Create image entry
                image_entry = {
                    "file": file_path,
                    "detections": [detection]
                }

                megadetector_output["images"].append(image_entry)

        # ...for each batch

    # ...with torch.no_grad()

    # Save results in MegaDetector format
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(megadetector_output, f, indent=2)

    print(f"Processed {len(megadetector_output['images'])} images")
    print(f"Model: {model_name}")
    print(f"Classes: {list(idx_to_class.values())}")
    print(f"Results saved in MegaDetector batch output format")

# ...def run_inference(...)

#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(description='Run inference with trained PyTorch Lightning model')
    parser.add_argument('checkpoint_path', help='Path to trained model checkpoint (.ckpt file)')
    parser.add_argument('input_dir', help='Directory containing images')
    parser.add_argument('--output', '-o', default='inference_results.json',
                       help='Output file for results (default: inference_results.json)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    parser.add_argument('--input-size', nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'),
                       help='Override input size (e.g., --input-size 448 448)')
    parser.add_argument('--classes', '-c', help='Path to file containing class names (one per line)')
    parser.add_argument('--output-absolute-filenames', action='store_true',
                       help='Output absolute filenames instead of relative paths (default: relative to input directory)')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file does not exist: {args.checkpoint_path}")
        return 1

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    if args.classes and not os.path.exists(args.classes):
        print(f"Error: Class file does not exist: {args.classes}")
        return 1

    # Convert input size to tuple if provided
    input_size = tuple(args.input_size) if args.input_size else None

    try:
        run_inference(
            checkpoint_path=args.checkpoint_path,
            input_dir=args.input_dir,
            output_file=args.output,
            batch_size=args.batch_size,
            num_workers=args.workers,
            input_size=input_size,
            class_file=args.classes,
            output_absolute_filenames=args.output_absolute_filenames
        )
        return 0
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
