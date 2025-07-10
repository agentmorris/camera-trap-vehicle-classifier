"""

train_vehicle_classifier.py

PyTorch Lightning script for fine-tuning vision models for vehicle classification.

Supports timm and Hugging Face models.

Input data is provided as:

* A root image path
* A COCO .json file containing relative filenames within that root path, with a "split" field in each
  image set to either "train" or "val.

See run_vehicle_classifier.py for the corresponding inference script.

"""

#%% Imports and constants

import os
import argparse
import pandas as pd
import csv
import json

from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import timm
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score


#%% Classes

class VehicleDataset(Dataset):
    """
    Dataset for vehicle classification
    """

    def __init__(self, metadata_df: pd.DataFrame, root_dir: str, transform=None):
        self.metadata_df = metadata_df
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]

        # Load image
        image_path = self.root_dir / row['image_path']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get label index
        label = row['label_idx']

        return image, label

# ...class VehicleDataset


class VehicleDataModule(pl.LightningDataModule):
    """
    Data module for vehicle classification
    """

    def __init__(
        self,
        metadata_file: str,
        root_dir: str,
        model_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.metadata_file = metadata_file
        self.root_dir = root_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load metadata and create class mappings
        self.set_up_data()

    def set_up_data(self):
        """
        Load COCO metadata and create class mappings
        """

        print(f"Loading COCO metadata from: {self.metadata_file}")

        # Load COCO JSON file
        with open(self.metadata_file, 'r') as f:
            coco_data = json.load(f)

        # Validate required fields
        required_fields = ['images', 'categories', 'annotations']
        for field in required_fields:
            if field not in coco_data:
                raise ValueError(f"COCO file missing required field: {field}")

        # Create category mappings
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        print(f"Found {len(categories)} categories: {list(categories.values())}")

        # Create image mappings
        images = {img['id']: img for img in coco_data['images']}

        # Validate splits
        valid_splits = {'train', 'val'}
        for img in coco_data['images']:
            if 'split' not in img:
                raise ValueError(f"Image {img['id']} missing 'split' field")
            if img['split'] not in valid_splits:
                raise ValueError(f"Image {img['id']} has invalid split '{img['split']}'. Must be 'train' or 'val'")

        # Create image_id to category mapping via annotations
        image_to_category = {}

        print('Mapping annotations')

        for ann in tqdm(coco_data['annotations']):

            image_id = ann['image_id']
            category_id = ann['category_id']

            if image_id in image_to_category:
                # Multiple annotations per image - for now, we'll take the first one
                # You might want to handle this differently depending on your use case
                continue

            if category_id not in categories:
                raise ValueError(f"Annotation references unknown category_id: {category_id}")

            image_to_category[image_id] = categories[category_id]

        # ...for each annotation

        # Build metadata list
        metadata_list = []
        missing_files = []

        print('Building image list and verifying image existence')

        for img_id, img_info in tqdm(images.items(),total=len(images)):

            if img_id not in image_to_category:
                print(f"Warning: Image {img_id} has no annotations, skipping")
                continue

            file_path = Path(self.root_dir) / img_info['file_name']

            # Check if file exists
            if not file_path.exists():
                missing_files.append(str(file_path))
                print('Warning: file {} is not available'.format(file_path))
                continue

            metadata_list.append({
                'image_path': img_info['file_name'],
                'category': image_to_category[img_id],
                'split': img_info['split']
            })

        # ...for each image

        # Assert all files exist
        if missing_files:
            print(f"Error: {len(missing_files)} image files do not exist:")
            for missing_file in missing_files[:10]:  # Show first 10
                print(f"  {missing_file}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
            raise FileNotFoundError(f"{len(missing_files)} image files are missing")

        print('Converting to dataframe')
        self.metadata_df = pd.DataFrame(metadata_list)

        if len(self.metadata_df) == 0:
            raise ValueError("No valid images found in COCO file")

        # Create class mappings
        unique_categories = sorted(self.metadata_df['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_categories)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(unique_categories)

        # Add label indices to metadata
        self.metadata_df['label_idx'] = self.metadata_df['category'].map(self.class_to_idx)

        print(f"Successfully loaded {len(self.metadata_df)} images")
        print(f"Found {self.num_classes} classes: {unique_categories}")

        # Print dataset sizes by split
        print(f"Dataset sizes:")
        for split in ['train', 'val']:
            split_df = self.metadata_df[self.metadata_df['split'] == split]
            count = len(split_df)
            print(f"  {split}: {count}")

            # Print category distribution for this split
            if count > 0:
                category_counts = split_df['category'].value_counts()
                print(f"    Category distribution in {split}:")
                for category, count in category_counts.items():
                    print(f"      {category}: {count}")

    # ...def set_up_data(...)

    def setup(self, stage=None):
        """
        Set up transforms and datasets
        """

        # Get the correct input size for the model
        is_timm_model = self.model_name.startswith('timm/')

        print(f"Setting up data for model: {self.model_name}")
        print(f"Is timm model: {is_timm_model}")

        if is_timm_model:

            # For timm models, get size from model's default_cfg
            timm_model_name = self.model_name[5:]  # Remove 'timm/' prefix
            print(f"Timm model name (without prefix): {timm_model_name}")

            try:
                # Create model to get its default config
                temp_model = timm.create_model(timm_model_name, pretrained=False)
                if hasattr(temp_model, 'default_cfg') and 'input_size' in temp_model.default_cfg:
                    input_size = temp_model.default_cfg['input_size']
                    size = (input_size[1], input_size[2])  # (height, width)
                    print(f"Using timm model input size from default_cfg: {size}")
                else:
                    # Fallback: force 448 for EVA02-448 models
                    if '448' in timm_model_name:
                        size = (448, 448)
                        print(f"Using hardcoded size for 448 model: {size}")
                    else:
                        size = (224, 224)  # fallback
                        print(f"Using default fallback size: {size}")
                del temp_model  # Clean up
            except Exception as e:
                print(f"Error getting timm config: {e}")
                # Fallback: force 448 for EVA02-448 models
                if '448' in timm_model_name:
                    size = (448, 448)
                    print(f"Using hardcoded size for 448 model: {size}")
                else:
                    size = (224, 224)  # fallback
                    print(f"Using default fallback size: {size}")

        else:

            # For Hugging Face models, use processor config
            processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
            if hasattr(processor, 'size'):
                if isinstance(processor.size, dict):
                    size = (processor.size.get('height', 224), processor.size.get('width', 224))
                else:
                    size = (processor.size, processor.size)
            else:
                size = (224, 224)  # fallback
            print(f"Using Hugging Face model input size: {size}")

        # ...is this a timm model or a Hugging Face model?

        # Get normalization values; try to get from timm if it's a timm model
        if is_timm_model:

            timm_model_name = self.model_name[5:]
            try:
                # Try to get normalization from timm model default_cfg
                temp_model = timm.create_model(timm_model_name, pretrained=False)
                if hasattr(temp_model, 'default_cfg'):
                    mean = temp_model.default_cfg.get('mean', [0.485, 0.456, 0.406])
                    std = temp_model.default_cfg.get('std', [0.229, 0.224, 0.225])
                    print(f"Using timm normalization - mean: {mean}, std: {std}")
                else:
                    raise Exception("No default_cfg found")
                del temp_model  # Clean up
            except:
                # Fallback to processor or defaults
                try:
                    processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
                    mean = processor.image_mean if hasattr(processor, 'image_mean') else [0.485, 0.456, 0.406]
                    std = processor.image_std if hasattr(processor, 'image_std') else [0.229, 0.224, 0.225]
                    print(f"Using processor normalization - mean: {mean}, std: {std}")
                except:
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                    print(f"Using default normalization - mean: {mean}, std: {std}")

        else:

            processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
            mean = processor.image_mean if hasattr(processor, 'image_mean') else [0.485, 0.456, 0.406]
            std = processor.image_std if hasattr(processor, 'image_std') else [0.229, 0.224, 0.225]
            print(f"Using HF processor normalization - mean: {mean}, std: {std}")

        # ...is this a timm model or a Hugging Face model?

        print(f"Final transform size: {size}")

        # Store preprocessing parameters for saving to checkpoint
        self.image_size = size
        self.normalization_mean = mean
        self.normalization_std = std

        # Training transforms with augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Validation/test transforms (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        print(f"Train transform: {self.train_transform}")
        print(f"Val transform: {self.val_transform}")

        # Create datasets
        train_df = self.metadata_df[self.metadata_df['split'] == 'train']
        val_df = self.metadata_df[self.metadata_df['split'] == 'val']

        self.train_dataset = VehicleDataset(train_df, self.root_dir, self.train_transform)
        self.val_dataset = VehicleDataset(val_df, self.root_dir, self.val_transform)

    # ...def setup(...)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        # We don't support a test dataset in this module
        return None

# ...class VehicleDataModule


class VehicleClassifier(pl.LightningModule):
    """
    Lightning module for vehicle classification
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        class_to_idx: Dict[str, int],
        learning_rate: float = 1e-4,
        freeze_layers: bool = True,
        output_dir: str = None
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_to_idx = class_to_idx
        self.learning_rate = learning_rate
        self.freeze_layers = freeze_layers
        self.output_dir = output_dir

        # Save hyperparameters for checkpoint loading (exclude output_dir as it's training-specific)
        self.save_hyperparameters(ignore=['output_dir'])

        # Create idx_to_class mapping for convenience
        self.idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

        # Initialize preprocessing parameters (will be set by data module)
        self.image_size = None
        self.normalization_mean = None
        self.normalization_std = None

        # Load model
        self.set_up_model()

        # For tracking metrics
        self.validation_step_outputs = []

        # Create metrics CSV file and save class names
        if self.output_dir:
            self.metrics_file = os.path.join(self.output_dir, 'training_metrics.csv')

            # Save class names to text file
            self._save_class_names()

            # Save detailed class mapping as JSON
            self._save_class_mapping()

    def set_up_metrics_file(self, resume_from=None):
        """
        Setup metrics CSV file, handling resume case
        """

        if not self.output_dir:
            return

        # Check whether we're resuming
        if resume_from and os.path.exists(self.metrics_file):
            print(f"Resuming training - will append to existing metrics file: {self.metrics_file}")
        else:
            # Initialize new CSV file
            print(f"Creating new metrics file: {self.metrics_file}")
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_macro_accuracy'])

    def _save_class_names(self):
        """
        Save class names to classes.txt file in output directory
        """

        if self.output_dir:
            classes_file = os.path.join(self.output_dir, 'classes.txt')
            with open(classes_file, 'w') as f:
                # Write class names in order of their indices
                for i in range(self.num_classes):
                    f.write(f"{self.idx_to_class[i]}\n")
            print(f"Saved class names to: {classes_file}")

    def _save_class_mapping(self):
        """
        Save detailed class mapping and metadata as JSON
        """

        if self.output_dir:
            metadata = {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class,
                'class_names': [self.idx_to_class[i] for i in range(self.num_classes)],
                'learning_rate': self.learning_rate,
                'freeze_layers': self.freeze_layers,
                'preprocessing': {
                    'image_size': getattr(self, 'image_size', None),
                    'normalization_mean': getattr(self, 'normalization_mean', None),
                    'normalization_std': getattr(self, 'normalization_std', None)
                },
                'training_info': {
                    'framework': 'pytorch_lightning',
                    'model_type': 'fine_tuned_classifier',
                    'metadata_format': 'coco_json'
                }
            }

            metadata_file = os.path.join(self.output_dir, 'model_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved model metadata to: {metadata_file}")

    def on_train_start(self):
        """
        Called when training starts, saves preprocessing parameters
        """

        # Get preprocessing parameters from data module
        if hasattr(self.trainer, 'datamodule'):
            dm = self.trainer.datamodule
            if hasattr(dm, 'image_size'):
                self.image_size = dm.image_size
                self.normalization_mean = dm.normalization_mean
                self.normalization_std = dm.normalization_std

                # Update hyperparameters to include preprocessing info
                if hasattr(self, 'hparams'):
                    self.hparams.update({
                        'image_size': self.image_size,
                        'normalization_mean': self.normalization_mean,
                        'normalization_std': self.normalization_std
                    })

                print(f"Saved preprocessing parameters:")
                print(f"  Image size: {self.image_size}")
                print(f"  Mean: {self.normalization_mean}")
                print(f"  Std: {self.normalization_std}")

                # Re-save metadata with preprocessing info
                self._save_class_mapping()

    def set_up_model(self):
        """
        Set up the model with appropriate freezing'
        """

        is_timm_model = self.model_name.startswith('timm/')

        if is_timm_model:
            # Use timm model
            timm_model_name = self.model_name[5:]  # Remove 'timm/' prefix
            self.model = timm.create_model(timm_model_name, pretrained=True, num_classes=self.num_classes)

            if self.freeze_layers:
                self._freeze_timm_layers()

        else:
            # Use Hugging Face model
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True
            )

            if self.freeze_layers:
                self._freeze_hf_layers()

        # ...is this a timm model or a Hugging Face model?

    def _freeze_timm_layers(self):
        """
        Freeze layers for timm models (keep classifier head + last 2-4 blocks unfrozen)
        """

        print("Freezing timm model layers...")

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier head
        if hasattr(self.model, 'head'):
            for param in self.model.head.parameters():
                param.requires_grad = True

        # For EVA02 models, unfreeze last few blocks
        if hasattr(self.model, 'blocks'):
            num_blocks = len(self.model.blocks)
            blocks_to_unfreeze = 3  # Unfreeze last 3 blocks

            for i in range(max(0, num_blocks - blocks_to_unfreeze), num_blocks):
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = True

            print(f"Unfroze classifier head and last {blocks_to_unfreeze} blocks out of {num_blocks}")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    def _freeze_hf_layers(self):
        """
        Freeze layers for Hugging Face models
        """

        print("Freezing Hugging Face model layers...")

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True

        # For ViT models, unfreeze last few encoder layers
        if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'encoder'):
            layers = self.model.vit.encoder.layer
            layers_to_unfreeze = 3  # Unfreeze last 3 layers

            for i in range(max(0, len(layers) - layers_to_unfreeze), len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True

            print(f"Unfroze classifier and last {layers_to_unfreeze} encoder layers out of {len(layers)}")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    def forward(self, x):
        if self.model_name.startswith('timm/'):
            return self.model(x)
        else:
            return self.model(x).logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.validation_step_outputs.append({
            'val_loss': loss,
            'preds': preds,
            'labels': labels
        })

        return loss

    def on_validation_epoch_end(self):

        # Compute metrics
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()

        # Overall accuracy
        accuracy = accuracy_score(all_labels.cpu(), all_preds.cpu())

        # Macro accuracy (per-class accuracy averaged)
        class_accuracies = []
        for class_idx in range(self.num_classes):
            mask = all_labels == class_idx
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).float().mean()
                class_accuracies.append(class_acc.item())

        macro_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0

        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_macro_accuracy', macro_accuracy, prog_bar=True)

        # Write to CSV
        if self.output_dir:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                train_loss = self.trainer.callback_metrics.get('train_loss', 0.0)
                writer.writerow([
                    self.current_epoch,
                    float(train_loss),
                    float(avg_loss),
                    float(accuracy),
                    float(macro_accuracy)
                ])

        # Clear outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):

        # Use different learning rates for different parts
        if self.freeze_layers:
            # Higher learning rate for unfrozen layers
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=0.01
            )
        else:
            # Lower learning rate for full fine-tuning
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate / 10,
                weight_decay=0.01
            )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate / 100
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

# ...class VehicleClassifier


#%% Utility functions

def strip_checkpoint(checkpoint_path, output_path=None, keep_hyperparams=True):
    """
    Strip optimizer state and training metadata from a Lightning checkpoint,
    keeping only the model weights and essential metadata for inference.

    Args:
        checkpoint_path (str): path to the original checkpoint
        output_path (str, optional): path for the stripped checkpoint (if None, adds '_stripped' suffix)
        keep_hyperparams (bool, optional): whether to keep hyperparameters (recommended for inference)
    """

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Calculate original size
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)

    # Create stripped checkpoint with only essential components
    stripped_checkpoint = {
        'state_dict': checkpoint['state_dict'],  # model weights
        'pytorch-lightning_version': checkpoint.get('pytorch-lightning_version'),
    }

    # Optionally keep hyperparameters (useful for inference)
    if keep_hyperparams and 'hyper_parameters' in checkpoint:
        stripped_checkpoint['hyper_parameters'] = checkpoint['hyper_parameters']

    # Keep epoch info for reference
    if 'epoch' in checkpoint:
        stripped_checkpoint['epoch'] = checkpoint['epoch']
    if 'global_step' in checkpoint:
        stripped_checkpoint['global_step'] = checkpoint['global_step']

    # Generate output path if not provided
    if output_path is None:
        base_path = Path(checkpoint_path)
        output_path = base_path.parent / f"{base_path.stem}_stripped{base_path.suffix}"

    # Save stripped checkpoint
    print(f"Saving stripped checkpoint: {output_path}")
    torch.save(stripped_checkpoint, output_path)

    # Calculate new size and savings
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    savings = original_size - new_size
    savings_percent = (savings / original_size) * 100

    print(f"Original size: {original_size:.1f} MB")
    print(f"Stripped size: {new_size:.1f} MB")
    print(f"Space saved: {savings:.1f} MB ({savings_percent:.1f}%)")

    return str(output_path)

# ...def strip_checkpoint(...)


def load_model_for_inference(checkpoint_path: str, map_location='cpu'):
    """
    Load a trained model for inference from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint (full or stripped)
        map_location: Device to load the model on

    Returns:
        Loaded VehicleClassifier model in eval mode
    """

    print(f"Loading model for inference from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Extract hyperparameters
    if 'hyper_parameters' not in checkpoint:
        raise ValueError("Checkpoint missing hyperparameters. Cannot determine model configuration.")

    hparams = checkpoint['hyper_parameters']

    # Create model instance
    model = VehicleClassifier(
        model_name=hparams['model_name'],
        num_classes=hparams['num_classes'],
        class_to_idx=hparams['class_to_idx'],
        learning_rate=hparams.get('learning_rate', 1e-4),
        freeze_layers=hparams.get('freeze_layers', True)
    )

    # Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"Model loaded successfully:")
    print(f"  Model: {hparams['model_name']}")
    print(f"  Classes: {hparams['num_classes']}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    return model

# ...def load_model_for_inference(...)


#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(description='Fine-tune vision models for vehicle classification')
    parser.add_argument('metadata_file', help='Path to COCO-formatted JSON metadata file')
    parser.add_argument('root_dir', help='Root directory containing images')
    parser.add_argument('output_dir', help='Output directory for checkpoints and logs')
    parser.add_argument('--model', '-m', default='timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
                       help='Model name (default: EVA02 Large)')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Maximum number of epochs (default: 50)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-workers', '-w', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    parser.add_argument('--no-freeze', action='store_true',
                       help='Do not freeze any layers (full fine-tuning)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--resume-from', help='Path to checkpoint to resume training from')
    parser.add_argument('--save-every-epoch', action='store_true',
                       help='Save a checkpoint after every epoch (in addition to best and last)')
    parser.add_argument('--strip-final-model', action='store_true',
                       help='Create a stripped version of the final best model (removes optimizer state)')
    parser.add_argument('--strip-checkpoint', type=str,
                       help='Strip an existing checkpoint and exit (provide checkpoint path)')

    args = parser.parse_args()

    # Handle checkpoint stripping mode (standalone operation)
    if args.strip_checkpoint:
        if not os.path.exists(args.strip_checkpoint):
            print(f"Error: Checkpoint does not exist: {args.strip_checkpoint}")
            return 1

        print("Stripping checkpoint...")
        stripped_path = strip_checkpoint(args.strip_checkpoint)
        print(f"Stripped checkpoint saved to: {stripped_path}")
        return 0

    # Validate resume checkpoint if provided
    if args.resume_from and not os.path.exists(args.resume_from):
        print(f"Error: Resume checkpoint does not exist: {args.resume_from}")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup data module
    data_module = VehicleDataModule(
        metadata_file=args.metadata_file,
        root_dir=args.root_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Set up model
    model = VehicleClassifier(
        model_name=args.model,
        num_classes=data_module.num_classes,
        class_to_idx=data_module.class_to_idx,
        learning_rate=args.learning_rate,
        freeze_layers=not args.no_freeze,
        output_dir=args.output_dir
    )

    # Set up metrics file (handle resume case)
    model.set_up_metrics_file(args.resume_from)

    # Setup callbacks
    callbacks = []

    # Best model checkpoint (always included)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='best-{epoch:02d}-{val_accuracy:.3f}',
        monitor='val_accuracy',
        mode='max',
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Every epoch checkpoint (optional)
    if args.save_every_epoch:
        print("Saving checkpoint after every epoch")
        every_epoch_callback = ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'epoch_checkpoints'),
            filename='epoch-{epoch:02d}-{val_accuracy:.3f}',
            save_top_k=-1,  # Save all checkpoints
            every_n_epochs=1,
            verbose=True
        )
        callbacks.append(every_epoch_callback)

        # Create subdirectory for epoch checkpoints
        os.makedirs(os.path.join(args.output_dir, 'epoch_checkpoints'), exist_ok=True)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=args.patience,
        verbose=True
    )
    callbacks.append(early_stopping)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        precision='16-mixed',  # Use mixed precision for efficiency
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    print(f"Training {args.model} on {data_module.num_classes} classes")
    print(f"Output directory: {args.output_dir}")
    print(f"Freeze layers: {not args.no_freeze}")
    print(f"Save every epoch: {args.save_every_epoch}")
    print(f"Strip final model: {args.strip_final_model}")
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")

    # Train model (with optional resume)
    trainer.fit(model, data_module, ckpt_path=args.resume_from)

    print(f"Training completed, best model saved to: {checkpoint_callback.best_model_path}")
    if args.save_every_epoch:
        print(f"Epoch checkpoints saved to: {os.path.join(args.output_dir, 'epoch_checkpoints')}")

    # Strip the final best model if requested
    if args.strip_final_model and checkpoint_callback.best_model_path:
        print("\nStripping final best model...")
        stripped_path = strip_checkpoint(checkpoint_callback.best_model_path)
        print(f"Stripped model saved to: {stripped_path}")

        # Optionally, you can also strip the last checkpoint
        last_checkpoint = os.path.join(args.output_dir, 'last.ckpt')
        if os.path.exists(last_checkpoint):
            print("Stripping last checkpoint...")
            stripped_last_path = strip_checkpoint(last_checkpoint)
            print(f"Stripped last checkpoint saved to: {stripped_last_path}")


if __name__ == '__main__':
    main()
