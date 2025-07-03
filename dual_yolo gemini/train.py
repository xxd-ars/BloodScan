import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import time
import yaml # For loading data config

# Add project root for module imports if running script directly
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir) # BloodScan/
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from dual_yolo.model import DualYOLO, C_FEATURES, XFORMER_D_MODEL, XFORMER_N_HEADS

# --- Configuration --- 
# These would typically be in a config file or more advanced arg parsing
DEFAULT_IMAGE_SIZE = 640 # Placeholder, should match dataset
DEFAULT_NUM_CLASSES = 5  # Example: background, layer1, layer2, layer3, layer4

# --- Dummy Dataset (Placeholder) ---
class BloodSampleDataset(Dataset):
    """Placeholder for blood sample dataset."""
    def __init__(self, data_config_path, image_size=DEFAULT_IMAGE_SIZE, split='train'):
        self.image_size = image_size
        self.split = split
        # In a real scenario, parse data_config_path (e.g., a YAML file)
        # to find image paths, annotations, etc.
        # For example:
        # with open(data_config_path, 'r') as f:
        #     data_cfg = yaml.safe_load(f)
        # self.image_files_blue = data_cfg[split]['blue_images']
        # self.image_files_white = data_cfg[split]['white_images']
        # self.label_files = data_cfg[split]['labels']
        
        print(f"[Dataset] Initializing {split} dataset. Image size: {image_size}x{image_size}")
        print(f"[Dataset] WARN: Using DUMMY data. Replace with actual data loading.")
        self.num_samples = 100 if split == 'train' else 20 # Dummy number of samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy data generation
        img_blue = torch.randn(3, self.image_size, self.image_size)
        img_white = torch.randn(3, self.image_size, self.image_size)
        
        # Segmentation mask: (NumClasses, H, W) for cross-entropy or (1, H, W) for BCE with logits
        # Assuming output of model is (B, NumClasses, H_feat, W_feat)
        # Target mask should be (B, H_feat, W_feat) with class indices for nn.CrossEntropyLoss
        # Or (B, NumClasses, H_feat, W_feat) for other losses like Dice.
        # For nn.CrossEntropyLoss, target is (N, d1, d2, ...)
        # Model output: (N, C, d1, d2, ...) where C = num_classes
        # Target for CE: (N, H_out, W_out) where H_out=image_size//8, W_out=image_size//8
        # Values in target should be class indices from 0 to NumClasses-1.
        target_h, target_w = self.image_size // 8, self.image_size // 8
        # seg_mask = torch.randint(0, DEFAULT_NUM_CLASSES, (target_h, target_w), dtype=torch.long)
        # If using a loss that expects one-hot or per-class probability maps:
        seg_mask_one_hot = torch.rand(DEFAULT_NUM_CLASSES, target_h, target_w) 
        # For CrossEntropy, we need class indices. Let's use the one-hot version and convert for CE.
        seg_mask_indices = torch.argmax(seg_mask_one_hot, dim=0).long()

        return img_blue, img_white, seg_mask_indices # Using index mask for CrossEntropyLoss

# --- Loss Function (Placeholder) ---
def get_loss_function(num_classes):
    """Returns the segmentation loss function."""
    # Common loss for semantic segmentation is CrossEntropyLoss
    # It expects model output (logits) of shape (B, NumClasses, H, W)
    # and target of shape (B, H, W) with class indices.
    criterion = nn.CrossEntropyLoss()
    print(f"[Loss] Using nn.CrossEntropyLoss for {num_classes} classes.")
    return criterion

# --- Training Loop ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, fusion_type):
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for i, (imgs_blue, imgs_white, masks) in enumerate(dataloader):
        imgs_blue, imgs_white, masks = imgs_blue.to(device), imgs_white.to(device), masks.to(device)

        optimizer.zero_grad()
        
        outputs = model(imgs_blue, imgs_white) # Model output: (B, NumClasses, H_feat, W_feat)
        
        # Ensure mask and output shapes are compatible for the loss
        # For CrossEntropyLoss: output (B, C, H, W), target (B, H, W)
        # print(f"Output shape: {outputs.shape}, Mask shape: {masks.shape}") # For debugging
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 20 == 0 or (i + 1) == num_batches: # Log every 20 batches or last batch
            print(f"  Epoch [{epoch+1}] Batch [{i+1}/{num_batches}] Loss: {loss.item():.4f}")
    
    avg_epoch_loss = running_loss / num_batches
    print(f"Epoch [{epoch+1}] Average Training Loss ({fusion_type}): {avg_epoch_loss:.4f}")
    return avg_epoch_loss

# --- Main Training Function ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    # Replace with actual path to your data configuration file (e.g., dataset.yaml)
    # For now, we pass a dummy path which the dummy dataset won't use.
    dummy_data_config = "path/to/your/dataset_config.yaml" 
    if args.data_config and os.path.exists(args.data_config):
        data_config_path = args.data_config
    else:
        print(f"Warning: Data config {args.data_config} not found. Using dummy path: {dummy_data_config}")
        data_config_path = dummy_data_config

    train_dataset = BloodSampleDataset(data_config_path=data_config_path, image_size=args.img_size, split='train')
    # val_dataset = BloodSampleDataset(data_config_path=data_config_path, image_size=args.img_size, split='val') # Placeholder

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers) # Placeholder
    print(f"Train loader: {len(train_loader)} batches. Val loader: Placeholder")

    # --- Model Initialization ---
    print(f"Initializing DualYOLO model with {args.fusion_type} fusion...")
    model = DualYOLO(fusion_type=args.fusion_type, 
                     num_classes=args.num_classes,
                     # yolo_cfg_path relevant if using more integrated YOLO neck/head
                     backbone_in_channels=3, 
                     C=C_FEATURES,
                     d_model_xfmr=XFORMER_D_MODEL,
                     n_heads_xfmr=XFORMER_N_HEADS)
    model.to(device)

    # --- Optimizer and Loss Function ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Example scheduler
    criterion = get_loss_function(num_classes=args.num_classes)

    # --- Training ---
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.fusion_type)
        
        # Placeholder for validation loop
        # if val_loader:
        #     evaluate_one_epoch(model, val_loader, criterion, device, epoch)
        
        # Placeholder for learning rate scheduling
        # scheduler.step()

        # Placeholder for saving model checkpoints
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f"dualyolo_{args.fusion_type}_epoch_{epoch+1}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'args': args
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    total_training_time = time.time() - start_time
    print(f"Training finished in {total_training_time // 3600:.0f}h {(total_training_time % 3600) // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"Final model weights might be saved at: {os.path.join(args.output_dir, f'dualyolo_{args.fusion_type}_final.pt')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DualYOLO Training Script")
    parser.add_argument("--data_config", type=str, default="config/dataset.yaml", help="Path to data configuration YAML file.")
    parser.add_argument("--fusion_type", type=str, default="ctr", choices=['add', 'cat', 'ctr'], help="Fusion type for DualYOLO.")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="Number of segmentation classes.")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Input image size.")
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")
    
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--output_dir", type=str, default="runs/train", help="Directory to save checkpoints and logs.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    args = parser.parse_args()
    
    print("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("---")

    main(args)
