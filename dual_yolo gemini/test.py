import torch
import torch.nn as nn
import unittest
import sys
import os

# Add the parent directory (dual_yolo) to sys.path to allow imports
# This assumes test.py is inside dual_yolo directory and model.py is also there.
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir) # If test.py is in a subfolder like dual_yolo/tests
# For current structure where test.py is in dual_yolo/
project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, project_root) # This line might cause issues in some environments
                                 # It's often better to run tests from the workspace root
                                 # with `python -m dual_yolo.test` if dual_yolo is a package.
                                 # For now, let's assume direct import works if run from workspace root
                                 # or dual_yolo/ is in PYTHONPATH.

from dual_yolo.model import AddFusion, CatFusion, XFormerFusion, DualYOLO, C_FEATURES, XFORMER_D_MODEL, XFORMER_N_HEADS

# Configuration for tests
BATCH_SIZE = 2
INPUT_H, INPUT_W = 256, 256 # Example input dimensions
IN_CHANNELS_IMG = 3
NUM_CLASSES_SEG = 5 # Example segmentation classes

# Helper to print model summary (simplified version of torchinfo)
def print_model_summary(model, input_size_blue, input_size_white=None):
    print(f"Model: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")
    # More detailed summary would require hooks or iterating through named modules
    print("Layers (first few levels):")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20} {module.__class__.__name__:<30} Params: {num_params}")
    # Try a forward pass to check for runtime errors with dummy data
    try:
        if input_size_white:
            dummy_input_b = torch.randn(BATCH_SIZE, *input_size_blue)
            dummy_input_w = torch.randn(BATCH_SIZE, *input_size_white)
            _ = model(dummy_input_b, dummy_input_w)
        else:
            dummy_input = torch.randn(BATCH_SIZE, *input_size_blue) # For single input modules
            _ = model(dummy_input, dummy_input) # Fusion modules expect two inputs
        print("Dummy forward pass successful.")
    except Exception as e:
        print(f"Dummy forward pass FAILED: {e}")

class TestFusionModules(unittest.TestCase):
    def setUp(self):
        self.f_b = torch.randn(BATCH_SIZE, C_FEATURES, INPUT_H // 8, INPUT_W // 8)
        self.f_w = torch.randn(BATCH_SIZE, C_FEATURES, INPUT_H // 8, INPUT_W // 8)
        self.expected_shape = (BATCH_SIZE, C_FEATURES, INPUT_H // 8, INPUT_W // 8)

    def test_add_fusion_output_shape(self):
        module = AddFusion(in_channels=C_FEATURES)
        module.eval()
        with torch.no_grad():
            output = module(self.f_b, self.f_w)
        self.assertEqual(output.shape, self.expected_shape)
        print(f"AddFusion test passed. Output shape: {output.shape}")

    def test_cat_fusion_output_shape(self):
        module = CatFusion(in_channels=C_FEATURES)
        module.eval()
        with torch.no_grad():
            output = module(self.f_b, self.f_w)
        self.assertEqual(output.shape, self.expected_shape)
        print(f"CatFusion test passed. Output shape: {output.shape}")

    def test_xformer_fusion_output_shape(self):
        module = XFormerFusion(in_channels=C_FEATURES, d_model=XFORMER_D_MODEL, n_heads=XFORMER_N_HEADS)
        module.eval()
        with torch.no_grad():
            output = module(self.f_b, self.f_w)
        self.assertEqual(output.shape, self.expected_shape)
        print(f"XFormerFusion test passed. Output shape: {output.shape}")

class TestDualYOLOModel(unittest.TestCase):
    def setUp(self):
        self.dummy_blue_img = torch.randn(BATCH_SIZE, IN_CHANNELS_IMG, INPUT_H, INPUT_W)
        self.dummy_white_img = torch.randn(BATCH_SIZE, IN_CHANNELS_IMG, INPUT_H, INPUT_W)
        # Expected output is from the placeholder seg head: (B, NUM_CLASSES_SEG, H_out, W_out)
        # H_out, W_out are H/8, W/8 due to backbone downsampling by 8 for P3 features.
        self.expected_output_shape = (BATCH_SIZE, NUM_CLASSES_SEG, INPUT_H // 8, INPUT_W // 8)

    def _test_dualyolo_variant(self, fusion_type):
        print(f"\n--- Testing DualYOLO with {fusion_type.upper()} fusion ---")
        model = DualYOLO(fusion_type=fusion_type, 
                         num_classes=NUM_CLASSES_SEG,
                         backbone_in_channels=IN_CHANNELS_IMG,
                         C=C_FEATURES,
                         d_model_xfmr=XFORMER_D_MODEL,
                         n_heads_xfmr=XFORMER_N_HEADS)
        model.eval()
        with torch.no_grad():
            output_seg_logits = model(self.dummy_blue_img, self.dummy_white_img)
        
        self.assertEqual(output_seg_logits.shape, self.expected_output_shape)
        print(f"DualYOLO ({fusion_type}) test passed. Output shape: {output_seg_logits.shape}")
        
        # Print model summary for this variant
        print_model_summary(model, 
                              (IN_CHANNELS_IMG, INPUT_H, INPUT_W), 
                              (IN_CHANNELS_IMG, INPUT_H, INPUT_W))
        print("-----------------------------------------------------")

    def test_dualyolo_add_fusion(self):
        self._test_dualyolo_variant('add')

    def test_dualyolo_cat_fusion(self):
        self._test_dualyolo_variant('cat')

    def test_dualyolo_ctr_fusion(self):
        self._test_dualyolo_variant('ctr')


if __name__ == '__main__':
    print("Running tests for DualYOLO model and fusion modules...")
    
    # For imports to work correctly when running dual_yolo/test.py directly:
    # Ensure dual_yolo's parent directory is in PYTHONPATH or run with `python -m dual_yolo.test` from workspace root.
    # The following attempts to add parent dir to path if not already there.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(current_dir) # Assuming dual_yolo is a top-level dir in workspace
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)
    
    # Re-import after path modification if necessary, though ideally structure handles this.
    # from dual_yolo.model import AddFusion, CatFusion, XFormerFusion, DualYOLO, C_FEATURES, XFORMER_D_MODEL, XFORMER_N_HEADS

    # Run unit tests
    unittest.main(verbosity=2)

    # After tests, optionally print a summary of a specific model configuration
    print("\n--- Example Model Instantiation & Summary (ctr fusion) ---")
    example_model = DualYOLO(fusion_type='ctr', num_classes=NUM_CLASSES_SEG)
    print_model_summary(example_model, 
                          (IN_CHANNELS_IMG, INPUT_H, INPUT_W), 
                          (IN_CHANNELS_IMG, INPUT_H, INPUT_W))
    print("---------------------------------------------------------")
    print("To generate a more detailed model graph, consider using torchinfo or netron.")
    print("E.g., with torchinfo: from torchinfo import summary; summary(model, input_size=[...])")
