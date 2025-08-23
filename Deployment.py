
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import time
from pathlib import Path


class BugDetectionEvaluator:
    """Evaluate bug detection model on real test images"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.transform = None
        self.class_names = {0: 'background', 1: 'bug'}  # Adjust based on your classes
        
        self._load_model()
        self._setup_transform()
        
    def _load_model(self):
        """Load the trained model"""
        print(f"🔄 Loading model from: {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            print(f"✅ Checkpoint loaded successfully")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint and hasattr(checkpoint['model'], 'eval'):
                    # Complete model object
                    self.model = checkpoint['model']
                    print("✅ Loaded complete model object")
                else:
                    # State dict - try to create compatible model
                    if 'model_state_dict' in checkpoint:
                        model_state = checkpoint['model_state_dict']
                        print(f"📊 Epoch: {checkpoint.get('epoch', 'Unknown')}")
                        if 'config' in checkpoint:
                            print(f"📋 Config: {checkpoint['config']}")
                    else:
                        model_state = checkpoint
                    
                    # Create model architecture
                    from torchvision.models.detection import fasterrcnn_resnet50_fpn
                    
                    # Remove 'model.' prefix if present
                    clean_state = {}
                    for key, value in model_state.items():
                        if key.startswith('model.'):
                            clean_state[key[6:]] = value
                        else:
                            clean_state[key] = value
                    
                    # Detect number of classes
                    num_classes = 2
                    for key in clean_state.keys():
                        if 'roi_heads.box_predictor.cls_score.weight' in key:
                            num_classes = clean_state[key].shape[0]
                            break
                    
                    print(f"🎯 Detected {num_classes} classes")
                    
                    # Create model - avoid downloading pretrained weights
                    self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
                    
                    # Load state dict
                    missing_keys, unexpected_keys = self.model.load_state_dict(clean_state, strict=False)
                    if missing_keys:
                        print(f"⚠️  Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                    
                    print("✅ Model architecture created and weights loaded")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Test forward pass
            test_input = torch.randn(3, 224, 224).to(self.device)
            with torch.no_grad():
                test_output = self.model([test_input])
            print("✅ Model test successful - ready for inference!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
            
    def _setup_transform(self):
        """Setup image transforms"""
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def detect_bugs_in_image(self, image_path, conf_threshold=0.5):
        """Detect bugs in a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            tensor_image = self.transform(image)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model([tensor_image.to(self.device)])
            
            # Process predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            # Filter by confidence threshold
            keep_idx = scores >= conf_threshold
            filtered_boxes = boxes[keep_idx]
            filtered_scores = scores[keep_idx]
            filtered_labels = labels[keep_idx]
            
            return {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'original_size': original_size,
                'detections': {
                    'boxes': filtered_boxes.tolist(),
                    'scores': filtered_scores.tolist(),
                    'labels': filtered_labels.tolist(),
                    'count': len(filtered_boxes)
                },
                'all_detections': {  # All detections regardless of threshold
                    'boxes': boxes.tolist(),
                    'scores': scores.tolist(),
                    'labels': labels.tolist(),
                    'count': len(boxes)
                }
            }, image
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            return None, None
    
    def visualize_detections(self, image, detection_result, show_all_detections=False, conf_threshold=0.5):
        """Create visualization of detections"""
        draw = ImageDraw.Draw(image)
        
        # Load font
        try:
            font_size = 12  # Fixed small font size
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                font = ImageFont.load_default()
        
        # Choose which detections to show
        if show_all_detections:
            boxes = detection_result['all_detections']['boxes']
            scores = detection_result['all_detections']['scores']
            labels = detection_result['all_detections']['labels']
        else:
            boxes = detection_result['detections']['boxes']
            scores = detection_result['detections']['scores']
            labels = detection_result['detections']['labels']
        
        # Color scheme
        colors = {
            'high_conf': 'red',      # > 0.8
            'medium_conf': 'orange', # 0.5 - 0.8
            'low_conf': 'yellow',    # < 0.5
            'background': 'blue'
        }
        
        detection_count = 0
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            
            # Skip background class
            if label == 0:
                continue
                
            detection_count += 1
            
            # Choose color based on confidence
            if score > 0.8:
                color = colors['high_conf']
                conf_level = "HIGH"
            elif score > 0.5:
                color = colors['medium_conf'] 
                conf_level = "MED"
            else:
                color = colors['low_conf']
                conf_level = "LOW"
            
            # Draw bounding box with thinner lines
            line_width = 1  # Fixed thin line width
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # Create label text
            class_name = self.class_names.get(label, f'Class_{label}')
            label_text = f"{class_name}: {score:.3f} ({conf_level})"
            
            # Get text dimensions
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width, text_height = draw.textsize(label_text, font=font)
            
            # Draw label background
            label_bg = [x1, y1-text_height-6, x1+text_width+8, y1-2]
            draw.rectangle(label_bg, fill=color)
            
            # Draw label text
            draw.text((x1+4, y1-text_height-4), label_text, fill='white', font=font)
        
        # Add summary text
        summary_text = f"Detections: {detection_count}"
        if show_all_detections:
            summary_text += f" (All predictions, threshold: {conf_threshold})"
        else:
            summary_text += f" (Conf ≥ {conf_threshold})"
            
        # Draw summary at top
        try:
            summary_bbox = draw.textbbox((0, 0), summary_text, font=font)
            summary_width = summary_bbox[2] - summary_bbox[0]
            summary_height = summary_bbox[3] - summary_bbox[1]
        except:
            summary_width, summary_height = draw.textsize(summary_text, font=font)
        
        draw.rectangle([10, 10, 10+summary_width+8, 10+summary_height+4], fill='black')
        draw.text((14, 12), summary_text, fill='white', font=font)
        
        return image
    
    def evaluate_on_test_set(self, test_dir, output_dir, conf_threshold=0.5, max_images=None):
        """Evaluate model on entire test set"""
        print(f"🔍 Evaluating bug detection on test images")
        print(f"📁 Test directory: {test_dir}")
        print(f"📊 Confidence threshold: {conf_threshold}")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'high_confidence'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'all_detections'), exist_ok=True)
        
        # Find test images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(test_dir).glob(f'*{ext}'))
            image_files.extend(Path(test_dir).glob(f'*{ext.upper()}'))
        
        if max_images:
            image_files = image_files[:max_images]
            
        print(f"📸 Found {len(image_files)} test images")
        
        if len(image_files) == 0:
            print("❌ No images found! Check the test directory path.")
            return
        
        # Process each image
        all_results = []
        images_with_bugs = 0
        total_bug_detections = 0
        confidence_distribution = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🖼️  Processing {i}/{len(image_files)}: {image_path.name}")
            
            # Detect bugs
            result, original_image = self.detect_bugs_in_image(str(image_path), conf_threshold)
            
            if result is None:
                continue
                
            all_results.append(result)
            
            # Count detections
            num_detections = result['detections']['count']
            if num_detections > 0:
                images_with_bugs += 1
                total_bug_detections += num_detections
                
            # Collect confidence scores
            confidence_distribution.extend(result['detections']['scores'])
            
            print(f"   🐛 Found {num_detections} bugs (conf ≥ {conf_threshold})")
            
            if result['detections']['scores']:
                max_conf = max(result['detections']['scores'])
                avg_conf = np.mean(result['detections']['scores'])
                print(f"   📊 Confidence - Max: {max_conf:.3f}, Avg: {avg_conf:.3f}")
            
            # Create visualizations
            if num_detections > 0:
                # High confidence detections
                vis_image = self.visualize_detections(
                    original_image.copy(), result, 
                    show_all_detections=False, conf_threshold=conf_threshold
                )
                vis_path = os.path.join(output_dir, 'high_confidence', f'detected_{image_path.name}')
                vis_image.save(vis_path)
                
                # All detections (including low confidence)
                vis_all = self.visualize_detections(
                    original_image.copy(), result,
                    show_all_detections=True, conf_threshold=0.1
                )
                vis_all_path = os.path.join(output_dir, 'all_detections', f'all_{image_path.name}')
                vis_all.save(vis_all_path)
                
                print(f"   💾 Saved visualizations")
            else:
                # Save original for no detections
                no_detect_path = os.path.join(output_dir, 'visualizations', f'no_bugs_{image_path.name}')
                original_image.save(no_detect_path)
        
        # Generate summary report
        self.generate_evaluation_report(all_results, output_dir, conf_threshold, confidence_distribution)
        
        print(f"\n🏆 EVALUATION SUMMARY")
        print(f"📸 Images processed: {len(all_results)}")
        print(f"🐛 Images with bugs detected: {images_with_bugs}")
        print(f"📊 Detection rate: {images_with_bugs/len(all_results)*100:.1f}%")
        print(f"🎯 Total bug detections: {total_bug_detections}")
        print(f"📈 Average bugs per image: {total_bug_detections/len(all_results):.2f}")
        
        if confidence_distribution:
            print(f"🔍 Confidence stats:")
            print(f"   Mean: {np.mean(confidence_distribution):.3f}")
            print(f"   Max: {np.max(confidence_distribution):.3f}")
            print(f"   Min: {np.min(confidence_distribution):.3f}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        
        return all_results
    
    def generate_evaluation_report(self, results, output_dir, conf_threshold, confidence_distribution):
        """Generate detailed evaluation report"""
        
        # Save detailed JSON results
        json_path = os.path.join(output_dir, 'detection_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary statistics
        summary = {
            'total_images': len(results),
            'images_with_detections': sum(1 for r in results if r['detections']['count'] > 0),
            'total_detections': sum(r['detections']['count'] for r in results),
            'confidence_threshold': conf_threshold,
            'detection_rate': sum(1 for r in results if r['detections']['count'] > 0) / len(results),
            'avg_detections_per_image': sum(r['detections']['count'] for r in results) / len(results),
            'confidence_stats': {
                'mean': float(np.mean(confidence_distribution)) if confidence_distribution else 0,
                'std': float(np.std(confidence_distribution)) if confidence_distribution else 0,
                'min': float(np.min(confidence_distribution)) if confidence_distribution else 0,
                'max': float(np.max(confidence_distribution)) if confidence_distribution else 0,
            } if confidence_distribution else None
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create detailed report
        report_lines = [
            "# Bug Detection Evaluation Report\n",
            f"**Model**: {self.model_path}",
            f"**Confidence Threshold**: {conf_threshold}",
            f"**Total Images**: {summary['total_images']}",
            f"**Images with Detections**: {summary['images_with_detections']}",
            f"**Detection Rate**: {summary['detection_rate']*100:.2f}%",
            f"**Total Detections**: {summary['total_detections']}",
            f"**Average Detections per Image**: {summary['avg_detections_per_image']:.2f}\n",
        ]
        
        if confidence_distribution:
            report_lines.extend([
                "## Confidence Statistics",
                f"- Mean Confidence: {summary['confidence_stats']['mean']:.3f}",
                f"- Standard Deviation: {summary['confidence_stats']['std']:.3f}",
                f"- Min Confidence: {summary['confidence_stats']['min']:.3f}",
                f"- Max Confidence: {summary['confidence_stats']['max']:.3f}\n",
            ])
        
        # Add per-image details
        report_lines.append("## Per-Image Results\n")
        for result in results:
            report_lines.append(f"**{result['image_name']}**:")
            report_lines.append(f"- Detections: {result['detections']['count']}")
            if result['detections']['scores']:
                avg_conf = np.mean(result['detections']['scores'])
                max_conf = max(result['detections']['scores'])
                report_lines.append(f"- Confidence: {avg_conf:.3f} (avg), {max_conf:.3f} (max)")
            report_lines.append("")
        
        # Save report
        report_path = os.path.join(output_dir, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📋 Detailed report saved to: {report_path}")


# Main deployment code
if __name__ == "__main__":
    MODEL_PATH = "enhanced_faster_rcnn_bug_detection_best.pth"
    IMAGE_FOLDER = "/home/pi/captured_images"
    OUTPUT_FOLDER = "/home/pi/detections_output"
    CONF_THRESHOLD = 0.5
    POLL_INTERVAL = 5

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "high_confidence"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "all_detections"), exist_ok=True)

    print("🔄 Initializing bug detection model...")
    evaluator = BugDetectionEvaluator(MODEL_PATH)

    processed_images = set()

    print("Deployment started. Monitoring for new images...")

    while True:
        try:
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            image_files = [f for f in Path(IMAGE_FOLDER).glob("*") if f.suffix.lower() in image_extensions]

            for image_path in image_files:
                if str(image_path) in processed_images:
                    continue

                print(f"\n  Processing new image: {image_path.name}")
                result, original_image = evaluator.detect_bugs_in_image(str(image_path), conf_threshold=CONF_THRESHOLD)
                if result is None:
                    print(" Detection failed for this image.")
                    processed_images.add(str(image_path))
                    continue

                if result['detections']['count'] > 0:
                    vis_image = evaluator.visualize_detections(original_image.copy(), result, show_all_detections=False, conf_threshold=CONF_THRESHOLD)
                    vis_path = os.path.join(OUTPUT_FOLDER, "high_confidence", f"detected_{image_path.name}")
                    vis_image.save(vis_path)

                    vis_all = evaluator.visualize_detections(original_image.copy(), result, show_all_detections=True, conf_threshold=0.1)
                    vis_all_path = os.path.join(OUTPUT_FOLDER, "all_detections", f"all_{image_path.name}")
                    vis_all.save(vis_all_path)
                else:
                    vis_path = os.path.join(OUTPUT_FOLDER, "high_confidence", f"no_bugs_{image_path.name}")
                    original_image.save(vis_path)

                processed_images.add(str(image_path))
                print(f" Saved visualizations for {image_path.name}")

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n Deployment stopped by user.")
            break
        except Exception as e:
            print(f" Error in deployment loop: {e}")
            time.sleep(POLL_INTERVAL)
