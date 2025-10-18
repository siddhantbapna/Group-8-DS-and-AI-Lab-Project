#!/usr/bin/env python3
"""
Final Working EDA for BraTS2023 Dataset - Handles correct file naming
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib
from scipy import stats
from scipy.ndimage import label, center_of_mass
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class FinalBraTS2023EDA:
    """Final Working EDA for BraTS2023 dataset"""
    
    def __init__(self, data_path="data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"):
        self.data_path = Path(data_path)
        self.patients = []
        self.modalities = ['t1n', 't1c', 't2w', 't2f']
        self.results = {}
        
        print("BraTS2023 Final EDA Analysis")
        print("="*60)
        
    def load_dataset_info(self):
        """Load basic dataset information"""
        print("\n1. DATASET STRUCTURE ANALYSIS")
        print("-" * 40)
        
        # Find all patient directories
        patient_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        self.patients = [d.name for d in patient_dirs]
        
        print(f"Total patients: {len(self.patients)}")
        print(f"Data path: {self.data_path}")
        
        # Analyze file structure for first few patients
        file_types = {}
        total_files = 0
        sample_patients = self.patients[:5]
        
        for patient in sample_patients:
            patient_path = self.data_path / patient
            files = list(patient_path.glob("*.nii.gz"))
            
            for file in files:
                # Handle hyphen naming convention
                file_type = file.name.split('-')[-1].replace('.nii.gz', '')
                file_types[file_type] = file_types.get(file_type, 0) + 1
                total_files += 1
        
        print(f"\nFile types found (sample of {len(sample_patients)} patients):")
        for file_type, count in file_types.items():
            print(f"  {file_type}: {count} files")
        
        # Check for missing files
        missing_files = 0
        for patient in sample_patients:
            patient_path = self.data_path / patient
            expected_files = [f"{patient}-{mod}.nii.gz" for mod in self.modalities] + [f"{patient}-seg.nii.gz"]
            
            for expected_file in expected_files:
                if not (patient_path / expected_file).exists():
                    missing_files += 1
        
        print(f"\nMissing files in sample: {missing_files}")
        
        self.results['dataset_info'] = {
            'total_patients': len(self.patients),
            'file_types': file_types,
            'missing_files': missing_files
        }
        
        return self.patients
    
    def analyze_image_properties(self, sample_size=5):
        """Analyze image properties across modalities"""
        print(f"\n2. IMAGE PROPERTIES ANALYSIS (Sample: {sample_size} patients)")
        print("-" * 40)
        
        image_props = {
            'shapes': [],
            'spacings': [],
            'intensities': {mod: [] for mod in self.modalities},
        }
        
        # Sample patients for analysis
        sample_patients = self.patients[:sample_size]
        
        for i, patient in enumerate(sample_patients):
            print(f"Processing patient {i+1}/{len(sample_patients)}: {patient}")
            
            patient_path = self.data_path / patient
            
            for modality in self.modalities:
                file_path = patient_path / f"{patient}-{modality}.nii.gz"
                
                if file_path.exists():
                    try:
                        # Load image
                        img = nib.load(file_path)
                        data = img.get_fdata()
                        
                        # Store properties
                        image_props['shapes'].append(data.shape)
                        image_props['spacings'].append(img.header.get_zooms())
                        
                        # Sample intensities (to avoid memory issues)
                        sample_data = data[::4, ::4, ::4]  # Sample every 4th voxel
                        image_props['intensities'][modality].extend(sample_data.flatten())
                        
                        print(f"  {modality}: shape={data.shape}, spacing={img.header.get_zooms()}")
                        
                    except Exception as e:
                        print(f"  Error loading {file_path}: {e}")
        
        # Analyze shapes
        if image_props['shapes']:
            shapes_array = np.array(image_props['shapes'])
            print(f"\nImage shapes:")
            print(f"  Mean: {shapes_array.mean(axis=0).astype(int)}")
            print(f"  Min:  {shapes_array.min(axis=0)}")
            print(f"  Max:  {shapes_array.max(axis=0)}")
            print(f"  Std:  {shapes_array.std(axis=0).astype(int)}")
        
        # Analyze spacings
        if image_props['spacings']:
            spacings_array = np.array(image_props['spacings'])
            print(f"\nVoxel spacings:")
            print(f"  Mean: {spacings_array.mean(axis=0)}")
            print(f"  Min:  {spacings_array.min(axis=0)}")
            print(f"  Max:  {spacings_array.max(axis=0)}")
        
        self.results['image_properties'] = image_props
        
        return image_props
    
    def analyze_intensity_distributions(self, sample_size=5):
        """Analyze intensity distributions across modalities"""
        print(f"\n3. INTENSITY DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        intensity_stats = {}
        
        for modality in self.modalities:
            intensities = []
            
            # Sample patients
            sample_patients = self.patients[:sample_size]
            
            for patient in sample_patients:
                file_path = self.data_path / patient / f"{patient}-{modality}.nii.gz"
                
                if file_path.exists():
                    try:
                        img = nib.load(file_path)
                        data = img.get_fdata()
                        
                        # Sample data to avoid memory issues
                        sample_data = data[::4, ::4, ::4]  # Sample every 4th voxel
                        
                        # Remove background (intensity = 0)
                        brain_data = sample_data[sample_data > 0]
                        intensities.extend(brain_data.flatten())
                        
                    except Exception as e:
                        continue
            
            if intensities:
                intensities = np.array(intensities)
                intensity_stats[modality] = {
                    'mean': np.mean(intensities),
                    'std': np.std(intensities),
                    'min': np.min(intensities),
                    'max': np.max(intensities),
                    'median': np.median(intensities),
                    'q25': np.percentile(intensities, 25),
                    'q75': np.percentile(intensities, 75),
                    'data': intensities
                }
                
                print(f"\n{modality.upper()} intensity statistics:")
                print(f"  Mean: {intensity_stats[modality]['mean']:.2f}")
                print(f"  Std:  {intensity_stats[modality]['std']:.2f}")
                print(f"  Range: {intensity_stats[modality]['min']:.2f} - {intensity_stats[modality]['max']:.2f}")
                print(f"  Median: {intensity_stats[modality]['median']:.2f}")
        
        self.results['intensity_stats'] = intensity_stats
        return intensity_stats
    
    def analyze_tumor_statistics(self, sample_size=5):
        """Analyze tumor statistics and segmentation properties"""
        print(f"\n4. TUMOR STATISTICS ANALYSIS")
        print("-" * 40)
        
        tumor_stats = {
            'tumor_presence': [],
            'tumor_sizes': [],
            'tumor_volumes': [],
            'class_distributions': [],
        }
        
        sample_patients = self.patients[:sample_size]
        
        for i, patient in enumerate(sample_patients):
            print(f"Processing patient {i+1}/{len(sample_patients)}: {patient}")
            
            seg_path = self.data_path / patient / f"{patient}-seg.nii.gz"
            
            if seg_path.exists():
                try:
                    seg_img = nib.load(seg_path)
                    seg_data = seg_img.get_fdata()
                    
                    # Analyze tumor presence
                    has_tumor = np.any(seg_data > 0)
                    tumor_stats['tumor_presence'].append(has_tumor)
                    
                    if has_tumor:
                        # Calculate tumor sizes
                        unique_labels, counts = np.unique(seg_data, return_counts=True)
                        
                        # Class distribution
                        class_dist = {int(label): count for label, count in zip(unique_labels, counts)}
                        tumor_stats['class_distributions'].append(class_dist)
                        
                        # Tumor volumes (excluding background)
                        tumor_volumes = {int(label): count for label, count in zip(unique_labels, counts) if label > 0}
                        tumor_stats['tumor_volumes'].append(tumor_volumes)
                        
                        print(f"  Has tumor: {has_tumor}")
                        print(f"  Classes found: {list(unique_labels)}")
                        print(f"  Class counts: {dict(zip(unique_labels, counts))}")
                
                except Exception as e:
                    print(f"  Error processing {seg_path}: {e}")
        
        # Calculate summary statistics
        if tumor_stats['tumor_presence']:
            tumor_presence_rate = np.mean(tumor_stats['tumor_presence'])
            print(f"\nTumor presence rate: {tumor_presence_rate:.2%}")
        
        # Analyze class distributions
        all_class_counts = {}
        for class_dist in tumor_stats['class_distributions']:
            for class_id, count in class_dist.items():
                all_class_counts[class_id] = all_class_counts.get(class_id, 0) + count
        
        if all_class_counts:
            print(f"\nClass distribution (total pixels):")
            total_pixels = sum(all_class_counts.values())
            for class_id in sorted(all_class_counts.keys()):
                count = all_class_counts[class_id]
                percentage = count / total_pixels * 100
                class_name = {0: 'Background', 1: 'NCR/NET', 2: 'ED', 3: 'ET'}.get(class_id, f'Class_{class_id}')
                print(f"  {class_name}: {count:,} pixels ({percentage:.2f}%)")
        
        self.results['tumor_stats'] = tumor_stats
        return tumor_stats
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print(f"\n5. CREATING VISUALIZATIONS")
        print("-" * 40)
        
        # Create output directory
        output_dir = Path("eda_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Intensity distribution plots
        self._plot_intensity_distributions(output_dir)
        
        # 2. Image properties plots
        self._plot_image_properties(output_dir)
        
        # 3. Tumor statistics plots
        self._plot_tumor_statistics(output_dir)
        
        # 4. Sample image visualizations
        self._plot_sample_images(output_dir)
        
        print(f"\nAll visualizations saved to: {output_dir}")
    
    def _plot_intensity_distributions(self, output_dir):
        """Plot intensity distributions across modalities"""
        if 'intensity_stats' not in self.results or not self.results['intensity_stats']:
            print("No intensity statistics available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (modality, stats) in enumerate(self.results['intensity_stats'].items()):
            if i < 4:
                data = stats['data']
                
                # Plot histogram
                axes[i].hist(data, bins=100, alpha=0.7, density=True)
                axes[i].axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.1f}")
                axes[i].axvline(stats['median'], color='green', linestyle='--', label=f"Median: {stats['median']:.1f}")
                
                axes[i].set_title(f'{modality.upper()} Intensity Distribution')
                axes[i].set_xlabel('Intensity')
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'intensity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Box plot comparison
        if len(self.results['intensity_stats']) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            data_for_box = []
            labels = []
            
            for modality, stats in self.results['intensity_stats'].items():
                data_for_box.append(stats['data'])
                labels.append(modality.upper())
            
            if data_for_box and labels:
                ax.boxplot(data_for_box, labels=labels)
                ax.set_title('Intensity Distribution Comparison Across Modalities')
                ax.set_ylabel('Intensity')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'intensity_boxplot.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_image_properties(self, output_dir):
        """Plot image properties analysis"""
        if 'image_properties' not in self.results or not self.results['image_properties']['shapes']:
            print("No image properties available for plotting")
            return
        
        props = self.results['image_properties']
        
        # Shape analysis
        shapes_array = np.array(props['shapes'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot shape distributions
        for i, dim in enumerate(['Height', 'Width', 'Depth']):
            if i < 3:  # Only plot first 3 dimensions
                row, col = i // 2, i % 2
                axes[row, col].hist(shapes_array[:, i], bins=20, alpha=0.7)
                axes[row, col].set_title(f'{dim} Distribution')
                axes[row, col].set_xlabel(f'{dim} (voxels)')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
        
        # Volume analysis
        volumes = np.prod(shapes_array, axis=1)
        axes[1, 0].hist(volumes, bins=20, alpha=0.7)
        axes[1, 0].set_title('Volume Distribution')
        axes[1, 0].set_xlabel('Volume (voxels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spacing analysis
        if props['spacings']:
            spacings_array = np.array(props['spacings'])
            axes[1, 1].hist(spacings_array[:, 0], bins=20, alpha=0.7, label='X')
            axes[1, 1].hist(spacings_array[:, 1], bins=20, alpha=0.7, label='Y')
            axes[1, 1].hist(spacings_array[:, 2], bins=20, alpha=0.7, label='Z')
            axes[1, 1].set_title('Voxel Spacing Distribution')
            axes[1, 1].set_xlabel('Spacing (mm)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'image_properties.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tumor_statistics(self, output_dir):
        """Plot tumor statistics"""
        if 'tumor_stats' not in self.results or not self.results['tumor_stats']['tumor_presence']:
            print("No tumor statistics available for plotting")
            return
        
        tumor_stats = self.results['tumor_stats']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Tumor presence
        if tumor_stats['tumor_presence']:
            presence_counts = np.bincount(tumor_stats['tumor_presence'])
            axes[0, 0].pie(presence_counts, labels=['No Tumor', 'Has Tumor'], autopct='%1.1f%%')
            axes[0, 0].set_title('Tumor Presence Distribution')
        
        # Class distribution
        all_class_counts = {}
        for class_dist in tumor_stats['class_distributions']:
            for class_id, count in class_dist.items():
                all_class_counts[class_id] = all_class_counts.get(class_id, 0) + count
        
        if all_class_counts:
            class_names = {0: 'Background', 1: 'NCR/NET', 2: 'ED', 3: 'ET'}
            labels = [class_names.get(class_id, f'Class_{class_id}') for class_id in sorted(all_class_counts.keys())]
            counts = [all_class_counts[class_id] for class_id in sorted(all_class_counts.keys())]
            
            axes[0, 1].pie(counts, labels=labels, autopct='%1.1f%%')
            axes[0, 1].set_title('Class Distribution')
        
        # Tumor volume distribution
        tumor_volumes = []
        for vol_dict in tumor_stats['tumor_volumes']:
            for class_id, volume in vol_dict.items():
                tumor_volumes.append(volume)
        
        if tumor_volumes:
            axes[1, 0].hist(tumor_volumes, bins=30, alpha=0.7)
            axes[1, 0].set_title('Tumor Volume Distribution')
            axes[1, 0].set_xlabel('Volume (voxels)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tumor_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_images(self, output_dir):
        """Plot sample images from different patients"""
        print("Creating sample image visualizations...")
        
        # Select a few patients for visualization
        sample_patients = self.patients[:2]  # Reduced for faster execution
        
        for patient in sample_patients:
            patient_path = self.data_path / patient
            
            # Load all modalities and segmentation
            modalities_data = {}
            seg_data = None
            
            for modality in self.modalities:
                file_path = patient_path / f"{patient}-{modality}.nii.gz"
                if file_path.exists():
                    try:
                        img = nib.load(file_path)
                        modalities_data[modality] = img.get_fdata()
                    except:
                        continue
            
            seg_path = patient_path / f"{patient}-seg.nii.gz"
            if seg_path.exists():
                try:
                    seg_img = nib.load(seg_path)
                    seg_data = seg_img.get_fdata()
                except:
                    continue
            
            if modalities_data and seg_data is not None:
                # Create visualization
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Get middle slice
                middle_slice = seg_data.shape[2] // 2
                
                # Plot each modality
                for i, (modality, data) in enumerate(modalities_data.items()):
                    if i < 4:
                        row, col = i // 2, i % 2
                        if i < 2:
                            axes[row, col].imshow(data[:, :, middle_slice], cmap='gray')
                            axes[row, col].set_title(f'{modality.upper()} - Slice {middle_slice}')
                            axes[row, col].axis('off')
                
                # Plot segmentation
                axes[0, 2].imshow(seg_data[:, :, middle_slice], cmap='tab10')
                axes[0, 2].set_title(f'Segmentation - Slice {middle_slice}')
                axes[0, 2].axis('off')
                
                # Plot overlay
                if len(modalities_data) > 0:
                    first_modality = list(modalities_data.values())[0]
                    axes[1, 2].imshow(first_modality[:, :, middle_slice], cmap='gray', alpha=0.7)
                    axes[1, 2].imshow(seg_data[:, :, middle_slice], cmap='tab10', alpha=0.5)
                    axes[1, 2].set_title(f'Overlay - Slice {middle_slice}')
                    axes[1, 2].axis('off')
                
                plt.suptitle(f'Patient: {patient}', fontsize=16)
                plt.tight_layout()
                plt.savefig(output_dir / f'sample_images_{patient}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\n6. GENERATING SUMMARY REPORT")
        print("-" * 40)
        
        report = []
        report.append("# BraTS2023 Dataset EDA Summary Report")
        report.append("=" * 50)
        
        # Dataset info
        if 'dataset_info' in self.results:
            info = self.results['dataset_info']
            report.append(f"\n## Dataset Information")
            report.append(f"- Total patients: {info['total_patients']}")
            report.append(f"- File types: {info['file_types']}")
            report.append(f"- Missing files: {info['missing_files']}")
        
        # Image properties
        if 'image_properties' in self.results and self.results['image_properties']['shapes']:
            props = self.results['image_properties']
            shapes_array = np.array(props['shapes'])
            report.append(f"\n## Image Properties")
            report.append(f"- Mean shape: {shapes_array.mean(axis=0).astype(int)}")
            report.append(f"- Shape range: {shapes_array.min(axis=0)} - {shapes_array.max(axis=0)}")
        
        # Intensity statistics
        if 'intensity_stats' in self.results:
            report.append(f"\n## Intensity Statistics")
            for modality, stats in self.results['intensity_stats'].items():
                report.append(f"- {modality.upper()}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
        
        # Tumor statistics
        if 'tumor_stats' in self.results:
            tumor_stats = self.results['tumor_stats']
            if tumor_stats['tumor_presence']:
                presence_rate = np.mean(tumor_stats['tumor_presence'])
                report.append(f"\n## Tumor Statistics")
                report.append(f"- Tumor presence rate: {presence_rate:.2%}")
                
                # Class distribution
                all_class_counts = {}
                for class_dist in tumor_stats['class_distributions']:
                    for class_id, count in class_dist.items():
                        all_class_counts[class_id] = all_class_counts.get(class_id, 0) + count
                
                if all_class_counts:
                    report.append(f"- Class distribution:")
                    for class_id in sorted(all_class_counts.keys()):
                        count = all_class_counts[class_id]
                        class_name = {0: 'Background', 1: 'NCR/NET', 2: 'ED', 3: 'ET'}.get(class_id, f'Class_{class_id}')
                        report.append(f"  - {class_name}: {count:,} pixels")
        
        # Save report
        with open("eda_outputs/eda_summary_report.txt", "w") as f:
            f.write("\n".join(report))
        
        print("Summary report saved to: eda_outputs/eda_summary_report.txt")
        
        # Print summary to console
        print("\n" + "="*60)
        print("EDA SUMMARY")
        print("="*60)
        for line in report:
            print(line)
    
    def run_complete_eda(self, sample_size=5):
        """Run complete EDA analysis"""
        print("Starting comprehensive EDA analysis...")
        
        # Run all analyses
        self.load_dataset_info()
        self.analyze_image_properties(sample_size)
        self.analyze_intensity_distributions(sample_size)
        self.analyze_tumor_statistics(sample_size)
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate summary report
        self.generate_summary_report()
        
        print(f"\nEDA Analysis Complete!")
        print(f"Results saved to: eda_outputs/")

def main():
    """Main function to run EDA"""
    # Initialize EDA
    eda = FinalBraTS2023EDA()
    
    # Run complete analysis
    eda.run_complete_eda(sample_size=5)
    
    print(f"\nEDA Analysis Complete!")
    print(f"Check the 'eda_outputs' folder for all results and visualizations.")

if __name__ == "__main__":
    main()
