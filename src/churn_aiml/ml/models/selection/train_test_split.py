import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional, List
import warnings
import seaborn as sns


class StratifiedChurnSampler:
    """
    A comprehensive class for performing safe stratified sampling on churn data with
    extensive visualization and analysis capabilities.

    This class handles the complete stratification process including:
    - Creating time-based stratification bins
    - Handling classes with insufficient samples
    - Performing safe train-test splits
    - Comprehensive visualizations and verification
    - Statistical analysis and reporting
    """

    def __init__(self, min_samples_per_class: int = 2,
                 bin_days: List[int] = None,
                 random_state: int = 42):
        """
        Initialize the StratifiedChurnSampler.

        Parameters:
        -----------
        min_samples_per_class : int, default=2
            Minimum samples required per class for stratification
        bin_days : List[int], optional
            Custom bin boundaries in days. Default is [365, 730, 1095, 1460]
        random_state : int, default=42
            Random state for reproducibility
        """
        self.min_samples_per_class = min_samples_per_class
        self.bin_days = bin_days or [365, 730, 1095, 1460]
        self.random_state = random_state

        # Results storage
        self.stratified_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Statistics storage
        self.distribution_stats = {}
        self.bin_ranges = []
        self.stratification_success = False

        # Store original dataframe for analysis
        self.original_df = None

    def fit_transform(self, cust_churn_df: pd.DataFrame,
                      test_size: float = 0.2,
                      keep_all_data: bool = True,
                      remove_last_bin: bool = True,
                      plot_distributions: bool = True,
                      plot_dashboard: bool = True,
                      plot_verification: bool = True) -> Dict:
        """
        Main method to perform stratified sampling on the churn data.

        Parameters:
        -----------
        cust_churn_df : pd.DataFrame
            DataFrame with at least 'CUST_ACCOUNT_NUMBER' and 'DAYS_TO_CHURN' columns
        test_size : float, default=0.2
            Proportion of dataset to include in the test split
        keep_all_data : bool, default=True
            If True, combine small classes; if False, remove them
        remove_last_bin : bool, default=True
            If True, remove the last bin (>4 years) as in original code
        plot_distributions : bool, default=True
            If True, plot basic distribution visualizations
        plot_dashboard : bool, default=True
            If True, plot comprehensive analysis dashboard
        plot_verification : bool, default=True
            If True, plot verification dashboard

        Returns:
        --------
        Dict with results including train/test splits and statistics
        """
        print("=== StratifiedChurnSampler: Starting Stratification Process ===\n")

        # Store original dataframe
        self.original_df = cust_churn_df.copy()

        # Step 1: Create stratified bins
        self._create_stratified_bins(cust_churn_df)

        # Step 2: Handle the last bin if requested
        if remove_last_bin:
            self._remove_last_bin()

        # Step 3: Handle small classes
        self._handle_small_classes(keep_all_data)

        # Step 4: Prepare features
        self._prepare_features()

        # Step 5: Perform train-test split
        self._perform_train_test_split(test_size)

        # Step 6: Generate visualizations
        if plot_distributions:
            self.plot_distributions()
        if plot_dashboard:
            self.plot_analysis_dashboard()
        if plot_verification:
            self.plot_verification_dashboard()

        # Return results
        return self.get_results()

    def _create_stratified_bins(self, cust_churn_df: pd.DataFrame):
        """Create stratified bins based on DAYS_TO_CHURN."""
        print("Step 1: Creating stratified bins based on DAYS_TO_CHURN...")

        self.stratified_df = pd.DataFrame()
        self.bin_ranges = []

        label_count = 1
        prev_boundary = 0

        for boundary in self.bin_days:
            mask = ((prev_boundary < cust_churn_df["DAYS_TO_CHURN"]) &
                   (cust_churn_df["DAYS_TO_CHURN"] <= boundary))
            XX = cust_churn_df[mask].copy()
            XX["SAMPLE"] = label_count
            self.stratified_df = pd.concat([self.stratified_df, XX], axis=0, ignore_index=True)

            years_start = prev_boundary / 365
            years_end = boundary / 365
            bin_range = (f"Bin {label_count}: {prev_boundary} < DAYS_TO_CHURN <= {boundary} days "
                        f"({years_start:.0f}-{years_end:.0f} years)")
            self.bin_ranges.append(bin_range)
            print(f"  - {bin_range}: {len(XX)} samples")

            prev_boundary = boundary
            label_count += 1

        # Add final bin for values beyond the last boundary
        XX = cust_churn_df[cust_churn_df["DAYS_TO_CHURN"] > self.bin_days[-1]].copy()
        XX["SAMPLE"] = label_count
        self.stratified_df = pd.concat([self.stratified_df, XX], axis=0, ignore_index=True)

        bin_range = f"Bin {label_count}: DAYS_TO_CHURN > {self.bin_days[-1]} days (>{self.bin_days[-1]/365:.0f} years)"
        self.bin_ranges.append(bin_range)
        print(f"  - {bin_range}: {len(XX)} samples")

        # Store original distribution
        self.distribution_stats['original_distribution'] = (
            self.stratified_df["SAMPLE"].value_counts().sort_index().to_dict()
        )

    def _remove_last_bin(self):
        """Remove the last bin (highest SAMPLE value)."""
        max_sample = self.stratified_df["SAMPLE"].max()
        print(f"\nStep 2: Removing SAMPLE=={max_sample} (last bin)...")

        self.stratified_df = self.stratified_df[self.stratified_df['SAMPLE'] != max_sample]
        self.bin_ranges = self.bin_ranges[:-1]  # Remove last bin description

        print("Distribution after removing last bin:")
        current_dist = self.stratified_df["SAMPLE"].value_counts().sort_index()
        print(current_dist)

    def _handle_small_classes(self, keep_all_data: bool):
        """Handle classes with insufficient samples."""
        print(f"\nStep 3: Handling classes with less than {self.min_samples_per_class} samples...")

        current_dist = self.stratified_df["SAMPLE"].value_counts().sort_index()
        small_classes = current_dist[current_dist < self.min_samples_per_class]

        if len(small_classes) > 0:
            print(f"Found {len(small_classes)} classes with insufficient samples: {list(small_classes.index)}")

            if keep_all_data:
                print("Combining small classes with adjacent ones...")
                self._combine_small_classes()
            else:
                print("Removing classes with insufficient samples...")
                classes_to_keep = current_dist[current_dist >= self.min_samples_per_class].index
                self.stratified_df = self.stratified_df[
                    self.stratified_df['SAMPLE'].isin(classes_to_keep)
                ]
        else:
            print("All classes have sufficient samples!")

        # Store final distribution before splitting
        self.distribution_stats['final_distribution'] = (
            self.stratified_df["SAMPLE"].value_counts().sort_index().to_dict()
        )

    def _combine_small_classes(self):
        """Combine classes with insufficient samples with adjacent classes."""
        sample_counts = self.stratified_df["SAMPLE"].value_counts().sort_index()
        changes_made = []

        for sample_class in sorted(sample_counts.index):
            # Recount after each combination
            current_counts = self.stratified_df["SAMPLE"].value_counts().sort_index()

            if sample_class in current_counts and current_counts[sample_class] < self.min_samples_per_class:
                available_classes = sorted(current_counts.index)
                current_idx = available_classes.index(sample_class)

                # Try combining with previous class first
                if current_idx > 0:
                    target_class = available_classes[current_idx - 1]
                    self.stratified_df.loc[
                        self.stratified_df["SAMPLE"] == sample_class, "SAMPLE"
                    ] = target_class
                    changes_made.append(f"Combined class {sample_class} into class {target_class}")
                # Otherwise combine with next class
                elif current_idx < len(available_classes) - 1:
                    target_class = available_classes[current_idx + 1]
                    self.stratified_df.loc[
                        self.stratified_df["SAMPLE"] == sample_class, "SAMPLE"
                    ] = target_class
                    changes_made.append(f"Combined class {sample_class} into class {target_class}")

        if changes_made:
            print("Class combinations made:")
            for change in changes_made:
                print(f"  - {change}")

    def _prepare_features(self):
        """Prepare features and target variable."""
        print("\nStep 4: Preparing features...")

        # Add NO_OF_RENEWALS column
        self.stratified_df["NO_OF_RENEWALS"] = self.stratified_df["SAMPLE"] - 1

        # Store features and target
        self.features = self.stratified_df.drop(["DAYS_TO_CHURN", "SAMPLE"], axis=1)
        self.y = self.stratified_df["SAMPLE"]

        print(f"Total samples: {len(self.features)}")
        print(f"Number of features: {self.features.shape[1]}")

    def _perform_train_test_split(self, test_size: float):
        """Perform safe train-test split with stratification if possible."""
        print(f"\nStep 5: Performing train-test split (test_size={test_size})...")

        final_dist = self.y.value_counts().sort_index()

        if final_dist.min() < self.min_samples_per_class:
            warnings.warn("Classes with insufficient samples found. Using regular split.")
            self._regular_split(test_size)
            self.stratification_success = False
        else:
            # Check if test_size allows for proper stratification
            min_test_samples = final_dist * test_size

            if min_test_samples.min() < 1:
                self._handle_small_test_size(test_size, final_dist)
            else:
                self._stratified_split(test_size)
                self.stratification_success = True

        # Store distribution statistics
        self._store_split_statistics()

    def _regular_split(self, test_size: float):
        """Perform regular train-test split without stratification."""
        print("Using regular split without stratification...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.y, test_size=test_size,
            random_state=self.random_state, shuffle=True
        )

    def _stratified_split(self, test_size: float):
        """Perform stratified train-test split."""
        print("Using stratified split...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.y, test_size=test_size,
            random_state=self.random_state, shuffle=True, stratify=self.y
        )

    def _handle_small_test_size(self, test_size: float, final_dist: pd.Series):
        """Handle case where test_size is too small for proper stratification."""
        new_test_size = 1.0 / final_dist.min()

        if new_test_size > 0.5:
            warnings.warn(f"Classes too small for {test_size:.0%} test split. Using regular split.")
            self._regular_split(test_size)
            self.stratification_success = False
        else:
            adjusted_test_size = max(new_test_size, test_size)
            print(f"Adjusted test_size to {adjusted_test_size:.2f} for proper stratification")
            self._stratified_split(adjusted_test_size)
            self.stratification_success = True

    def _store_split_statistics(self):
        """Store train/test distribution statistics."""
        train_dist = self.y_train.value_counts().sort_index()
        test_dist = self.y_test.value_counts().sort_index()

        self.distribution_stats.update({
            'train_distribution': train_dist.to_dict(),
            'test_distribution': test_dist.to_dict(),
            'stratified_split': self.stratification_success,
            'total_samples': len(self.stratified_df),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'bin_ranges': self.bin_ranges
        })

        # Print summary
        print(f"\nSplit results:")
        print(f"  - Train set: {len(self.X_train)} samples ({len(self.X_train)/len(self.features)*100:.1f}%)")
        print(f"  - Test set: {len(self.X_test)} samples ({len(self.X_test)/len(self.features)*100:.1f}%)")
        print(f"  - Stratified: {self.stratification_success}")

    def plot_distributions(self):
        """Plot basic distribution visualizations."""
        # Get distributions
        original_dist = pd.Series(self.distribution_stats['original_distribution']).sort_index()
        final_dist = pd.Series(self.distribution_stats['final_distribution']).sort_index()
        train_dist = pd.Series(self.distribution_stats['train_distribution']).sort_index()
        test_dist = pd.Series(self.distribution_stats['test_distribution']).sort_index()

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stratified Churn Sampling Distribution Analysis', fontsize=16)

        # Plot 1: Original distribution
        self._plot_bar_with_values(axes[0, 0], original_dist,
                                  'Original Distribution', 'skyblue')

        # Plot 2: Final distribution
        self._plot_bar_with_values(axes[0, 1], final_dist,
                                  'Final Distribution (After Processing)', 'lightgreen')

        # Plot 3: Train vs Test distribution
        self._plot_train_test_comparison(axes[1, 0], train_dist, test_dist)

        # Plot 4: Percentage distribution
        self._plot_percentage_distribution(axes[1, 1], train_dist, test_dist)

        plt.tight_layout()
        plt.show()

        # Additional plot: Bin ranges summary
        self._plot_bin_ranges()

    def plot_analysis_dashboard(self):
        """Plot comprehensive analysis dashboard with 9 visualizations."""
        fig = plt.figure(figsize=(20, 12))

        # 1. Days to Churn Distribution by Sample Class
        ax1 = plt.subplot(3, 3, 1)
        for sample_class in sorted(self.stratified_df['SAMPLE'].unique()):
            class_data = self.stratified_df[self.stratified_df['SAMPLE'] == sample_class]['DAYS_TO_CHURN']
            ax1.hist(class_data, alpha=0.6, label=f'Class {sample_class}', bins=30)
        ax1.set_xlabel('Days to Churn')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Days to Churn Distribution by Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box Plot of Days to Churn by Class
        ax2 = plt.subplot(3, 3, 2)
        data_for_box = [self.stratified_df[self.stratified_df['SAMPLE'] == i]['DAYS_TO_CHURN'].values
                        for i in sorted(self.stratified_df['SAMPLE'].unique())]
        box_plot = ax2.boxplot(data_for_box, labels=sorted(self.stratified_df['SAMPLE'].unique()))
        ax2.set_xlabel('Sample Class')
        ax2.set_ylabel('Days to Churn')
        ax2.set_title('Days to Churn Box Plot by Class')
        ax2.grid(True, alpha=0.3)

        # 3. NO_OF_RENEWALS Distribution
        ax3 = plt.subplot(3, 3, 3)
        renewals_dist = self.stratified_df['NO_OF_RENEWALS'].value_counts().sort_index()
        renewals_dist.plot(kind='bar', ax=ax3, color='darkgreen', edgecolor='black')
        ax3.set_xlabel('Number of Renewals')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of NO_OF_RENEWALS')
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(renewals_dist.values):
            ax3.text(i, v + 5, str(v), ha='center', va='bottom')

        # 4. Train/Test Split Balance Check
        ax4 = plt.subplot(3, 3, 4)
        train_props = self.y_train.value_counts(normalize=True).sort_index()
        test_props = self.y_test.value_counts(normalize=True).sort_index()

        x = np.arange(len(train_props))
        width = 0.35

        bars1 = ax4.bar(x - width/2, train_props.values, width, label='Train', alpha=0.8)
        bars2 = ax4.bar(x + width/2, test_props.values, width, label='Test', alpha=0.8)

        ax4.set_xlabel('Sample Class')
        ax4.set_ylabel('Proportion')
        ax4.set_title('Class Balance: Train vs Test (Proportions)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(train_props.index)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Cumulative Distribution
        ax5 = plt.subplot(3, 3, 5)
        days_sorted = np.sort(self.stratified_df['DAYS_TO_CHURN'])
        cumulative = np.arange(1, len(days_sorted) + 1) / len(days_sorted)
        ax5.plot(days_sorted, cumulative, linewidth=2)
        ax5.set_xlabel('Days to Churn')
        ax5.set_ylabel('Cumulative Probability')
        ax5.set_title('Cumulative Distribution of Days to Churn')
        ax5.grid(True, alpha=0.3)

        # Add vertical lines for bin boundaries
        for boundary in self.bin_days:
            ax5.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
            ax5.text(boundary, 0.95, f'{boundary}d', rotation=90, va='top')

        # 6. Sample Size vs Class
        ax6 = plt.subplot(3, 3, 6)
        sample_sizes = self.stratified_df['SAMPLE'].value_counts().sort_index()
        ax6.plot(sample_sizes.index, sample_sizes.values, 'o-', markersize=10, linewidth=2)
        ax6.set_xlabel('Sample Class')
        ax6.set_ylabel('Number of Samples')
        ax6.set_title('Sample Size by Class')
        ax6.grid(True, alpha=0.3)

        # Add annotations
        for i, (x, y) in enumerate(zip(sample_sizes.index, sample_sizes.values)):
            ax6.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

        # 7. Train Set Class Distribution (Pie Chart)
        ax7 = plt.subplot(3, 3, 7)
        train_counts = self.y_train.value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(train_counts)))
        wedges, texts, autotexts = ax7.pie(train_counts.values, labels=train_counts.index,
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        ax7.set_title('Train Set Class Distribution')

        # 8. Test Set Class Distribution (Pie Chart)
        ax8 = plt.subplot(3, 3, 8)
        test_counts = self.y_test.value_counts().sort_index()
        wedges, texts, autotexts = ax8.pie(test_counts.values, labels=test_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax8.set_title('Test Set Class Distribution')

        # 9. Feature Correlation Heatmap
        ax9 = plt.subplot(3, 3, 9)
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = self.X_train[numerical_cols].corr()
            im = ax9.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax9.set_xticks(range(len(numerical_cols)))
            ax9.set_yticks(range(len(numerical_cols)))
            ax9.set_xticklabels(numerical_cols, rotation=45, ha='right')
            ax9.set_yticklabels(numerical_cols)
            ax9.set_title('Feature Correlation Matrix')
            plt.colorbar(im, ax=ax9)
        else:
            ax9.text(0.5, 0.5, 'Correlation matrix\nnot available',
                     ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Feature Correlation Matrix')

        plt.tight_layout()
        plt.show()

    def plot_verification_dashboard(self):
        """Plot verification dashboard to confirm correct implementation."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Verification: Implementation Correctness', fontsize=16)

        # Plot 1: Verify bin assignments
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.stratified_df['DAYS_TO_CHURN'], self.stratified_df['SAMPLE'],
                             alpha=0.5, c=self.stratified_df['SAMPLE'], cmap='viridis')
        ax1.set_xlabel('Days to Churn')
        ax1.set_ylabel('Sample Class')
        ax1.set_title('Bin Assignment Verification')
        ax1.grid(True, alpha=0.3)

        # Add bin boundary lines
        for boundary in self.bin_days:
            ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
            ax1.text(boundary, max(self.stratified_df['SAMPLE']) + 0.1, f'{boundary}',
                    rotation=90, va='top', ha='right')

        # Plot 2: NO_OF_RENEWALS verification
        ax2 = axes[0, 1]
        renewal_check = self.stratified_df[['SAMPLE', 'NO_OF_RENEWALS']].drop_duplicates().sort_values('SAMPLE')
        ax2.plot(renewal_check['SAMPLE'], renewal_check['NO_OF_RENEWALS'], 'o-', markersize=10)
        ax2.set_xlabel('SAMPLE')
        ax2.set_ylabel('NO_OF_RENEWALS')
        ax2.set_title('NO_OF_RENEWALS = SAMPLE - 1 Verification')
        ax2.grid(True, alpha=0.3)

        # Add expected line
        x_line = np.array([renewal_check['SAMPLE'].min(), renewal_check['SAMPLE'].max()])
        ax2.plot(x_line, x_line - 1, 'r--', label='Expected: y = x - 1')
        ax2.legend()

        # Plot 3: Train/Test ratio verification
        ax3 = axes[1, 0]
        sizes = [len(self.X_train), len(self.X_test)]
        labels = [f'Train\n({len(self.X_train)})', f'Test\n({len(self.X_test)})']
        colors = ['lightblue', 'lightcoral']
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        expected_test_pct = self.distribution_stats['test_samples'] / self.distribution_stats['total_samples'] * 100
        ax3.set_title(f'Train/Test Split Verification (Expected Test: ~{expected_test_pct:.0f}%)')

        # Plot 4: Process verification summary
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.9, "Process Verification Summary:", fontsize=14, fontweight='bold')

        # Check various conditions
        checks = [
            f"✓ Number of bins created: {len(self.bin_ranges)}",
            f"✓ Classes in final data: {sorted(self.stratified_df['SAMPLE'].unique())}",
            f"✓ NO_OF_RENEWALS added: {'NO_OF_RENEWALS' in self.X_train.columns}",
            f"✓ Train samples: {len(self.X_train)}",
            f"✓ Test samples: {len(self.X_test)}",
            f"✓ Stratification successful: {self.stratification_success}",
            f"✓ Random state: {self.random_state}",
            f"✓ Min samples per class: {self.min_samples_per_class}"
        ]

        y_pos = 0.75
        for check in checks:
            ax4.text(0.1, y_pos, check, fontsize=11)
            y_pos -= 0.08

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_statistical_summary(self):
        """Plot statistical summary of days to churn by class."""
        print("\n=== Statistical Summary ===")
        print("\nDays to Churn Statistics by Class:")
        print("-" * 80)
        print(f"{'Class':>10} | {'Mean':>10} | {'Median':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10} | {'Count':>10}")
        print("-" * 80)

        stats_data = []
        for class_label in sorted(self.stratified_df['SAMPLE'].unique()):
            class_data = self.stratified_df[self.stratified_df['SAMPLE'] == class_label]['DAYS_TO_CHURN']
            stats_data.append({
                'Class': class_label,
                'Mean': class_data.mean(),
                'Median': class_data.median(),
                'Std': class_data.std(),
                'Min': class_data.min(),
                'Max': class_data.max(),
                'Count': len(class_data)
            })
            print(f"{class_label:>10} | {class_data.mean():>10.1f} | {class_data.median():>10.1f} | "
                  f"{class_data.std():>10.1f} | {class_data.min():>10.1f} | {class_data.max():>10.1f} | "
                  f"{len(class_data):>10}")

        # Create visual summary
        stats_df = pd.DataFrame(stats_data)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Statistical Summary by Class', fontsize=16)

        # Plot 1: Mean and Median
        ax1 = axes[0, 0]
        x = stats_df['Class']
        ax1.plot(x, stats_df['Mean'], 'o-', label='Mean', markersize=10)
        ax1.plot(x, stats_df['Median'], 's-', label='Median', markersize=10)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Days')
        ax1.set_title('Mean and Median Days to Churn by Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Standard Deviation
        ax2 = axes[0, 1]
        ax2.bar(x, stats_df['Std'], color='orange', edgecolor='black')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Variability (Std Dev) by Class')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Min-Max Range
        ax3 = axes[1, 0]
        ax3.bar(x, stats_df['Max'] - stats_df['Min'], bottom=stats_df['Min'],
                color='lightblue', edgecolor='black')
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Days to Churn')
        ax3.set_title('Min-Max Range by Class')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Sample Count
        ax4 = axes[1, 1]
        ax4.bar(x, stats_df['Count'], color='lightgreen', edgecolor='black')
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Count')
        ax4.set_title('Sample Count by Class')
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(stats_df['Count']):
            ax4.text(i, v + 5, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def _plot_bar_with_values(self, ax, data, title, color):
        """Helper method to plot bar chart with value labels."""
        data.plot(kind='bar', ax=ax, color=color, edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('Sample Class')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, v in enumerate(data.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom')

    def _plot_train_test_comparison(self, ax, train_dist, test_dist):
        """Plot train vs test distribution comparison."""
        x = np.arange(len(train_dist))
        width = 0.35

        bars1 = ax.bar(x - width/2, train_dist.values, width,
                       label='Train', color='coral', edgecolor='black')
        bars2 = ax.bar(x + width/2, test_dist.values, width,
                       label='Test', color='lightcoral', edgecolor='black')

        ax.set_xlabel('Sample Class')
        ax.set_ylabel('Count')
        ax.set_title('Train vs Test Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(train_dist.index)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

    def _plot_percentage_distribution(self, ax, train_dist, test_dist):
        """Plot percentage distribution comparison."""
        train_pct = (train_dist / train_dist.sum() * 100).round(1)
        test_pct = (test_dist / test_dist.sum() * 100).round(1)

        x = np.arange(len(train_pct))
        width = 0.35

        bars1 = ax.bar(x - width/2, train_pct.values, width,
                       label='Train %', color='darkblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, test_pct.values, width,
                       label='Test %', color='navy', edgecolor='black')

        ax.set_xlabel('Sample Class')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Train vs Test Percentage Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(train_pct.index)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add percentage labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    def _plot_bin_ranges(self):
        """Plot bin ranges summary."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.set_title('DAYS_TO_CHURN Ranges by Sample Class', fontsize=14)

        # Create text summary
        y_pos = len(self.bin_ranges)
        for i, bin_range in enumerate(self.bin_ranges):
            ax.text(0.05, y_pos - i - 1, bin_range, fontsize=12,
                   transform=ax.transData, bbox=dict(boxstyle="round,pad=0.3",
                                                     facecolor="lightblue",
                                                     edgecolor="black",
                                                     alpha=0.7))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(self.bin_ranges))
        ax.axis('off')

        plt.tight_layout()
        plt.show()

    def get_results(self) -> Dict:
        """
        Get all results from the stratification process.

        Returns:
        --------
        Dictionary containing all results and statistics
        """
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'stratified_df': self.stratified_df,
            'distribution_stats': self.distribution_stats,
            'stratification_success': self.stratification_success
        }

    def print_summary(self):
        """Print a comprehensive summary of the stratification results."""
        print("\n" + "="*60)
        print("STRATIFICATION SUMMARY")
        print("="*60)

        print(f"\nTotal samples processed: {self.distribution_stats['total_samples']}")
        print(f"Train samples: {self.distribution_stats['train_samples']} "
              f"({self.distribution_stats['train_samples']/self.distribution_stats['total_samples']*100:.1f}%)")
        print(f"Test samples: {self.distribution_stats['test_samples']} "
              f"({self.distribution_stats['test_samples']/self.distribution_stats['total_samples']*100:.1f}%)")
        print(f"Stratification successful: {self.stratification_success}")

        print("\nClass distributions:")
        print("-" * 40)
        print(f"{'Class':>10} | {'Train':>10} | {'Test':>10}")
        print("-" * 40)

        for class_label in sorted(self.distribution_stats['train_distribution'].keys()):
            train_count = self.distribution_stats['train_distribution'][class_label]
            test_count = self.distribution_stats['test_distribution'].get(class_label, 0)
            print(f"{class_label:>10} | {train_count:>10} | {test_count:>10}")

        print("="*60)


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Create example data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'CUST_ACCOUNT_NUMBER': range(1000),
        'DAYS_TO_CHURN': np.random.exponential(scale=500, size=1000),
        'FEATURE_1': np.random.normal(100, 15, 1000),
        'FEATURE_2': np.random.uniform(0, 1, 1000)
    })

    # Example matching your exact code
    print("=== Example: Matching Your Original Code ===")
    print("-" * 50)

    # Initialize sampler with your parameters
    sampler = StratifiedChurnSampler(
        min_samples_per_class=2,
        bin_days=[365, 730, 1095, 1460],
        random_state=42
    )

    # Perform stratification with all visualizations
    results = sampler.fit_transform(
        sample_data,                  # Your cust_churn_df
        test_size=0.2,               # Your test size
        keep_all_data=True,          # Keep all data by combining
        remove_last_bin=True,        # Remove SAMPLE==5
        plot_distributions=True,     # Basic distributions
        plot_dashboard=True,         # Comprehensive dashboard
        plot_verification=True       # Verification plots
    )

    # Get your variables
    X_train = results['X_train']
    X_test = results['X_test']
    y_train = results['y_train']
    y_test = results['y_test']

    # Print summary
    sampler.print_summary()

    # Show statistical summary with plots
    sampler.plot_statistical_summary()