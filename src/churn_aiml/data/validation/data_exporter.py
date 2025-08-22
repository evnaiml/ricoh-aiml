"""
Data Exporter Class for CSV export, schema generation, and comprehensive reporting.

This module handles the export of analyzed DataFrames with:
- CSV data export with proper type formatting
- JSON schema generation with Pydantic-compatible type definitions
- Detailed analysis reports with inferred data types
- Proper formatting of datetime values (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
- Integer values without unnecessary decimal points
- NaN handling for failed conversions (displayed as null in JSON, 'NaN' in CSV)

Key Features:
- Exports DataFrames to CSV with type-aware formatting
- Generates JSON schemas with properly formatted sample values
- Creates comprehensive reports with 'Inferred_Dtype' column
- Handles datetime objects with appropriate ISO format strings
- Removes decimal zeros from integer values
- Properly represents NaN values in all output formats
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-13
# * CREATED ON: 2025-07-31
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# -----------------------------------------------------------------------------

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import numpy as np

from churn_aiml.data.validation.pydantic_analyzer import PydanticDataAnalyzer


class DataExporter:
    """General data exporter class for CSV export, schema generation, and reporting"""

    def __init__(self, output_dir: Path, logger=None):
        """
        Initialize the DataExporter

        Args:
            output_dir: Path where files will be saved
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.analyzer = PydanticDataAnalyzer()

        # Create directory structure if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.logger:
            self.logger.info(f"📁 DataExporter initialized with output directory: {self.output_dir}")

    def export_with_analysis(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """
        Export DataFrame with Pydantic analysis and reporting

        Args:
            df: DataFrame to export
            table_name: Name of the table/dataset

        Returns:
            Dict with export results including file paths and metadata
        """

        # File paths
        csv_path = self.output_dir / f"{table_name}.csv"
        schema_path = self.output_dir / f"{table_name}_schema.json"
        report_path = self.output_dir / f"{table_name}_report.csv"

        # Export CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        if self.logger:
            self.logger.info(f"✅ CSV exported: {csv_path}")

        # Generate Pydantic analysis
        analysis_result = self.analyzer.analyze_dataframe(df, table_name)

        # Save schema with proper JSON serialization
        schema_to_save = self._prepare_schema_for_json(analysis_result['schema'])
        with open(schema_path, 'w') as f:
            json.dump(schema_to_save, f, indent=2, default=str)
        if self.logger:
            self.logger.info(f"✅ Schema saved: {schema_path}")

        # Generate CSV report
        report_df = self.generate_report_csv(df, table_name, analysis_result)
        report_df.to_csv(report_path, index=False, encoding='utf-8')
        if self.logger:
            self.logger.info(f"✅ Report saved: {report_path}")

        file_size_mb = csv_path.stat().st_size / (1024 * 1024)

        return {
            'csv_path': csv_path,
            'schema_path': schema_path,
            'report_path': report_path,
            'rows': len(df),
            'columns': len(df.columns),
            'size_mb': file_size_mb
        }

    def _prepare_schema_for_json(self, schema: Dict) -> Dict:
        """
        Prepare schema for JSON serialization by formatting sample values
        """
        prepared_schema = schema.copy()
        
        for field_name, field_info in prepared_schema['fields'].items():
            if 'metadata' in field_info and 'sample_values' in field_info['metadata']:
                # Format sample values based on actual_dtype
                actual_dtype = field_info['metadata'].get('actual_dtype', 'unknown')
                sample_values = field_info['metadata']['sample_values']
                
                # Format sample values properly for JSON
                formatted_samples = []
                for val in sample_values:
                    if pd.isna(val):
                        # Keep NaN as null in JSON
                        formatted_samples.append(None)
                    elif actual_dtype in ['datetime', 'inferred_datetime']:
                        # Convert datetime objects to ISO format strings
                        if isinstance(val, (pd.Timestamp, datetime)):
                            # Format as date only if time is midnight, otherwise include time
                            if val.hour == 0 and val.minute == 0 and val.second == 0:
                                formatted_samples.append(val.strftime('%Y-%m-%d'))
                            else:
                                formatted_samples.append(val.strftime('%Y-%m-%d %H:%M:%S'))
                        elif isinstance(val, str):
                            # Keep special strings like 'Latest Month' as is
                            formatted_samples.append(val)
                        else:
                            formatted_samples.append(None)
                    elif pd.isna(val):
                        formatted_samples.append(None)
                    elif actual_dtype in ['int', 'inferred_int']:
                        # Ensure integers are stored as numbers
                        if isinstance(val, (int, np.integer)):
                            formatted_samples.append(int(val))
                        elif isinstance(val, (float, np.floating)) and np.isfinite(val) and float(val).is_integer():
                            formatted_samples.append(int(val))
                        else:
                            formatted_samples.append(val)
                    elif actual_dtype in ['float', 'inferred_float']:
                        # Remove unnecessary decimal zeros for floats
                        if isinstance(val, (float, np.floating)) and np.isfinite(val) and float(val).is_integer():
                            formatted_samples.append(int(val))
                        else:
                            formatted_samples.append(val)
                    else:
                        # Keep other values as is
                        formatted_samples.append(val)
                
                field_info['metadata']['sample_values'] = formatted_samples
                
                # Convert missing_count to string for JSON consistency
                field_info['metadata']['missing_count'] = str(field_info['metadata']['missing_count'])
        
        return prepared_schema

    def generate_report_csv(self, df: pd.DataFrame, table_name: str, analysis_result: Dict) -> pd.DataFrame:
        """
        Generate detailed analysis report as CSV DataFrame

        Args:
            df: DataFrame being analyzed
            table_name: Name of the table
            analysis_result: Results from Pydantic analysis

        Returns:
            DataFrame containing the analysis report
        """

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_size_mb = (df.memory_usage(deep=True).sum()) / (1024 * 1024)
        total_rows = len(df)

        # Prepare report data
        report_data = []

        # Add header information as first rows
        header_info = [
            {"Feature_Name": "REPORT_INFO", "DType": "Generated", "Inferred_Dtype": "", "Feature_Store": timestamp, "Missing": "", "Sample": "", "Table_Name": table_name},
            {"Feature_Name": "SUMMARY", "DType": f"Rows: {total_rows:,}", "Inferred_Dtype": "", "Feature_Store": f"Cols: {len(df.columns)}", "Missing": f"Size: {file_size_mb:.2f}MB", "Sample": "", "Table_Name": table_name},
            {"Feature_Name": "---", "DType": "---", "Inferred_Dtype": "---", "Feature_Store": "---", "Missing": "---", "Sample": "---", "Table_Name": "---"},
        ]
        report_data.extend(header_info)

        # Add feature details (sorted alphabetically)
        for feature_name in sorted(df.columns):
            if feature_name in analysis_result['features']:
                feature_info = analysis_result['features'][feature_name]

                # Use the enhanced dtype directly (already includes unique counts)
                dtype = feature_info['inferred_type']
                
                # Get the actual inferred dtype
                actual_dtype = feature_info.get('actual_dtype', 'unknown')
                
                # Format the inferred dtype for display
                if actual_dtype == 'inferred_int':
                    inferred_dtype_display = 'int (inferred)'
                elif actual_dtype == 'inferred_float':
                    inferred_dtype_display = 'float (inferred)'
                elif actual_dtype == 'inferred_datetime':
                    inferred_dtype_display = 'datetime (inferred)'
                else:
                    inferred_dtype_display = actual_dtype

                role = feature_info.get('role', 1)
                missing_count = feature_info['missing_count']

                # Enhanced missing with percentage
                if total_rows > 0:
                    missing_pct = (missing_count / total_rows) * 100
                    missing_str = f"{missing_count} ({missing_pct:.1f}%)"
                else:
                    missing_str = str(missing_count)

                # Format sample values based on actual dtype
                sample_values = feature_info['sample_values'][:10]
                
                # Format samples appropriately for CSV display
                formatted_samples = []
                for val in sample_values:
                    if pd.isna(val):
                        # Display NaN as 'NaN' in CSV
                        formatted_samples.append('NaN')
                    elif actual_dtype in ['datetime', 'inferred_datetime']:
                        # Format datetime objects for display
                        if isinstance(val, (pd.Timestamp, datetime)):
                            # Format as date only if time is midnight, otherwise include time
                            if val.hour == 0 and val.minute == 0 and val.second == 0:
                                formatted_samples.append(val.strftime('%Y-%m-%d'))
                            else:
                                formatted_samples.append(val.strftime('%Y-%m-%d %H:%M:%S'))
                        elif isinstance(val, str):
                            # Keep special strings like 'Latest Month' as is
                            formatted_samples.append(val)
                        else:
                            formatted_samples.append('NaN')
                    elif pd.isna(val):
                        formatted_samples.append('NaN')
                    elif actual_dtype in ['int', 'inferred_int']:
                        # Show integers without decimal points
                        if isinstance(val, (float, np.floating)) and np.isfinite(val) and float(val).is_integer():
                            formatted_samples.append(int(val))
                        else:
                            formatted_samples.append(val)
                    elif actual_dtype in ['float', 'inferred_float']:
                        # Remove unnecessary decimal zeros
                        if isinstance(val, (float, np.floating)) and np.isfinite(val) and float(val).is_integer():
                            formatted_samples.append(int(val))
                        else:
                            formatted_samples.append(val)
                    else:
                        # Keep other values as is
                        formatted_samples.append(val)
                
                sample_str = str(formatted_samples)

                report_data.append({
                    "Feature_Name": feature_name,
                    "DType": dtype,  # Now uses enhanced dtype directly
                    "Inferred_Dtype": inferred_dtype_display,  # New column
                    "Feature_Store": role,
                    "Missing": missing_str,
                    "Sample": sample_str,
                    "Table_Name": table_name
                })

        return pd.DataFrame(report_data)

    def export_multiple_tables(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Export multiple tables at once

        Args:
            data_dict: Dictionary with table_name -> DataFrame mapping

        Returns:
            Dict with summary of all exports
        """

        results = {}
        successful_exports = []
        failed_exports = []

        for table_name, df in data_dict.items():
            try:
                result = self.export_with_analysis(df, table_name)
                successful_exports.append({'table': table_name, 'result': result})
                results[table_name] = result

                if self.logger:
                    self.logger.info(f"✅ {table_name}: exported successfully "
                                   f"({result['rows']:,} rows, {result['size_mb']:.2f} MB)")

            except Exception as e:
                failed_exports.append({'table': table_name, 'error': str(e)})
                if self.logger:
                    self.logger.error(f"❌ {table_name}: failed - {str(e)}")

        return {
            'successful_exports': successful_exports,
            'failed_exports': failed_exports,
            'total_success': len(successful_exports),
            'total_failed': len(failed_exports),
            'results': results
        }