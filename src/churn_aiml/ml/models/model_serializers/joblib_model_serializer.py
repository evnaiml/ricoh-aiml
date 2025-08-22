"""
Model serialization module with automatic format detection.

Supports:
- CatBoost models: Native .cbm format
- Other models: Joblib with optional .gz compression
- Automatic format detection on load
- Metadata tracking in JSON
"""
# -----------------------------------------------------------------------------
# Author: Evgeni Nikoolaev
# email: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# UPDATED ON: 2025-08-11
# CREATED ON: 2025-08-11
# -----------------------------------------------------------------------------
# COPYRIGHT@2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh
# -----------------------------------------------------------------------------

import joblib
import catboost as cb
from pathlib import Path
from typing import Any, Union
import json
from datetime import datetime


class JoblibModelSerializer:
    """
    Smart model serializer with automatic format detection.

    Suffix rules:
    - CatBoost models → always .cbm (removes .joblib/.gz if present)
    - Other models:
      • .joblib → saves without compression
      • .joblib.gz → saves with compression
      • no suffix or other → defaults to .joblib.gz (compressed)
    """

    @staticmethod
    def save_model(model: Any, save_path: Union[str, Path]) -> Path:
        """
        Save model with automatic format selection.

        Suffix behavior:
        - CatBoost: forces .cbm (replaces .joblib/.gz if present)
        - Others:
          • model.joblib → saves without compression
          • model.joblib.gz → saves with compression
          • model or model.other → saves as model.joblib.gz (compressed)
        """
        save_path = Path(save_path)

        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if it's a CatBoost model
        if isinstance(model, (cb.CatBoostRegressor, cb.CatBoostClassifier)):
            # Force CatBoost native format (.cbm)
            # Remove any .joblib.gz or other extensions
            save_path_str = str(save_path)
            if save_path_str.endswith('.joblib.gz'):
                # Replace .joblib.gz with .cbm
                save_path = Path(save_path_str[:-10] + '.cbm')
            elif save_path_str.endswith('.joblib'):
                # Replace .joblib with .cbm
                save_path = Path(save_path_str[:-7] + '.cbm')
            elif save_path_str.endswith('.gz'):
                # Replace .gz with .cbm
                save_path = Path(save_path_str[:-3] + '.cbm')
            elif not save_path.suffix or save_path.suffix != '.cbm':
                # Add or replace with .cbm
                save_path = save_path.with_suffix('.cbm')

            model.save_model(str(save_path))
            print(f"✓ CatBoost model saved in native format: {save_path}")
        else:
            # Use joblib for other models
            save_path_str = str(save_path)

            # Check existing extensions and handle accordingly
            if save_path_str.endswith('.joblib.gz'):
                # Already has .joblib.gz - save with compression
                joblib.dump(model, save_path, compress=3)
                print(f"✓ Model saved with joblib (compressed): {save_path}")
            elif save_path_str.endswith('.joblib'):
                # Has .joblib - save without compression
                joblib.dump(model, save_path, compress=0)
                print(f"✓ Model saved with joblib: {save_path}")
            else:
                # No .joblib extension - add .joblib.gz by default
                if save_path.suffix:
                    # Has some other extension, replace with .joblib.gz
                    save_path = save_path.with_suffix('.joblib.gz')
                else:
                    # No extension, add .joblib.gz
                    save_path = Path(save_path_str + '.joblib.gz')

                # Save with compression
                joblib.dump(model, save_path, compress=3)
                print(f"✓ Model saved with joblib (compressed): {save_path}")

        # Save metadata
        metadata = {
            'model_type': type(model).__name__,
            'saved_at': datetime.now().isoformat(),
            'format': 'catboost_native' if isinstance(model, (cb.CatBoostRegressor, cb.CatBoostClassifier)) else 'joblib'
        }
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return save_path

    @staticmethod
    def load_model(model_path: Union[str, Path]) -> Any:
        """
        Load model with automatic format detection based on file extension.
        Handles .cbm (CatBoost), .joblib, .pkl, and .gz files.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Try to load metadata to determine format
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Model type: {metadata['model_type']}")

        # Load based on file extension
        if model_path.suffix == '.cbm':
            # CatBoost native format
            model = cb.CatBoostRegressor()
            model.load_model(str(model_path))
            print(f"✓ CatBoost model loaded from: {model_path}")
        elif model_path.suffix in ['.joblib', '.pkl'] or str(model_path).endswith('.gz'):
            # Joblib format (automatically handles .gz decompression)
            model = joblib.load(model_path)
            if str(model_path).endswith('.gz'):
                print(f"✓ Model loaded with joblib (decompressed) from: {model_path}")
            else:
                print(f"✓ Model loaded with joblib from: {model_path}")
        else:
            # Try joblib as default
            try:
                model = joblib.load(model_path)
                print(f"✓ Model loaded from: {model_path}")
            except:
                raise ValueError(f"Unknown model format: {model_path.suffix}")

        return model


# ========== USAGE EXAMPLE ==========

if __name__ == "__main__":
    # After your training:
    # print("Training Starts: Total time ~ ")
    # with timer():
    #     bayes_search.fit(train_X, train_y)
    #     print("Training Finishes")

    # Get the best model
    # best_model = bayes_search.best_estimator_

    # Example 1: Save the model
    print("=== Saving Model ===")

    # Assume best_model is from: best_model = bayes_search.best_estimator_

    # For CatBoost model (will use .cbm format automatically)
    # saved_path = JoblibModelSerializer.save_model(
    #     model=best_model,
    #     save_path="outputs/models/churn_model_v1"  # Extension added automatically
    # )

    # Or specify full path with extension
    # saved_path = JoblibModelSerializer.save_model(
    #     model=best_model,
    #     save_path="outputs/models/2024/churn_model.cbm"
    # )

    # For compressed joblib (add .gz to compress)
    # saved_path = JoblibModelSerializer.save_model(
    #     model=sklearn_model,
    #     save_path="outputs/models/model.joblib.gz"  # Will be compressed
    # )

    # Example 2: Load the model
    print("\n=== Loading Model ===")

    # loaded_model = JoblibModelSerializer.load_model(saved_path)

    # Use the loaded model
    # predictions = loaded_model.predict(X_new)

    # Example 3: Quick functions for your workflow
    def save_churn_model(bayes_search, path="outputs/models/churn_model"):
        """Save the best model from BayesSearchCV."""
        best_model = bayes_search.best_estimator_
        return JoblibModelSerializer.save_model(best_model, path)

    def load_churn_model(path):
        """Load a saved churn model."""
        return JoblibModelSerializer.load_model(path)

    # Use in your code:
    # After training
    # saved_path = save_churn_model(bayes_search, "outputs/models/docuware_churn_v1")

    # Later
    # model = load_churn_model("outputs/models/docuware_churn_v1.cbm")