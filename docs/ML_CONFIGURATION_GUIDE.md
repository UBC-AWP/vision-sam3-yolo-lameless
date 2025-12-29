# ML Configuration Guide for Researchers

This guide explains all configurable parameters for the CatBoost, XGBoost, and LightGBM ensemble models used in the cow lameness detection system.

## Quick Start

1. Navigate to **ML Config** in the sidebar
2. Select the model tab you want to configure (CatBoost, XGBoost, LightGBM, Ensemble, or Training)
3. Adjust parameters using sliders and inputs
4. Click **Save** to apply changes
5. Click **Start Training** to retrain models with new settings

---

## Parameter Categories

Parameters are organized into categories:
- **Training** - Controls how long and fast models learn
- **Tree Structure** - Controls model complexity
- **Regularization** - Prevents overfitting
- **Model** - Algorithm-specific settings
- **Ensemble** - How models are combined

---

## CatBoost Parameters

CatBoost is a gradient boosting library that handles categorical features well and is generally robust out-of-the-box.

| Parameter | Default | Range | What It Does | How to Tune |
|-----------|---------|-------|--------------|-------------|
| **iterations** | 100 | 10-10,000 | Number of trees to build | Start with 100-500. Increase if underfitting, decrease if slow or overfitting |
| **learning_rate** | 0.1 | 0.001-1.0 | Step size for each iteration | Lower values (0.01-0.05) = more accurate but needs more iterations |
| **depth** | 6 | 1-16 | Maximum tree depth | 4-8 for most cases. Deeper = more complex patterns but risk overfitting |
| **l2_leaf_reg** | 3.0 | 0-100 | L2 regularization strength | Increase (5-10) if overfitting. Decrease if underfitting |
| **random_strength** | 1.0 | 0-10 | Randomness in split scoring | Increase for more regularization |
| **bagging_temperature** | 1.0 | 0-10 | Bootstrap sampling intensity | Higher = more randomization, helps prevent overfitting |
| **border_count** | 254 | 1-255 | Splits for numerical features | Higher = more precision but slower. 128-254 is good |
| **grow_policy** | SymmetricTree | - | How trees grow | SymmetricTree (fastest), Depthwise, Lossguide (best quality) |
| **bootstrap_type** | MVS | - | Sampling method | MVS (recommended), Bayesian, Bernoulli, No |

### CatBoost Tuning Tips
```
For small datasets (< 1000 samples):
  - depth: 4-6
  - iterations: 100-300
  - l2_leaf_reg: 5-10

For larger datasets (> 5000 samples):
  - depth: 6-10
  - iterations: 500-2000
  - l2_leaf_reg: 1-5
```

---

## XGBoost Parameters

XGBoost is a highly optimized gradient boosting library with extensive regularization options.

| Parameter | Default | Range | What It Does | How to Tune |
|-----------|---------|-------|--------------|-------------|
| **n_estimators** | 100 | 10-10,000 | Number of boosting rounds | Similar to CatBoost iterations |
| **learning_rate** | 0.1 | 0.001-1.0 | Step size (eta) | 0.01-0.1 for best results |
| **max_depth** | 6 | 1-20 | Maximum tree depth | 3-10 typically. Start with 6 |
| **min_child_weight** | 1.0 | 0-100 | Minimum sum of instance weight | Increase (3-10) to prevent overfitting |
| **gamma** | 0.0 | 0-10 | Minimum loss reduction for split | 0-0.5 typically. Increase for conservative splits |
| **subsample** | 1.0 | 0.1-1.0 | Fraction of data per tree | 0.7-0.9 helps prevent overfitting |
| **colsample_bytree** | 1.0 | 0.1-1.0 | Fraction of features per tree | 0.7-0.9 adds randomness |
| **colsample_bylevel** | 1.0 | 0.1-1.0 | Fraction of features per level | Optional, 0.7-1.0 |
| **reg_alpha** | 0.0 | 0-100 | L1 regularization | 0-1 for feature selection |
| **reg_lambda** | 1.0 | 0-100 | L2 regularization | 1-10 for smoothing |
| **scale_pos_weight** | 1.0 | 0.1-100 | Class weight balance | Set to (negative_samples / positive_samples) for imbalanced data |
| **booster** | gbtree | - | Booster type | gbtree (default), dart (dropout), gblinear (linear) |
| **tree_method** | hist | - | Tree construction algorithm | hist (fast), exact (accurate), gpu_hist (GPU) |

### XGBoost Tuning Tips
```
Preventing Overfitting:
  - Reduce max_depth (3-6)
  - Increase min_child_weight (5-10)
  - Set subsample to 0.8
  - Set colsample_bytree to 0.8
  - Increase gamma (0.1-0.5)

For Imbalanced Data:
  - Set scale_pos_weight = count(negative) / count(positive)
  - Or use class weights in training
```

---

## LightGBM Parameters

LightGBM is known for speed and efficiency, using leaf-wise tree growth.

| Parameter | Default | Range | What It Does | How to Tune |
|-----------|---------|-------|--------------|-------------|
| **n_estimators** | 100 | 10-10,000 | Number of iterations | Start with 100-500 |
| **learning_rate** | 0.1 | 0.001-1.0 | Boosting learning rate | 0.01-0.1 recommended |
| **max_depth** | 6 | -1 to 20 | Maximum tree depth | -1 = no limit. 5-10 typical |
| **num_leaves** | 31 | 2-131,072 | Maximum leaves per tree | **Key parameter!** 20-50 for small data, 50-200 for large |
| **min_child_samples** | 20 | 1-1,000 | Minimum samples in leaf | Increase (50-100) to prevent overfitting |
| **min_child_weight** | 0.001 | 0-100 | Minimum sum of hessian | Rarely needs tuning |
| **subsample** | 1.0 | 0.1-1.0 | Row sampling fraction | 0.7-0.9 for regularization |
| **colsample_bytree** | 1.0 | 0.1-1.0 | Column sampling fraction | 0.7-0.9 typical |
| **reg_alpha** | 0.0 | 0-100 | L1 regularization | 0-1 for sparsity |
| **reg_lambda** | 0.0 | 0-100 | L2 regularization | 0-10 for smoothing |
| **min_split_gain** | 0.0 | 0-10 | Minimum gain for split | 0-0.5, increase if overfitting |
| **boosting_type** | gbdt | - | Boosting algorithm | gbdt (default), dart, goss (faster), rf (random forest) |

### LightGBM Tuning Tips
```
Important: num_leaves should be < 2^max_depth to avoid overfitting

For Accuracy:
  - num_leaves: 50-100
  - max_depth: 10-15
  - min_child_samples: 10-20

For Speed:
  - boosting_type: goss
  - num_leaves: 31
  - max_depth: 6-8
```

---

## Ensemble Configuration

The ensemble combines predictions from all three models for better accuracy.

| Parameter | Default | Range | What It Does | How to Tune |
|-----------|---------|-------|--------------|-------------|
| **catboost_weight** | 0.33 | 0-1 | CatBoost contribution | Increase if CatBoost performs best |
| **xgboost_weight** | 0.33 | 0-1 | XGBoost contribution | Increase if XGBoost performs best |
| **lightgbm_weight** | 0.34 | 0-1 | LightGBM contribution | Increase if LightGBM performs best |
| **voting_method** | soft | - | How predictions combine | **soft** (probabilities) or **hard** (majority vote) |
| **threshold** | 0.5 | 0-1 | Classification cutoff | Lower = more sensitive, Higher = more specific |

### Ensemble Tuning Tips
```
Setting Weights:
1. Train each model separately
2. Check individual accuracy on validation set
3. Give higher weights to better-performing models

Example: If XGBoost has 85% accuracy, CatBoost 80%, LightGBM 82%:
  - xgboost_weight: 0.40
  - lightgbm_weight: 0.35
  - catboost_weight: 0.25

Threshold Selection:
  - High sensitivity needed (catch all lame cows): threshold = 0.3-0.4
  - High specificity needed (reduce false positives): threshold = 0.6-0.7
  - Balanced: threshold = 0.5
```

---

## Training Configuration

General training settings that apply to all models.

| Parameter | Default | Range | What It Does | How to Tune |
|-----------|---------|-------|--------------|-------------|
| **min_samples** | 10 | 2-1,000 | Minimum samples to start training | Set based on your data volume |
| **cv_folds** | 5 | 2-20 | Cross-validation folds | 5-10 typical. More = slower but more reliable |
| **test_size** | 0.2 | 0.1-0.5 | Test set proportion | 0.2-0.3 typical |
| **stratify** | true | - | Keep class balance in splits | Always keep **true** for classification |
| **shuffle** | true | - | Shuffle before splitting | Usually keep **true** |
| **early_stopping_rounds** | null | 1-100 | Stop if no improvement | Set to 10-50 to prevent overfitting |
| **feature_selection** | false | - | Enable feature selection | Enable if many irrelevant features |
| **scale_features** | true | - | Standardize features | Usually keep **true** |

---

## Recommended Configurations

### Conservative (Prevent Overfitting)
```
CatBoost:
  depth: 4
  iterations: 200
  l2_leaf_reg: 10
  learning_rate: 0.05

XGBoost:
  max_depth: 4
  n_estimators: 200
  min_child_weight: 5
  subsample: 0.8
  colsample_bytree: 0.8

LightGBM:
  num_leaves: 20
  max_depth: 6
  min_child_samples: 50
  subsample: 0.8
```

### Aggressive (Maximum Accuracy)
```
CatBoost:
  depth: 8
  iterations: 1000
  l2_leaf_reg: 1
  learning_rate: 0.03

XGBoost:
  max_depth: 8
  n_estimators: 1000
  min_child_weight: 1
  subsample: 1.0

LightGBM:
  num_leaves: 100
  max_depth: 12
  min_child_samples: 10
  subsample: 1.0
```

### Balanced (Good Starting Point)
```
CatBoost:
  depth: 6
  iterations: 500
  l2_leaf_reg: 3
  learning_rate: 0.1

XGBoost:
  max_depth: 6
  n_estimators: 500
  min_child_weight: 3
  subsample: 0.9

LightGBM:
  num_leaves: 31
  max_depth: 8
  min_child_samples: 20
  subsample: 0.9
```

---

## Troubleshooting

### Model is Overfitting (high training accuracy, low test accuracy)
1. Reduce tree depth/num_leaves
2. Increase regularization (l2_leaf_reg, reg_lambda)
3. Reduce subsample and colsample_bytree to 0.7-0.8
4. Increase min_child_weight/min_child_samples
5. Enable early_stopping_rounds

### Model is Underfitting (low accuracy overall)
1. Increase tree depth/num_leaves
2. Increase iterations/n_estimators
3. Decrease regularization
4. Try lower learning_rate with more iterations

### Training is Too Slow
1. Reduce iterations/n_estimators
2. Use hist tree_method (XGBoost)
3. Use goss boosting_type (LightGBM)
4. Reduce cv_folds
5. Reduce max_depth

### Imbalanced Classes (more healthy than lame cows)
1. Set scale_pos_weight (XGBoost) to balance ratio
2. Adjust classification threshold (lower = catch more positives)
3. Use stratified splitting (keep stratify: true)

---

## Best Practices

1. **Start with defaults** - They work well for most cases
2. **Change one thing at a time** - Easier to see what helps
3. **Use cross-validation** - More reliable than single train/test split
4. **Monitor both metrics** - Check accuracy AND precision/recall
5. **Save good configurations** - Note what works for your data
6. **Reset to defaults** - If things get confusing, start fresh

---

## Quick Reference Card

| Goal | CatBoost | XGBoost | LightGBM |
|------|----------|---------|----------|
| More trees | iterations ↑ | n_estimators ↑ | n_estimators ↑ |
| Deeper trees | depth ↑ | max_depth ↑ | max_depth ↑ or num_leaves ↑ |
| Prevent overfitting | l2_leaf_reg ↑ | reg_lambda ↑, subsample ↓ | reg_lambda ↑, min_child_samples ↑ |
| Faster training | iterations ↓ | tree_method=hist | boosting_type=goss |
| Better accuracy | learning_rate ↓, iterations ↑ | learning_rate ↓, n_estimators ↑ | learning_rate ↓, n_estimators ↑ |
