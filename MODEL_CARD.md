# Model card

**Project:** Wearable Activity Recognition: Generalization to Unseen Participants  
**Author:** Dr. Şeyma Bozkurt Uzan  
**Purpose:** Reproducible biomedical engineering education and benchmark analysis.

## Models and intended use

A most-frequent classifier provides a trivial reference. Two standardized multinomial logistic-regression models compare the released 561-feature representation with a prespecified 18-feature accelerometer representation. The models support teaching of participant-aware evaluation, feature design and biomedical interpretation. The repository is not a deployed monitoring device or a clinical decision-support tool.

## Data

UCI HAR includes scripted activities recorded using waist-mounted smartphones from 30 volunteers aged 19–48. Six activity labels are provided. The official split has 7,352 windows from 21 training participants and 2,947 windows from nine held-out participants. All six labels occur for each participant in the audited data.

Windows overlap by 50%; treating them as independent observations would exaggerate the available evidence. Subject identifiers are used for grouping and are excluded from predictors. Released features are already filtered, engineered and normalized; this project does not reconstruct that upstream pipeline.

## Training and selection

The feature sets and candidate regularization values are declared before test evaluation. Five-fold `StratifiedGroupKFold` on training participants selects `C` from 0.01, 0.1, 1 and 10. The selection metric averages each training participant's macro-F1 equally. Scaling is learned inside each training fold. Full and compact models select C=1 and C=0.1, respectively, then refit on the complete training set. Seed: 571.

The compact representation comprises the mean and standard deviation of X/Y/Z for `tBodyAcc`, `tGravityAcc` and `tBodyAccJerk`. It contains no gyroscope or mixed-angle feature. Because both feature family and complexity change, the experiment is not a clean estimate of the causal effect of removing one sensor.

## Evaluation

Primary metric: equal-weight mean of six-class macro-F1 across the nine held-out participants. Secondary metrics include pooled macro-F1, accuracy, balanced accuracy, class recall and individual participant results.

| Model | Mean participant macro-F1 | 95% interval | Pooled accuracy |
|---|---:|---:|---:|
| Most-frequent reference | 0.051 | 0.049–0.053 | 0.182 |
| Full logistic regression | 0.952 | 0.923–0.977 | 0.955 |
| Compact logistic regression | 0.818 | 0.730–0.897 | 0.843 |

The paired compact-minus-full difference is −0.134 (95% interval −0.199 to −0.070). Two thousand bootstrap draws resample participants with their paired model scores intact. Intervals describe variability in this test-participant sample conditional on the fitted models; they exclude training and model-selection uncertainty. Nine people provide a limited uncertainty estimate.

For the full model, sitting recall is 0.876; 58 of 491 sitting windows are classified as standing. The activity confusion matrix and participant scores show where aggregate accuracy conceals specific failures. Exact class reports and counts are in [metrics.json](results/metrics.json).

## Limits and unsuitable uses

- The source cohort and scripted acquisition do not establish performance in patients, older populations, other phone locations, other devices or unscripted daily life.
- No disease labels, fall events, rehabilitation outcomes or treatment effects are evaluated.
- No device energy consumption, latency, deployment robustness or clinical benefit is measured.
- Current test results must not be used to repeatedly choose new models. Further development needs new validation data or a separately designed evaluation.
- Participant-level uncertainty does not establish demographic fairness; relevant demographic subgroup labels and validation were not available for this analysis.
- Upstream normalization is inherited. Only this project's additional scaling is demonstrably fit on training rows.

No fitted model binary is distributed. Run the documented analysis to reconstruct models and outputs. See [data licensing](DATA_LICENSE.md) for the historical/current source-license discrepancy.
