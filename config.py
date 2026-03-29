"""Central configuration for model hyperparameters."""

LGBM_PARAMS = dict(
    num_leaves=200,
    learning_rate=0.0355003,
    n_estimators=1670,
    min_child_samples=55,
    feature_fraction=0.810305,
    bagging_fraction=0.92132,
    bagging_freq=6,
    reg_alpha=0.224335,
    reg_lambda=4.80759,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)