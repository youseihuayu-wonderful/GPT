<a href="record.md"><kbd>English</kbd></a> <a href="record.zh-CN.md"><kbd>中文</kbd></a>

# Level 1 end-to-end pilot · 2026-11-04

This is the English counterpart of the Chinese experimental record, not a new experiment. The original data, logs, model files, and historical videos remain unchanged.

**Conclusion: the pipeline for collecting real labels, training a classifier, predicting three speed settings, and physically executing its recommendations has run successfully. This batch did not demonstrate an advantage over the fixed fastest setting, nor did it test whether active learning outperforms random sampling or D-optimal design.**

## Scene, variables, and labels

- The experiment uses the MuJoCo 3.3.7 / Franka Panda workcell from the “black bowl into the drawer” recording, targeting the existing black bowl and bottle. It performs approach, descent, gripper closure, lift, and hold only. Drawer placement and closure are excluded, as is six-object sorting.
- The descent speed settings are **0.04, 0.10, and 0.20 m/s**. These are trajectory parameters, not a constant speed throughout the task or a measured constant TCP speed.
- Varied conditions are object category, reset XY offsets (±0.08 m on each axis), and yaw (±0.20 rad). Actual settled poses are saved separately, but the current classifier uses the prescribed reset parameters. There is no visual recognition.
- Each object's geometry, mass, friction, and grasping rule remain fixed. Initial gripper opening is fixed at 0.08 m for the bowl and 0.04 m for the bottle; the closure trigger height is 0 for both.
- Success requires an actual lift of at least 0.10 m and **2 continuous seconds** of bilateral grip, table clearance, and stable relative pose. An independent observer checks every 2 ms; the old page's 1-second controller success flag is not sufficient. Traces are saved every 20 ms and at phase transitions.
- Invalid initial states have no label. Numerical or internal errors abort the study rather than being disguised as negative examples. Expected controller failures are retained. Unexpected contacts are recorded separately; grasp success does not certify collision-free safety.

## Collection, training, and execution

The protocol was saved before labels were collected: 36 training conditions × 3 speeds, and 18 conditions distinct from training × 3 speeds for evaluation. Both splits use the same two specific objects, so this is not an unseen-object generalization test. The seed is fixed at 5644; no parameters were tuned using evaluation outcomes.

| Split | Attempts | Success labels | Failure labels | Invalid initial states |
|---|---:|---:|---:|---:|
| Training | 108 | 81 | 18 | 9 |
| Evaluation | 54 | 48 | 6 | 0 |

The valid training set contains 99 rows from 33 conditions; evaluation contains 54 rows from 18 conditions. The three speed outcomes within a condition are correlated and must not be treated as 54 independent objects. The model is a **random forest with 128 trees and maximum depth 6**. Its six input columns are the bowl indicator, bottle indicator, X and Y offsets, yaw, and descent speed. It does not use GPT, a 27B model, MRIoC, or paid model APIs.

All three-speed predictions and recommendations were saved before any evaluation execution. For each evaluation condition, the recommended speed was physically executed first, followed by the other two speeds from the same initial state. Previously observed labels were not used as a substitute for execution, and evaluation outcomes were not used for retraining. Initial physical-state fingerprints must match exactly across the three speeds.

The policy selects the fastest setting with a predicted value ≥0.9. If no setting meets the threshold, it selects the highest predicted value, breaking ties in favor of the slower setting, and explicitly marks `threshold_met=false`. Predictions are uncalibrated: this is not a 90% safety guarantee. In this batch, 3 recommendations did not meet the threshold and were still executed under the predefined rule, in simulation only.

| Prediction example | 0.04 m/s | 0.10 m/s | 0.20 m/s | Recommended setting | Actual recommended execution |
|---|---:|---:|---:|---:|---|
| `bowl.evaluation_00` | 1.0 | 1.0 | 1.0 | 0.20 | Success |
| `bottle.evaluation_02` | 0.0 | 0.0 | 0.0 | 0.04, below threshold | Failure |
| `bottle.evaluation_08` | 0.0 | 0.0 | 0.0 | 0.04, below threshold | Success |

These are model outputs, not calibrated success probabilities. The third row directly illustrates an incorrect prediction. See [recommendations.json](recommendations.json) for all recommendations and [recommendation_outcomes.json](recommendation_outcomes.json) for actual outcomes. Classification accuracy is 51/54 = 94.44%, with a Brier score of 0.05556. Classification performance must not be presented as a benefit from speed selection.

## Collection throughput and task performance

On the local Apple M5 Pro / 48 GB machine, collection ran headlessly while a separate live preview remained available. The entire command took **339.51 wall-clock seconds**; timing inside the study function was 338.58 seconds. The 153 trials with valid labels averaged **2.189 wall-clock seconds per trial**, including simulation construction, settling, execution, the 2-second hold observation, trace saving, and cleanup, but excluding the final per-trial row JSON write. All 162 attempts, including invalid initial states, averaged 2.087 seconds. The lower average that includes invalid setups must not be reported as valid-label collection throughput. Classifier training took **0.043 seconds**.

A rough extrapolation from this batch suggests about 40 minutes for 1,000 valid labels. This is not a completed thousand-trial experiment, and the same throughput is not guaranteed on other hardware, objects, or more complex workflows.

| Policy | Successes / 18 evaluation conditions | Mean successful-task duration, simulation seconds |
|---|---:|---:|
| Fixed 0.04 m/s | 16/18 | 18.642 |
| Fixed 0.10 m/s | 16/18 | 14.138 |
| Fixed 0.20 m/s | 16/18 | 12.634 |
| Random-forest recommendation | 16/18 | 13.010 |

The model recommended the fastest setting 15 times and the slowest setting 3 times. On the 16 conditions where both policies succeeded, recommendations were on average **0.376 simulation seconds slower** than the fixed fastest setting. Across both training and evaluation, every valid condition had the same success/failure label at all three speeds. No condition was observed where the fastest setting failed but a slower setting succeeded.

Training failures occurred during descent without arrival at the target (15 trials), a timeout during lift (1 trial), and loss of the hold conditions (2 trials). All evaluation failures were descent-arrival failures (6 trials). Unexpected contacts were recorded in 9 evaluation trials, including 3 that still met the grasp-success definition. There is therefore no claim of collision-free operation or readiness for direct real-robot deployment.

## Reproduction and evidence

Run from the project root. The output directory must not already exist:

```sh
uv sync --locked
uv run --locked python run_grasp_learning.py --output results/grasp_level1_reproduction
```

The code entry points are [`run_grasp_learning.py`](../../run_grasp_learning.py), [`grasp_trial.py`](../../grasp_trial.py), and [`grasp_speed_model.py`](../../grasp_speed_model.py). Outputs include the [protocol](protocol.json), [environment and source/asset fingerprints](environment.json), [training data](training_data.json), [evaluation data](evaluation_data.json), [model](random_forest.joblib), [complete summary](summary.json), individual `trials/`, and compressed physical `traces/`. Only load joblib files that you trained yourself or explicitly trust.

This directory contains a **same-protocol replay after error-boundary fixes**. The first batch remains in [`../grasp_level1_20261104`](../grasp_level1_20261104). The two batches of 162 attempts have identical physical results, initial-state fingerprints, simulation times, predictions, and model SHA-256 values; see the [post-hoc audit](posthoc_audit.json). The replay is not a second independent test set and must not be pooled into 324 independent samples. The earlier 9 throughput trials and 20 development probes are also archived separately and are not used in this batch's training or metrics.

## Answers to the professor's three questions

1. **Is the simulation feasible? Yes, with measured evidence.** MuJoCo + Panda has completed real positive/negative data collection, training, and physical execution of recommendations. Current collection throughput supports a small classifier experiment.
2. **What features are used besides speed? Object category, reset XY offsets, and yaw.** Each object's physical parameters and grasping rule remain fixed. Only two specific objects have been tested, and cross-object generalization has not been demonstrated.
3. **Why not D-optimal? This batch does not justify ruling it out.** Active learning, random sampling, and D-optimal design have not yet been compared fairly, and no superiority claim is warranted. The immediate priority is to identify genuinely speed-sensitive outcomes under reasonable physical conditions. If such conditions cannot be found, the research question should be revised rather than manufacturing negative examples or adding more models to conceal the issue.

**Next step: establish whether speed selection presents a nontrivial success-rate / completion-time tradeoff, then proceed to active-learning comparisons. Engineering feasibility has been demonstrated; the effectiveness of the research hypothesis remains to be tested.**

## Code verification

The full regression run exercised 319 tests, and all 34 newly added Level 1 checks passed. 3 test methods in the historical frozen-sorting setup produced 9 errors including subtests because the current sources/dependencies no longer matched the historical fingerprints. The [complete initial log](regression_initial.log) is retained. The unit tests were then changed to use explicitly labelled working-tree snapshots without rewriting the historical experiment protocol or bypassing production fingerprint checks. The relevant 9 protocol/physics tests and 1 server-layout test passed. The full suite was not rerun afterward; commands and results are recorded in [verification.json](verification.json).

The following problems were reproduced before they were fixed: internal IK errors becoming negative labels, non-finite IK updates, invalid recommendation setups being counted as executed, lost identity of interrupted attempts, and overwriting prior trace files. The physical replay did not change object parameters, motion rules, thresholds, or label definitions. Fault-injection data were not used to train the model.
