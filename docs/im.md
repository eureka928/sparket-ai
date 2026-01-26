# Incentive Mechanism (IM)

This document describes the incentive mechanism for the Sparket subnet.
It is written from a game theoretic perspective: we reward original,
sharp, and early odds that hold up against a strong market benchmark.

## Design goal
We want a system where the best response is to publish honest, sharp,
original probabilities early. Copy trading late should earn less, and
overconfident noise should be punished.

## Scoring pipeline overview
1. Ingest provider quotes and build ground truth closing lines.
2. Score each miner submission against closing lines (CLV, CLE, MES).
3. Score each submission against outcomes (Brier, LogLoss, PSS).
4. Aggregate miner metrics with time decay and shrinkage.
5. Compute calibration and sharpness.
6. Compute originality and lead-lag.
7. Normalize and combine into SkillScore and weights.

All windows and weights are deterministic and configured in
`sparket/validator/config/scoring_params.py`.

## Ground truth and closing lines
Validators ingest provider odds and create a consensus closing line
per market and side. This is the benchmark for both economic edge
and outcome skill scoring.

## Outcome submissions
Miners also submit outcomes for finished events. Validators record the
settled outcome and use it to compute outcome-based scores
(Brier, log-loss, and PSS).

## Per-submission economic metrics (odds vs close)
Let:
- `O_miner` be the miner decimal odds
- `p_miner` be the miner implied probability
- `O_close` be the closing decimal odds
- `p_close` be the closing implied probability

We compute:
- `CLV_odds = (O_miner - O_close) / O_close`
- `CLV_prob = (p_close - p_miner) / p_close`
- `CLE = O_miner * p_close - 1`
- `MES = 1 - min(1, |CLV_prob|)`

Interpretation:
- Positive CLV and CLE mean the miner beat the closing line.
- MES rewards staying close to the efficient market while still
  finding edge.

## Per-submission outcome metrics (proper scoring rules)
Let `p_k` be the miner probability for side k and `y_k` be the outcome.

We compute:
- Brier: `sum_k (p_k - y_k)^2`
- LogLoss: `-log(p_k*)` where k* is the realized outcome
- PSS: `1 - (miner_score / truth_score)`

PSS compares the miner to the closing line as a baseline. PSS > 0
means the miner beats the market.

## Time-to-close bonus (anti-copy incentive)
Scores are adjusted by time to event start:
- Early correct picks get full credit.
- Early wrong picks are clipped (uncertainty is expected).
- Late correct picks get reduced credit (copy risk).
- Late wrong picks get full penalty.

This is a logarithmic time factor with asymmetric treatment that
discourages last-minute copying.

## Rolling aggregates
Submissions are aggregated per miner with time decay:
- Recent submissions carry more weight.
- Effective sample size is tracked.
- Metrics are shrunk toward population means for low-sample miners.

Key aggregates:
- `brier_mean` and `fq_raw`
  - `FQ = 1 - 2 * brier_mean`
- `pss_mean` (time-adjusted PSS)
- `es_mean`, `es_std`, `es_adj`
  - `es_adj = es_mean / es_std` (Sharpe-like)
- `mes_mean`

## Calibration and sharpness
Calibration fits a logit regression:
```
logit(observed) = a + b * logit(predicted)
```
Calibration score:
```
CAL = 1 / (1 + |b - 1| + |a|)
```

Sharpness measures variance of predicted probabilities:
```
Sharp = min(1, var / target_var)
```

Bin edges are deterministically jittered per window to prevent
gaming the calibration bins.

## Originality and lead-lag
We compare miner probability time series to provider time series.

Source of Signal (SOS):
```
SOS = 1 - |correlation|
```

Lead ratio counts how often the miner moved before the market
on significant moves within a lead window.

High SOS and high lead ratio reward independent, early signals.

## SkillScore (final)
SkillScore combines four dimensions, each normalized into a comparable
0–1 range before weighting.

Task mapping:
- **Outcome accuracy**: ForecastDim (FQ_norm + CAL)
- **Outcome relative skill**: SkillDim (PSS_norm)
- **Odds origination edge**: EconDim (ES_norm + MES)
- **Information advantage**: InfoDim (SOS + LEAD)

Component definitions:
- **FQ_norm**: forecast quality from `FQ = 1 - 2 * brier_mean`, mapped to [0, 1].
- **CAL**: calibration score from the logit regression fit.
- **PSS_norm**: normalized, time-adjusted PSS vs market baseline.
- **ES_norm**: normalized economic edge from `ES_adj` (CLE mean/std).
- **MES**: market efficiency score.
- **SOS**: originality score `1 - |correlation|`.
- **LEAD**: lead ratio (how often the miner moves first).

Default dimension weights (from `sparket/validator/config/scoring_params.py`):
- ForecastDim: `w_fq = 0.60`, `w_cal = 0.40`
- EconDim: `w_edge = 0.70`, `w_mes = 0.30`
- InfoDim: `w_sos = 0.60`, `w_lead = 0.40`

Default SkillScore weights (targeting ~80/20 odds vs outcomes):
- `w_outcome_accuracy = 0.10`
- `w_outcome_relative = 0.10`
- `w_odds_edge = 0.50`
- `w_info_adv = 0.30`

Metrics are normalized and combined into four dimensions:

ForecastDim:
```
ForecastDim = w_fq * FQ_norm + w_cal * CAL
```

SkillDim:
```
SkillDim = PSS_norm
```
PSS is computed per submission (vs the market baseline) using both Brier
and log‑loss PSS, blended into a single value, time‑adjusted, aggregated
into `pss_mean`, then normalized to `PSS_norm`.

EconDim:
```
EconDim = w_edge * ES_norm + w_mes * MES
```

InfoDim:
```
InfoDim = w_sos * SOS + w_lead * LEAD
```

Final SkillScore:
```
SkillScore = w_outcome_accuracy * ForecastDim
           + w_outcome_relative * SkillDim
           + w_odds_edge * EconDim
           + w_info_adv * InfoDim
```

Normalization:
- FQ is mapped from [-1, 1] to [0, 1].
- PSS and ES are normalized via z-score logistic when enough miners
  exist, otherwise percentile normalization is used.
- CAL, MES, SOS, LEAD are clipped to [0, 1].

## How to excel (game theoretic view)
Dominant strategies are:
- Be early with information that survives to close.
- Be calibrated and sharp, not just extreme.
- Avoid copy-trading near close; time bonus reduces late credit.
- Maintain consistency to improve effective sample size and shrinkage.
- Provide independent signals that lead the market, not mirror it.

If you only mirror the closing line, you get:
- Low originality (SOS near 0)
- Lower lead ratio
- Less credit from time bonus

If you are noisy or overconfident, you get:
- Poor calibration (low CAL)
- Bad Brier and PSS

The best response is to publish honest probabilities early, with
evidence-backed deviations from the market.
