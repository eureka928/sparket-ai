# Scoring Equations Walkthrough (End-to-End)

This walkthrough follows the full scoring path from raw data ingestion to the
final SkillScore. Each step explains the surrounding logic and the equations
used to transform data into the final composite.

## Step 1: Ingest data and define ground truth
Validators ingest provider odds and build a consensus closing line for each
market and side. Outcomes are recorded when events settle. These two data
sources power the two task tracks: odds origination (vs close) and outcome
verification (vs realized result).

Notation used throughout:
- Miner odds: O_miner
- Closing odds: O_close
- Miner probabilities: p_miner or p_k
- Closing probabilities: p_close
- Realized outcome vector: y_k (one-hot)
- Submission time: t_submit
- Event start time: t_start
- Minutes to close: t_min
- Decay weight: w_i

## Step 2: Score odds origination per submission
Each submission is compared to the closing line. These are the raw signals for
economic edge and market efficiency.

$$CLV_{odds} = \frac{O_{miner} - O_{close}}{O_{close}}$$
$$CLV_{prob} = \frac{p_{close} - p_{miner}}{p_{close}}$$
$$CLE = O_{miner} \cdot p_{close} - 1$$
$$MES = 1 - \min\left(1,\; \left|CLV_{prob}\right|\right)$$

## Step 3: Score outcome accuracy per submission
Once outcomes settle, submissions are scored against the realized result.
These raw accuracy metrics feed forecast quality (FQ) and relative skill (PSS).

$$Brier = \sum_{k} \left(p_k - y_k\right)^2$$
$$LogLoss = -\log\left(p_{k^*}\right)$$

## Step 4: Convert accuracy into relative skill (PSS)
PSS compares the miner to the market baseline, yielding a relative skill
signal that is robust to market difficulty.

$$PSS = 1 - \frac{Score_{miner}}{Score_{baseline}}$$
$$PSS_{brier} = 1 - \frac{Brier_{miner}}{Brier_{baseline}}$$
$$PSS_{log} = 1 - \frac{LogLoss_{miner}}{LogLoss_{baseline}}$$
$$PSS_{blend} = \frac{PSS_{brier} + PSS_{log}}{2}$$

## Step 5: Apply time-to-close adjustment
Outcome skill is time-adjusted to reward early correct signals and reduce
credit for late copy-trading.

$$f(t) = \begin{cases} f_{min}, & t \le t_{min} \\ 1, & t \ge t_{max} \\ f_{min} + \left(1 - f_{min}\right)\frac{\log t - \log t_{min}}{\log t_{max} - \log t_{min}}, & t_{min} < t < t_{max} \end{cases}$$
$$Score_{time} = \begin{cases} Score \cdot f(t), & Score \ge 0 \\ Score \cdot \left(c + (1 - c)(1 - f(t))\right), & Score < 0 \end{cases}$$

## Step 6: Aggregate with decay and compute n_eff
All per-submission metrics are aggregated over a rolling window with
exponential decay. This emphasizes recent performance.

$$w_i = \exp\left(\ln(0.5)\cdot \frac{age_i}{half\_life}\right)$$
$$\overline{x} = \frac{\sum_i w_i x_i}{\sum_i w_i}$$
$$\sigma = \sqrt{\frac{\sum_i w_i (x_i - \overline{x})^2}{\sum_i w_i}}$$
$$n_{eff} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}$$

## Step 7: Shrink toward population mean
To reduce small-sample volatility, metrics are shrunk toward the population
mean when n_eff is low.

$$\alpha = \frac{\log(1 + n_{eff})}{\log(1 + n_{eff} + k)}$$
$$x_{shrunk} = \alpha \cdot x + (1 - \alpha)\cdot \mu$$

## Step 8: Compute rolling aggregates
These aggregates summarize the miner’s performance over the window.

$$ES_{adj} = \frac{ES_{mean}}{ES_{std}}$$
$$FQ = 1 - 2 \cdot Brier_{mean}$$

## Step 9: Calibration and sharpness
Calibration evaluates statistical reliability; sharpness rewards decisiveness.

$$logit(\hat{p}) = a + b \cdot logit(p)$$
$$CAL = \frac{1}{1 + |b - 1| + |a|}$$
$$Sharp = \min\left(1,\; \frac{Var(p)}{Var_{target}}\right)$$

## Step 10: Originality and lead-lag
Originality measures independence from the market; lead-lag measures whether
signals tend to move before consensus.

$$\rho = corr(p_{miner}, p_{market})$$
$$SOS = 1 - |\rho|$$
$$Lead = \frac{Moves_{led}}{Moves_{matched}}$$

## Step 11: Normalize metrics across miners
Metrics are normalized to a common 0–1 scale for combination.

$$FQ_{norm} = \frac{FQ + 1}{2}$$
$$z = \frac{x - \mu}{\sigma}$$
$$Norm = \frac{1}{1 + e^{-\alpha z}}$$
$$Norm = \frac{rank(x)}{n + 1}$$

## Step 12: Build dimension scores
Normalized components are grouped into four dimensions aligned to the tasks.

$$ForecastDim = w_{fq} \cdot FQ_{norm} + w_{cal} \cdot CAL$$
$$SkillDim = PSS_{norm}$$
$$EconDim = w_{edge} \cdot ES_{norm} + w_{mes} \cdot MES$$
$$InfoDim = w_{sos} \cdot SOS + w_{lead} \cdot Lead$$

## Step 13: Final SkillScore
The final SkillScore is the single composite used for chain weights.

$$SkillScore = w_{outcome\_accuracy}\cdot ForecastDim + w_{outcome\_relative}\cdot SkillDim + w_{odds\_edge}\cdot EconDim + w_{info\_adv}\cdot InfoDim$$
$$w_{outcome\_accuracy} = 0.10,\; w_{outcome\_relative} = 0.10,\; w_{odds\_edge} = 0.50,\; w_{info\_adv} = 0.30$$
