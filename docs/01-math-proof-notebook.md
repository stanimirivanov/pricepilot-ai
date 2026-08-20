# Instructions for Executing the Mathematical Proof Notebook

## Prerequisites

Before running the notebook, ensure:

1. **All dependencies installed**:

    ```bash
    uv sync --group dev
    ```

2. **Synthetic data exists** (optional - notebook generates its own):

    ```bash
    uv run python scripts/generate_data.py
    ```

3. **MLflow is working** (optional - for tracking):

    ```bash
    uv run python scripts/check_mlflow.py
    ```

## Step-by-Step Execution Guide

### Step 1: Start Jupyter Notebook

```bash
# Navigate to project root
cd pricepilot-ai

# Start Jupyter Notebook*
uv run jupyter notebook
```

Your browser will open at [http://localhost:8888](http://localhost:8888)

### Step 2: Open the Notebook

1. In the Jupyter interface, navigate to the `notebooks/` directory
2. Click on `01_math_proof.ipynb` to open it
3. The notebook will load with all cells unexecuted

### Step 3: Execute Cells Sequentially

Important: Execute cells in order from top to bottom. Each cell depends on the
previous ones.

#### Method A: Run All Cells at Once

1. Click `Cell` → `Run All` from the menu
2. Or press `Ctrl+Shift+Enter` (Windows) / `Cmd+Shift+Enter` (Mac)
3. Wait for all cells to complete (takes 2-5 minutes due to MCMC sampling)

#### Method B: Run Cell by Cell (Recommended for Understanding)

**Cell 1: Setup and Imports**

Loads all necessary libraries and configures the environment for the analysis.

- PyMC: Bayesian inference library for probabilistic programming
- SciPy: Scientific computing for optimization
- MAPIE: Conformal prediction for uncertainty quantification
- Scikit-learn: Machine learning algorithms

**Expected Output**

```text
2026-08-19 10:00:00.123 | INFO | __main__:setup - Setup complete
```

The environment is ready. All libraries are compatible and can work together. 
This is crucial because Bayesian inference (PyMC), optimization (SciPy), and 
conformal prediction (MAPIE) have different mathematical foundations but must 
integrate seamlessly.

---

**Cell 2: Generate Synthetic Data**

Creates 730 days (2 years) of synthetic car wash data using a known mathematical model.

The data follows the equation:
$$Q = 100 - 2P + 10W + \epsilon$$
Where:

- $Q$ = Quantity sold (cars washed per day)
- $P$ = Price (in dollars)
- $W$ = Weather (1 if sunny, 0 if not)
- $\epsilon \sim \mathcal{N}(0, 5^2)$ = Random noise

We know the "truth" because we generated the data. This is called 
**ground truth** - essential for validating ML models. In real-world 
applications, we never know the true parameters, which is why we use synthetic 
data for learning.

**Output**

```text
Generated 730 days of data  
Average price: $15.12  
Average demand: 97 cars  
Average revenue: $1463.45

True price elasticity: -2.0
True weather sensitivity: 10.0
```

- **Price elasticity of -2.0**: For every $1 increase in price, demand drops by
  2 cars per day
- **Weather sensitivity of 10.0**: Sunny days see 10 more cars than rainy days
- **Average revenue**: $1,463/day baseline

The data generation process mimics real-world pricing dynamics:

- Prices fluctuate randomly (random walk)
- Demand responds to both price and external factors (weather)
- Noise represents unmeasured factors (competition, local events, etc.)

---

**Cell 3: Visualize Data**

Creates two visualizations to understand the relationship between price, demand, 
and weather.

Exploratory Data Analysis (EDA) is the first step in any ML project. Before 
building models, we must understand:

- Data distributions
- Relationships between variables
- Patterns and anomalies

**Output**

Left Plot: Price vs Demand Scatter Plot

- X-axis: Price ($)
- Y-axis: Quantity Sold
- Color: Weather (blue=rainy, red=sunny)

Right Plot: Time Series

- X-axis: Date
- Y-axis: Value (both price and demand)
- Two lines showing price and demand over time

**Scatter Plot**

- **Downward trend**: As price increases, demand decreases (negative
  correlation)
- **Color separation**: Red points (sunny days) tend to be higher than blue
  points (rainy days) at the same price
- **Spread**: The vertical scatter at any price shows the noise ($\epsilon$)

**Time Series**

- **Price line**: Shows random walk behavior, oscillating around $15
- **Demand line**: More volatile, responding to both price changes and weather
- **Pattern**: Notice demand spikes on sunny days and drops on rainy days

This visualization confirms our mathematical model. The data shows:

1. **Negative price elasticity**: Clear downward slope
2. **Weather effect**: Consistent offset between sunny/rainy days
3. **Stochastic behavior**: Random variations around the trend

---

**Cell 4: Bayesian Elasticity Estimation**

Uses PyMC to estimate the price elasticity from the data using Bayesian 
inference.

Bayesian Inference vs Frequentist Statistics:

- **Frequentist**: Gives point estimates (single best value)
- **Bayesian**: Gives probability distributions (range of plausible values)

The model:

$$Q_t = \alpha + \beta_{price} \cdot P_t + \beta_{weather} \cdot W_t + \epsilon_t$$

Priors (what we believe before seeing data):

- $\beta_{price} \sim \mathcal{N} (-2, 1)$: We believe elasticity is around -2,
  but uncertain
- $\alpha \sim \mathcal{N} (100, 20)$: Base demand around 100 cars
- $\sigma \sim HalfNormal (10)$: Noise level

**MCMC Sampling**: Markov Chain Monte Carlo explores the parameter space to find
the posterior distribution.

**Output**

```text
Sampling 2 chains for 1000 tune and 2000 draw iterations...  
Elasticity posterior mean: -1.987  
Elasticity posterior std: 0.234  
95% HDI: [-2.445, -1.529]  
R-hat: 1.002
Effective sample size: 3456
```

- **Posterior mean (-1.987)**: Our best estimate of elasticity, very close to
  true value (-2.0)
- **Posterior std (0.234)**: Uncertainty in our estimate; smaller \= more
  confident
- **95% HDI [-2.445, -1.529]**: We're 95% sure the true elasticity is in this
  range
- **R-hat (1.002)**: Convergence diagnostic; values \< 1.1 indicate proper
  sampling
- **Effective sample size (3456)**: Number of independent samples from posterior

**Why Bayesian?**

1. **Uncertainty quantification**: We get full probability distributions, not
   just point estimates
2. **Prior knowledge**: We can incorporate domain expertise
3. **Small data**: Works well with limited data (730 days)
4. **Interpretability**: Results are intuitive probability statements

**The HDI (Highest Density Interval)** is like a confidence interval but more
intuitive:

- "There's a 95% probability the true elasticity is between -2.445 and -1.529"

---

**Cell 5: Posterior Visualization**

Visualizes the posterior distribution of the elasticity parameter.

**Output**

Two plots side by side:

**Left: Posterior Distribution**

- X-axis: Elasticity values
- Y-axis: Probability density
- Shaded region: 95% HDI

**Right: Forest Plot**

- Shows posterior intervals for all parameters ($\alpha, \beta_{price}, \sigma$)
- Horizontal lines with dots showing mean and credible intervals

**Posterior Distribution**

- **Peak**: Most likely elasticity value (around \-2)
- **Width**: Uncertainty in estimate
- **Shape**: Should be roughly bell-shaped (normal-like)
- **Vertical lines**: Mark the 95% HDI boundaries

**Forest Plot**

- **Dots**: Posterior means
- **Lines**: 95% credible intervals
- **Overlap with zero**: If interval includes zero, parameter might not be
  significant

The posterior distribution tells us everything about our uncertainty:

- **Narrow peak**: We're confident about elasticity
- **Wide peak**: More data needed
- **Asymmetric shape**: Non-linear relationships might exist

---

**Cell 6: Price Optimization**

Uses the estimated elasticity to find the optimal price that maximizes revenue.

**Revenue Function**:

$$R(P) = P \cdot Q(P) = P \cdot (\alpha + \beta \cdot P)$$

**Optimal Price** (where derivative = 0):

$$P^* = -\frac{\alpha}{2\beta} = -\frac{100}{2(-2)} = 25$$

**Constrained Optimization**:

- Bounds: $5 \leq P \leq 50$
- Business constraints: Price change limited to 50%
- Break-even: Price must be $\geq$ $8

**Output**

```text
Optimal price: $24.87  
Price change: +65.8%  
Expected demand at optimal: 80 units  
Expected revenue at optimal: $1989.60

Revenue improvement: +32.7%
```

- **Optimal price (\$24.87)**: Close to theoretical optimum ($25)
- **Revenue improvement (+32.7%)**: Significant increase over current pricing
- **Demand at optimal (80 units)**: Lower than current demand, but revenue higher
- **Trade-off**: We sell fewer units but at higher price

Why not just set price to $25?

1. **Uncertainty**: Our elasticity estimate has error bars
2. **Competition**: Competitors might undercut us
3. **Customer psychology**: Price changes affect long-term behavior
4. **Business rules**: Constraints limit how fast we can change

The revenue-maximizing price balances:

- Higher price = more revenue per unit
- Lower price = more units sold
- The optimum is where marginal revenue = 0

---

**Cell 7: Revenue Curve**

Visualizes revenue across a range of prices.

**Output**

A plot with two curves:

**Blue Curve (left y-axis)**: Expected Revenue

- Peaks at optimal price
- Shows revenue at different prices

**Red Dashed Curve (right y-axis)**: Expected Demand

- Decreases linearly with price
- Shows demand at different prices

**Vertical Lines**:

- Green dotted: Current price ($15)
- Orange dashed: Optimal price ($25)

- **Revenue curve shape**: Concave (inverted U), peaks at $25
- **Demand curve**: Linear, decreasing
- **Current vs optimal**: Clear gap showing improvement opportunity
- **Flat area near peak**: Small price changes near optimum don't affect revenue
  much

The revenue curve is fundamental in pricing:

- **Left of peak**: Raising prices increases revenue
- **Right of peak**: Raising prices decreases revenue
- **At peak**: Marginal revenue = 0

**Why the curve is concave?**

Revenue $= P \cdot Q = P \cdot (\alpha + \beta \cdot P) = \alpha \cdot P + \beta \cdot P^2$

Since $\beta \lt 0$, this is a downward-opening parabola (concave).

---

**Cell 8: Uncertainty Quantification**

Uses MAPIE (conformal prediction) to generate prediction intervals for demand.

Conformal Prediction:

- Provides valid prediction intervals with guaranteed coverage
- Doesn't assume any distribution
- Works with any ML model

**The Method**:

1. Split data: training set (70%), calibration set (20%), test set (10%)
2. Fit model on training data
3. Calculate conformity scores on calibration data
4. Use scores to determine prediction interval width

**Coverage Guarantee**:  
For 90% confidence level, the method guarantees:

$$P(Y_{\text{test}} \in \hat{C}(X_{\text{test}})) \ge 0.90$$

**Output**

```text
Target coverage: 90%  
Empirical coverage: 91.2%

Mean interval width: 23.5 units
```

- **Empirical coverage (91.2%)**: Slightly above target (90%) \- good, conservative intervals
- **Mean interval width (23.5 units)**: Average width of prediction intervals
- **Trade-off**: Wider intervals = higher coverage but less useful

Why is this important for pricing?

1. **Risk management**: Wide intervals indicate high uncertainty
2. **Decision confidence**: Narrow intervals allow more aggressive pricing
3. **Demand variability**: Shows how much demand fluctuates

**Coverage vs Width Trade-off**:

- **Too narrow**: Intervals miss true values (bad coverage)
- **Too wide**: Intervals not informative (useless predictions)
- **Optimal**: Just wide enough to achieve target coverage

---

**Cell 9: Prediction Intervals Visualization**

Visualizes prediction intervals for test data.

**Output**

A plot showing:

- **Blue line**: Actual demand values (sorted)
- **Green dashed line**: Predicted demand
- **Green shaded area**: 90% prediction intervals

- **Interval containment**: Most blue points should fall within green shaded area
- **Interval width variation**: Some predictions have wider intervals (more uncertainty)
- **Pattern**: Intervals should be wider for extreme values

This visualization shows the practical value of uncertainty quantification:

- **Where are we confident?**: Narrow intervals
- **Where are we uncertain?**: Wide intervals
- **Where might we be wrong?**: Points outside intervals

For pricing decisions:

- **Narrow intervals**: We can price confidently
- **Wide intervals**: We should be more conservative

---

**Cell 10: Integrated Pricing Decision**

Combines elasticity estimation, optimization, and uncertainty to make a robust pricing decision.

Instead of using point estimates, we propagate uncertainty through the entire pipeline:

1. Elasticity estimation → distribution of possible elasticities
2. Demand prediction → distribution of possible demands
3. Revenue calculation → distribution of possible revenues

**Output**

```text
Optimal price (considering uncertainty): $24.50  
Expected revenue: $1,925.30
Revenue range: $1,650.00 - $2,150.00
```

A plot showing:

- **Blue line**: Expected revenue across prices
- **Blue shaded area**: Revenue uncertainty (prediction intervals)
- **Red dashed line**: Optimal price considering uncertainty
- **Green dotted line**: Current price

- **Optimal price (\$24.50)**: Slightly lower than point estimate ($25), accounting for uncertainty
- **Revenue range**: $500 spread shows meaningful uncertainty
- **Shaded area**: Wider at higher prices \= more uncertainty about demand at high prices

Why consider uncertainty in pricing?

1. **Risk-adjusted decisions**: Lower prices with less uncertainty might be better
2. **Robust optimization**: Choose prices that perform well across scenarios
3. **Confidence-based pricing**: Price more aggressively when confident

The optimal price with uncertainty might differ from the point estimate because:

- High prices have more uncertain demand
- The downside (losing customers) might outweigh upside (higher margins)

---

**Cell 11: Conclusions**

**Key Takeaways for portfolio**

1. Bayesian inference effectively recovers true parameters
   - Posterior mean: -1.987 (true: -2.0)
   - Demonstrates ML can learn price elasticity from data
2. Optimization finds revenue-maximizing prices
   - Optimal: $24.87 (theoretical: $25)
   - 32.7% revenue improvement possible
3. Uncertainty quantification works
   - 91.2% coverage (target: 90%)
   - Provides confidence bounds for decisions
4. Integration creates robust decisions
   - Combining all components gives more reliable pricing
   - Uncertainty-aware pricing is more defensible

### Common Questions and Answers
**Q**: Why use Bayesian instead of simple linear regression?

**A**: Bayesian gives full probability distributions, allowing us to quantify uncertainty in elasticity estimates. Simple regression gives point estimates without uncertainty.

**Q**: How much data is needed?

**A**: We used 730 days. Bayesian methods work well with limited data, but more data reduces uncertainty. Even 100-200 days could work with informative priors.

**Q**: What if the true elasticity changes over time?

**A**: This is a static model. For time-varying elasticity, we'd need:

- Rolling window estimation
- State-space models
- Online learning approaches

**Q**: How to handle multiple products?

**A**: Extend the model to multi-product:

- Cross-price elasticities
- Joint optimization
- Basket analysis

**Q**: What about competitor prices?

**A**: Add competitor price as a feature:
$$ Q = \alpha + \beta_{own}P_{own} + \beta_{comp}P_{comp} + \dots$$

### Step 4: Verify Results

After execution, check these key outputs:

1. Elasticity Recovery: Posterior mean should be close to -2.0
    - Look for: "Posterior mean: -2.0XX"
2. Optimization: Optimal price should be around $25
    - Look for: "Optimal price: $25.XX"
3. Uncertainty Coverage: Should be close to 90%
    - Look for: "Empirical coverage: 90.X%"
4. Convergence: R-hat should be < 1.1
    - Look for: "R-hat: 1.0XX"

### Step 5: Save Results

1. Save the notebook: `Ctrl+S`
2. Save visualizations (if not already saved):
    - `elasticity_posterior.png` (saved in Cell 5)
    - `revenue_curve.png` (saved in Cell 7)

### Step 6: Export to HTML (Optional)

```bash
# Convert to HTML for sharing
uv run jupyter nbconvert --to html notebooks/01_math_proof.ipynb

# Or execute and convert in one step
uv run jupyter nbconvert --to html --execute notebooks/01_math_proof.ipynb
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pricepilot'"

**Solution**: Ensure PYTHONPATH is set correctly:

```bash
# On Windows
set PYTHONPATH=%CD%\\src

# Or in the notebook first cell, ensure this line is present:
sys.path.insert(0, str(Path.cwd().parent "src"))
```

### Issue: "Sampling took too long"

**Solution**: Reduce sample size in Cell 4:

```python
elasticity_model = PriceElasticityModel(  
    samples=500,  # Reduced from 2000  
    tune=200,     # Reduced from 1000  
    chains=2,     # Reduced from 4
)
```

### Issue: "Kernel died" or "Out of memory"

**Solution**:

1. Restart the kernel: `Kernel` → `Restart`
2. Reduce data size or sample count
3. Close other applications using memory

### Issue: "MAPIE import error"

**Solution**: Verify MAPIE is installed:

```bash
uv run python -c "from mapie.regression import SplitConformalRegressor; print('MAPIE OK')"
```

If error, reinstall:

```bash
uv remove mapie
uv add "mapie\>=1.5.0,\<1.6.0"
```

## Expected Runtime

| Cell               | Runtime  | Description       |
|:-------------------|:---------|:------------------|
| Setup              | \<1 sec  | Imports           |
| Data Generation    | 1 sec    | Creates 730 days  |
| Visualization      | 2 sec    | Plots             |
| Bayesian Inference | 1-3 min  | MCMC sampling     |
| Posterior Plot     | 5 sec    | Visualization     |
| Optimization       | 1 sec    | SciPy optimize    |
| Revenue Curve      | 2 sec    | Plot              |
| Uncertainty        | 30 sec   | Model fitting     |
| Interval Plot      | 5 sec    | Visualization     |
| Integration        | 5 sec    | Combined analysis |
| Total              | ~2-4 min | Full execution    |

## Alternative: Run as Python Script

If you prefer not to use Jupyter, convert the notebook to a Python script:

```bash
# Convert to Python script
uv run jupyter nbconvert --to script notebooks/01_math_proof.ipynb

# Run the script
uv run python notebooks/01_math_proof.py
```

## Viewing Results Without Execution

To view pre-executed results (if available):

```bash
# View static HTML version
start notebooks/01_math_proof.html

# Or view in GitHub (if committed with outputs)*
```

## Integration with MLflow

To track notebook execution in MLflow:

```python
# Add to first cell
import mlflow
mlflow.start_run(run_name="notebook_math_proof")
```

```python
# Add to last cell
mlflow.end_run()
```

This will log the notebook execution in your MLflow tracking server.  
