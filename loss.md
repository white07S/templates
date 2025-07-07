Here’s a practical, quantitative play-book for turning “qualitative” likelihood scores into data-driven probability estimates for inherent risk (IR):

⸻

1. Gather and Structure Your Event Data
	1.	Define your event taxonomy
– What counts as a “loss event,” “near-miss” or “control failure”?
– Group by process, business line, risk category, severity band.
	2.	Collect a time series of counts
– For each bucket (e.g. “trade-capture errors in Europe”), assemble Nᵢ = number of loss events over total exposure time Tᵢ (e.g. in years).
– Ideally include both internal data and external/industry benchmarks (e.g. ORX).

⸻

2. Fit a Frequency Distribution

A. Poisson Model (if events roughly independent & low volume)
	•	

\hat\lambda_i = \frac{N_i}{T_i}
	•	Probability of ≥1 event in next period of length Δ:
P(\text{≥1 in Δ}) = 1 - e^{-\,\hat\lambda_i\,Δ}

B. Negative-Binomial (if you see over-dispersion)
	•	Fit NB parameters (r,p) so \mathbb{E}[N]=r\frac{1-p}{p}, \mathrm{Var}[N]=r\frac{1-p}{p^2}.
	•	Then compute
P(N\ge1)=1-\bigl(p/(1-(1-p)e^{-\mu\,Δ})\bigr)^r
(where \mu=\hat\lambda_i).

⸻

3. Convert to a “Likelihood Score”

Once you have P_i = P(\text{≥1 event in horizon }Δ), map it back onto your 1–5 scale:

Probability range	Likelihood score
P<0.10	1 (Rare)
0.10\le P<0.30	2 (Unlikely)
0.30\le P<0.60	3 (Possible)
0.60\le P<0.85	4 (Likely)
P\ge0.85	5 (Almost certain)

You can adjust cut-points to your bank’s risk appetite thresholds.

⸻

4. Bayesian Updating for Sparse Data

When you have few historical events:
	1.	Choose a prior for \lambda, e.g. \text{Gamma}(\alpha_0,\beta_0) reflecting expert view.
	2.	Update with data N_i over exposure T_i:
	•	Posterior \lambda\sim\Gamma(\alpha_0+N_i,\,\beta_0+T_i).
	3.	Use posterior mean
\displaystyle \hat\lambda_i^{\rm post}=\frac{\alpha_0+N_i}{\beta_0+T_i}
in your Poisson formula above.
	4.	Quantify uncertainty via credible intervals—e.g. take 90% lower bound for a conservative likelihood.

⸻

5. Incorporate Leading Indicators

Beyond pure loss counts, build a logistic‐regression (or tree‐based) model:

\Pr(\text{event in next }Δ \mid X) = \sigma\bigl(\beta_0 + \beta_1\,\text{KRI}_1 + \cdots + \beta_k\,\text{KRI}_k\bigr)
	•	Features X could include:
	•	Number of control failures in last quarter
	•	Vendor downtime hours
	•	System-alert volumes
	•	External risk scores (e.g. cyber-threat index)
	•	Train on historical periods; validate out-of-sample calibration so predicted probabilities match realized frequencies.

⸻

6. Scenario-Based Stress Adjustments

To make it forward-looking under stress:
	1.	Define stress multipliers m for λ under adverse scenarios (e.g. “20% higher fraud attempts”).
	2.	Compute \lambda_{\rm stressed} = m\times\hat\lambda_i.
	3.	Re-compute P(\ge1) and re-map to your likelihood score.

⸻

7. Embedding in Your IR Workflow
	1.	Automate: have your data pipeline recompute \hat\lambda monthly or quarterly.
	2.	Dashboard: show real-time P(\ge1) alongside qualitative controls assessments.
	3.	Governance: if P breaches an appetite threshold (e.g. 0.6), trigger a deep-dive or control enhancement.

⸻

In summary, you can turn “likelihood = 4” from a gut call into “we estimate a 72% chance of at least one loss event in the coming year,” by:
	1.	Fitting a frequency model (Poisson/NB) to your event counts
	2.	(Optionally) Bayesian‐updating for sparse data
	3.	Augmenting with predictive analytics on KRIs
	4.	Stress-testing forward scenarios

This not only quantifies your IR likelihood but also makes it fully data-driven and repeatable.
