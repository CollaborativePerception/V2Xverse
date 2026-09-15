# Reviewer Comment → LaTeX Change Mapping

Revision of "Adaptive Control Barrier Functions for Collaborative Autonomous Driving" (IROS 2026 submission 3119, rejected) for ICRA 2027 resubmission. This maps every LaTeX edit made in this revision pass to the specific reviewer comment(s) it addresses.

## 1_Introduction.tex

| Change made | Reviewer comment addressed |
|---|---|
| Added paragraph justifying frozen-adapter design over retraining the planner | R1 Q1: "Why was the upstream planner not retrained with safety objectives given its access to V2X data?" · R8 W1: "Novelty is somewhat incremental" |

## 2_Related_work.tex

| Change made | Reviewer comment addressed |
|---|---|
| Expanded the Berducci et al. [9] comparison into 3 concrete technical differences | R8 W1: "Novelty is somewhat incremental; the main contribution is system integration/application rather than a fundamentally new method" |

## 3_Method.tex

| Change made | Reviewer comment addressed |
|---|---|
| Added η_i-floor guard, new equation for $\bar\eta_i$ | R1 (Flawed Theoretical Formulation): "The direction alignment term eta_i disables the filter for lateral actors... removing longitudinal protection exactly when it is needed" · R1 Q2 |
| Folded actor-acceleration bound $a_i^{\max}$ into the $h_i$ barrier equation | R1 (Flawed Theoretical Formulation): "Actor acceleration is silently ignored, which is highly problematic for modeling pedestrians and cyclists" |
| Added the QP-coefficient derivation paragraph (∂h_i/∂a, dropped term, receding-horizon justification) | R1 (Flawed Theoretical Formulation): "The QP coefficients in Equations 11-13 lack a derivation from the continuous barrier derivative" |
| Inserted numeric values for d_warn (10.0m), d_col (2.5m), τ_ttc (3.0s) | R1 (Reporting Gaps): "Critical cost parameters d_warn, d_col, tau_ttc are never assigned numerical values" · R1 Q3 |

## 4_Setup_Design.tex

| Change made | Reviewer comment addressed |
|---|---|
| Replaced checkpoint-selection sentence with full disclosure (old methodology named, fix described, headline number flagged as pre-fix) | R1 (Experimental Validity): "The evaluation checkpoint was improperly selected using the highest score on the test routes, artificially inflating the reported metrics" · R1 Q5 |
| Added η_floor and a_max rows to Table I | Supports the two Method fixes above |
| Added exact driving-score formula + infraction penalty weights | R1 (Reporting Gaps): "The primary 'CARLA driving score' metric... is not mathematically defined regarding infraction weights" |
| New "Additional Ablations and Robustness Probes" subsection describing every extra experiment | R1: "fixed CBF baseline lacks a sensitivity sweep" · R1: "No ablations exist for major design choices: class-specific vs global γ, observation space, cost function components, or PID vs simple Lagrangian" · R8 W2, W3, W4, Q1, Q3, Q5, Q6 |

## 5_Results.tex

| Change made | Reviewer comment addressed |
|---|---|
| New "Execution Time and Constraint Activation" subsection (solve-time, slack-activation, fallback-rate numbers) | R1: "Execution time is omitted" · R1 Q4 · R1: "frequency of nonzero slack activation... is never reported" · R8 W5, Q4 |
| New "Fixed-Gain Sensitivity Sweep" subsection, table, and figure | R1: "fixed CBF baseline lacks a sensitivity sweep, making the 25.9% improvement claim weak" · R8 W3, Q1 |
| New "Cross-Map Generalization" subsection and table (Town07/Town10) | R1 (Experimental Validity): "Training and evaluation occur entirely on Town05. Generalization is not proven" · R1 Q5 (held-out map) · R8 W2, Q3 |
| New "Ablation Studies" subsection (collaboration ablation, rule-based baseline, class-specific-γ and observation-space proxies) | R1: "the 'collaborative' aspect is not ablated against a single-agent perception baseline" · R1: "No ablations exist for... class-specific vs global γ, observation space" · R8 W4a, W4c, Q5 |
| New "Traffic Density and V2X Robustness" subsection | R8 Q3: "under different traffic densities" · R8 Q6: "robust to noise or latency in collaborative perception or V2X inputs" |

## 6_Conclusion.tex

| Change made | Reviewer comment addressed |
|---|---|
| Full limitations paragraph (forward-invariance not guaranteed, slack numbers, constant-velocity bound, longitudinal-only scope) | R1 (Flawed Theoretical Formulation): "Using a slack penalty... voids the forward-invariance guarantee" · R8 W5: "Safety claims should be phrased carefully" · R8 W6: "only adjusts longitudinal control" |
| Checkpoint re-selection named as top-priority future work | Ties back to R1's checkpoint-selection complaint above |
| Explicitly-declared open items: cost-function-weighting ablation, PID-vs-plain-Lagrangian ablation | R1: "No ablations exist for... cost function components, or PID vs simple Lagrangian" · R8 W4c |

## Not addressed / left open (stated directly in the paper, not silently omitted)

- **Training-seed variability** (R8 Q2) — requires multiple training runs, not completed this cycle.
- **Simulator/upstream-stack diversity** (R8 W2) — only CARLA, only the CoDriving stack; out of scope for this revision.
- **Headline checkpoint regeneration under corrected selection methodology** — code fix implemented, but the reported number predates it; a fresh training run was not completed.
- **Class-specific-vs-shared-γ and cost-function-weighting ablations, PID-vs-plain-Lagrangian ablation** — each requires training a separate policy under the corresponding constraint; training-free proxies were substituted where possible (logged in Results) but are explicitly not presented as equivalent evidence.

## Known open issues (not part of the reviewer-comment mapping, flagged separately)

- Paper text states training takes "800k–1M environment steps," but the actual checkpoint behind every reported number logs 100,360 decision steps internally — discrepancy not resolved, needs checking with Fabian/Qi before finalizing.
- Page count grew from 9 to 11 pages with these additions — may need trimming depending on ICRA 2027's page limit.
- Author line has unresolved placeholders (`Hena Parween xx?`, `Sara Mahjabin xx?`) for affiliation footnotes.
