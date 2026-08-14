# Frozen Confirmatory Protocol
## Observation, Endogenous Cognition, and Decision Under Uncertainty

### Scientific claim

The integration claim is not that OWGE, EAC, or DQU is a newly invented primitive. It is that they describe distinct functional bottlenecks in an adaptive agent:

1. **OWGE layer:** externally available evidence becomes retained, usable structure under finite resources.
2. **EAC layer:** retained structure can be internally reactivated or composed during intervals without new task-relevant external evidence.
3. **DQU layer:** candidate beliefs/actions are converted into commitments that account for uncertainty, tail risk, inaction, and thought cost.

The confirmatory question is whether a task that genuinely requires all three shows predictable stage-specific degradation when any layer is ablated, and whether cross-stage interactions matter.

### Claim boundary

These studies can reject operational null hypotheses in the implemented environments. They cannot prove a universal theory of intelligence. Positive results justify harder external validation; negative results require revision.

## Study 1

Fresh seeds:

\[
s_i=95001+101i,\qquad i=0,\ldots,59.
\]

Eight matched conditions use three bits: OWGE-like weighted retention, EAC internal composition, and DQU risk-aware commitment.

Primary A1: full `111` > no-OWGE `011`.

Primary A2: full `111` > no-EAC `101`.

Primary A3: full `111` > no-DQU `110`.

Primary A4: full `111` > the maximum of `011`, `101`, and `110` within seed.

Primary A5 tests the interaction

\[
(U_{111}-U_{101})>(U_{011}-U_{001}).
\]

Primary A6 compares adaptive EAC invocation against always-think under an explicit thought cost.

## Study 2

Fresh seeds:

\[
s_i=135001+89i,\qquad i=0,\ldots,59.
\]

B1: unverified recombination reduces real labeled encounters required for a structured future composition.

B2: tagging recombination as unverified reduces early negative transfer when the future relation conflicts with the internally generated candidate.

B3: priming improves early post-encounter learning accuracy across the first ten labeled encounters.

B4: after primed learning, DQU that includes a newly observed hazard context yields higher utility than risk-neutral commitment that uses learned success alone.

The primed and replay agents are identical before future labels arrive. Priming is therefore preparation, not immediate knowledge.

## Statistics

All ten primaries use

\[
H_0:\mu_T\leq\mu_C,\qquad H_A:\mu_T>\mu_C.
\]

Report paired means, difference, 95% CI, one-sided paired t-test, Cohen \(d_z\), Wilcoxon sensitivity, sign test, and wins/ties/losses.

`combine_confirmatory.py` applies ONE Holm correction across all ten primaries.

Confirmatory rejection requires:

\[
p_{\mathrm{Holm,global}}<0.05.
\]

Do not reinterpret opposite-direction effects as new confirmatory hypotheses.

## Required interpretation diagnostics

- If a stage ablation creates almost no policy disagreement, a null is an uninformative saturation result, not evidence that the stage is useless.
- A3 and B4 must report catastrophe exposure separately from mean utility.
- A6 must report thought invocation rate and explicit thought cost.
- Study 2 pre-encounter primed and replay accuracy must remain equal.
- No universal scalar intelligence score is inferred from these tests.
