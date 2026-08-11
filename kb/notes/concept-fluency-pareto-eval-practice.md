# Standard Practice: Measuring Concept Presence vs. Fluency for (SAE) Feature/Activation Steering

This note collects how recent steering-evaluation papers construct the exact Pareto-front axes the target task needs (x = fluency, y = concept presence), as a cross-check/complement to the GLP paper's own protocol (see `kb/notes/glp-generative-latent-prior.md`, which is the most directly relevant single source and already documents its 0–2 and 0–100 LLM-judge scales).

## 1. SAEs Are Good for Steering — If You Select the Right Features

- **Citation**: arXiv:2505.20063, "SAEs Are Good for Steering -- If You Select the Right Features."
- URL: https://arxiv.org/abs/2505.20063 · PDF downloaded → `kb/pdf/saes-good-for-steering-2505.20063.pdf`
- **Relevant practice**: distinguishes **input features** (capture patterns in what activates them) from **output features** (have a measurable causal effect on generation) and proposes input/output *scores* to tell them apart before choosing which SAE feature to steer with at all — a useful pre-filtering step upstream of any Pareto-front eval: don't waste your alpha sweep on a feature that never had a real causal effect to begin with. Reports 2–3x steering-effectiveness improvement just from this filtering, making unsupervised SAE-feature steering competitive with supervised (e.g. persona-vector-style) direction extraction.
- Evaluation setup (per secondary sources, `[UNVERIFIED]` against full paper text — not deep-read): a "Concept500"-style benchmark of concept/feature pairs, with separate **concept score** (did the target concept appear) and **fluency score** (is the text coherent), each via LLM judge, plus an instruction-following score — structurally the same two-axis (concept, fluency) design as everything else in this KB.

## 2. Dynamically Scaled Activation Steering (DSAS)

- Found via search; exact arXiv id not independently re-verified in this pass — treat citation details as `[UNVERIFIED]`, re-search "Dynamically Scaled Activation Steering" if you need the precise reference.
- **Relevant practice**: instead of a fixed alpha, **dynamically scale the steering coefficient per-token/per-context**. Reported result: "for any given toxicity level, DSAS achieves higher MMLU and lower perplexity than unconditional (fixed-alpha) steering" — i.e. dynamic scaling **strictly improves the Pareto front** versus a fixed-alpha sweep. Directly actionable: if your denoiser-based approach still shows a binding fluency/concept tradeoff, consider making alpha itself a function of the current activation (e.g. scaled down when the raw activation is already far from typical, scaled up when it's central) as a complementary axis of improvement, on top of the denoiser.

## 3. Feature Guided Activation Additions (FGAA)

- **Citation**: arXiv:2501.09929, "Interpretable Steering of Large Language Models with Feature Guided Activation Additions."
- PDF downloaded → `kb/pdf/feature-guided-activation-additions-2501.09929.pdf`
- **Relevant practice**: reports perplexity-vs-steering-scale curves showing SAE-direct-feature steering is "notably aggressive" at scale 0–40, with **a shared inflection point around scale 40** across methods, beyond which capability/fluency drops sharply. Practical implication already noted in `kb/notes/norm-preserving-steering-fluency.md`: sample your alpha grid densely near the expected inflection region.

## 4. GLP's own Pareto-front protocol (cross-reference, fullest primary-source detail already captured)
See `kb/notes/glp-generative-latent-prior.md`, section (d), items 3 and the "Steering experiment matrix" table — it is the single most load-bearing reference in this KB for exactly this deliverable: LLM-judge on a 0–2 scale (for SAE-feature and DiffMean/sentiment steering) or a 0–100 scale (persona elicitation, reusing the persona_vectors judge convention), plotted as (fluency, concept) points swept over the steering coefficient, with the explicit claim/visualization that a good denoiser "expands the Pareto frontier outward."

## Synthesized recommendation for the target task's own eval protocol
Combine, rather than pick one:
- **x-axis (fluency)**: perplexity (GPT-2-small's own loss, or an external reference LM's loss on the generations) **+** distinct-1/2/3 (`kb/notes/distinct-n-li2016.md`) **+** optionally an LLM coherence judge (verbatim reusable prompt in `kb/notes/persona-vectors-judge-prompts.md`) — three complementary signals, since perplexity alone misses repetition, distinct-n alone misses ungrammaticality, and an LLM judge is the most holistic but costs API calls.
- **y-axis (concept presence)**: an LLM-judge trait/concept-expression score (0–100, logprob-weighted-average scoring — see `persona-vectors.md` §(d)) or, if you have a SAE feature with a Neuronpedia description, "does the generation match the feature's description" (GLP's approach) — or a lightweight classifier (e.g. a sentiment/topic classifier) if a suitable one exists for your concept, which is cheaper than an LLM judge and was GLP's approach for sentiment (SetFit classifier).
- **Sweep**: parameterize alpha relative to the empirical average activation norm at your steering layer ($\alpha = r\cdot\lVert\bar h\rVert_2$, per GLP), not as a raw absolute number, and densify sampling near the expected fluency-collapse inflection rather than sampling uniformly.
- **Baseline curve**: raw steering (no denoiser) alpha-sweep is your reference Pareto front; report your denoiser's curve as "expanding the frontier outward" the same way GLP's Figures 5–6 do, and consider also reporting the training-free covariance-projection baseline from `kb/notes/norm-preserving-steering-fluency.md` item 2 as a second reference point.
