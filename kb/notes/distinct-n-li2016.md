# A Diversity-Promoting Objective Function for Neural Conversation Models (origin of distinct-1/2/3)

## (a) Full citation + URL

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, Bill Dolan. **"A Diversity-Promoting Objective Function for Neural Conversation Models."** Proceedings of NAACL-HLT 2016, pp. 110–119, San Diego, California. Also arXiv:1510.03055 [cs.CL] (v2, 7 Jan 2016).

- Abstract: https://arxiv.org/abs/1510.03055
- PDF: https://arxiv.org/pdf/1510.03055v2 (downloaded → `kb/pdf/distinct-n-li2016-1510.03055.pdf`)
- ACL Anthology: https://aclanthology.org/N16-1014/

## (b) Problem addressed
Seq2seq conversational models trained with standard MLE tend to generate **generic, "safe" responses** (e.g. "I don't know", "I'm not sure") because such responses have high likelihood under the training objective regardless of input — the objective doesn't reward *informativeness/diversity* of the response relative to the input. The paper's main contribution is a **Maximum Mutual Information (MMI)** training/decoding objective as a fix; **distinct-1/2/3 are introduced as an *evaluation* metric** to quantify how much this generic-response problem is actually happening (and how much a proposed fix reduces it) — i.e. exactly a **diversity/degeneracy detector**, not a fluency metric per se, but used in the field ever since as a cheap fluency/diversity proxy.

## (c) Exact definition / formula

Verified (via WebFetch of the paper text plus corroborating quotes from three independent sources — the paper's own Section 5.2, and its Table 2 caption paraphrase):

> "We report the degree of diversity by calculating the number of distinct unigrams and bigrams in generated responses. The value is **scaled by the total number of generated tokens** to avoid favoring long sentences, shown as distinct-1 and distinct-2."

Formally, for a set of generated responses (pooled across the whole evaluation corpus, not averaged per-sentence):
$$\text{distinct-}n = \frac{|\{\text{unique } n\text{-grams appearing in all generated responses}\}|}{\text{total number of tokens generated across all responses}}$$

- **distinct-1** = unique-unigram-count / total-token-count
- **distinct-2** = unique-bigram-count / total-token-count
- **distinct-3** (not in the original 2016 paper, but the natural extension universally used in later steering/generation papers, including presumably the target task's own eval) = unique-trigram-count / total-token-count

Key properties for correct implementation:
- **Corpus-level, not per-sentence**: compute over the *entire* pool of generated text for a given condition/coefficient, not per-sample-then-averaged — averaging per-sentence changes the statistic's behavior (a corpus of many short, repetitive sentences looks different under each convention).
- **Normalized by total token count, not by n-gram count** — this is what prevents systems that simply generate longer outputs from trivially scoring higher diversity.
- Higher distinct-n = more lexical diversity = (weak) proxy for "not degenerately repetitive" — but note it says nothing about *grammaticality* or *semantic coherence*, only lexical variety, so it should be paired with perplexity and/or an LLM coherence judge (see `kb/notes/persona-vectors-judge-prompts.md`'s coherence prompt) rather than used alone as "the" fluency metric.

## (d) How it's used as an eval protocol in later steering literature
Distinct-n became a standard secondary fluency/diversity metric (alongside perplexity) in text-generation and later activation-steering papers precisely because perplexity alone can be gamed by repetitive, low-perplexity degenerate text (e.g. "the the the the..."); distinct-n catches that failure mode directly. In the steering literature, it's typically reported as one line in a table alongside perplexity and an LLM-judge concept/fluency score, computed per steering coefficient, to show the point where steering starts producing repetitive/degenerate output even if perplexity hasn't obviously spiked yet.

## Directly reusable techniques / pitfalls
1. **Compute distinct-1/2/3 corpus-level (pooled tokens across all generations at a given alpha), normalized by total token count** — this is the exact, original, unambiguous formula; don't reinvent a per-sentence-averaged variant without checking it changes conclusions.
2. **Always special-case very short generations** — if `max_new_tokens` is small, distinct-n is noisy/inflated (a 5-token generation trivially has distinct-1≈1.0); make sure your generation length is held fixed and reasonably long across all alpha conditions before comparing distinct-n values.
3. **Pair with perplexity, not instead of it** — distinct-n catches repetition/degeneracy, perplexity catches "is this a plausible sentence"; they catch different failure modes and both belong on your fluency axis (or fold them into one composite fluency score for the Pareto x-axis).
4. **Exclude special tokens (BOS/EOS/padding) from both the n-gram set and the token-count denominator** — consistent with how persona_vectors/SAELens code excludes BOS tokens from their metrics (see e.g. SAELens's L0 computation, `feature_acts[:, 1:]`), the same discipline applies here.
