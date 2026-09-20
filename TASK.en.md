# T-Lab 2026. Mechanistic Interpretability — test assignment

*English translation of the original assignment statement; the Russian original is TASK.md.*

- Source: https://edu.tbank.ru/selection/019fb273-85a2-7889-b09f-6aa9ebe44bdb/practice/85022/task/1
- Deadline: August 23, 23:59. 1 task, 100 points. Graded by the instructor.
- Illustration (Pareto front): https://edu.tbank.ru/files/a926a0b4-af64-4f2f-a85e-013b77811afd

## Interpretability

### What this is about

Besides interesting insights and nice pictures, interpretability gives us the ability to control
computations in language models. At the moment the most promising way of doing so is **steering**.
It works as follows: suppose we have found that some direction $v \in \mathbb{R}^d$
(for example, a vector from an SAE decoder) is responsible for a property of the model that we want to
amplify. In that case we perform an intervention of the form

$$\tilde h = h + \alpha v,$$

where $h \in \mathbb{R}^d$ is the model's hidden state and $\alpha \in \mathbb{R}_+$ is the strength
of the vector's application. Usually, for an effect to appear, one uses fairly large $\alpha$, exceeding
the norm of the original $h$. Because of this the model often breaks down (perplexity grows). You can
read about this here: https://huggingface.co/spaces/dlouapre/eiffel-tower-llama

### The task

In this test assignment you will study how to reduce the negative effect of steering.

### How to measure the negative effect

Two axes are usually used for this: on $x$, typically something related to text quality
(fluency score, perplexity…). On the $y$ axis — how strongly the desired property is present in the
generated texts.

By varying $\alpha$ we obtain a Pareto front in these axes. Usually, the more strongly the desired
property is present in the generated texts, the worse the fluency. In the picture above we want the line
to lie as close as possible to the upper right corner.

### How it can be improved

In recent work, GLP [1] proposes training flow matching for activations in order to then carefully
"denoise" them after steering. However, this takes a long time to train and is very expensive at
inference. We suggest that you train a simple model

$$\hat h = \operatorname{denoiser}(h + \varepsilon), \quad \varepsilon \sim N(0, \sigma^2)$$
$$L = \lVert h - \hat h \rVert_2^2$$

After which, at inference time, we do

$$\tilde h = \operatorname{denoiser}(h + \alpha v)$$

For simplicity, let us fix that we want to perform interventions after the middle layer of the LM
(for example, after layer 6 for gpt-2, which has 12 layers).

Be sure to think about various implementation details, for example:

- What is the best way to add noise to $h$? Besides $h + \varepsilon$ you can think about
  $t \cdot h + (1-t) \cdot \varepsilon$, $t \sim U[0;1]$.
- What architecture the denoiser should have.
- How can steering errors be corrected? Perhaps there is something cleverer than simply
  $\operatorname{denoiser}(h + \alpha v)$.
- Is it possible to simply fine-tune the MLPs already present in the model on such a loss, so as not to
  change the model's structure?
- Any other idea for how to improve steering for LMs.

The above is only an example of the fact that a cheap way to do good LLM steering can be built. If you
have other ideas, or they come to you while doing the assignment — be sure to reflect this in the report
and try your own method.

### Plan of action

1. Choose a small LM and the vectors $v$ on which you will validate (in the minimal variant there are
   gpt-2 and an SAE for it [2]).
2. Decide how you will measure fluency/concept score. Ideally it is better to take the pipeline from
   persona vectors, but it requires API access to chatgpt. Pay attention to dist-1, dist-2, dist-3 —
   they are the easiest to compute, and they say something about fluency.
3. Validate your vectors in a simple steering setup: $\tilde h = h + \alpha \cdot v$.
   Make sure you get a picture similar to the example.
4. Train your model. **It is important that it does not know about the existence of any $v$ from validation.**
5. Only after that, check the trained model against ordinary steering.
6. Write a report on the work done.

### Solution format

Put the code and the report (in TeX or markdown) in a repository on GitHub, and also save the best adapter
or checkpoint to a public repository on Huggingface.

### Criteria

What matters is not just dumping this page into GPT and looking at the report at the end; we will strongly
reward those who come up with ideas more interesting than the ones described above and can show that they
work better.

Besides that, one should not forget about analysis of the method. A method that works well is only half of
a paper; the second half is showing why and how it works.

### Useful repositories and links

- https://github.com/TransformerLensOrg/TransformerLens
- https://github.com/decoderesearch/SAELens
- https://github.com/safety-research/persona_vectors

### References

[1] Learning a Generative Meta-Model of LLM Activations. Grace Luo, Jiahai Feng, Trevor Darrell,
Alec Radford, Jacob Steinhardt. https://generative-latent-prior.github.io/

[2] https://github.com/openai/sparse_autoencoder
