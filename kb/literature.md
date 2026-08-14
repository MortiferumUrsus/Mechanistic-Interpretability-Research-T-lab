# Литобзор: денойзинг активаций против разрушительного эффекта стиринга

Дата составления: 2026-08-14. Все arxiv id проверены через веб-поиск/fetch; там, где формулу или число не удалось подтвердить первоисточником — помечено явно.

---

## 1. Метрики оценки стиринга

### 1.1 Ось fluency/coherence

**Perplexity под judge-моделью.**
Общий паттерн в литературе — не мерить perplexity той же моделью, которую стирят (иначе давление стиринга на logits самой модели искажает метрику), а либо (а) использовать отдельную judge-LLM с промптом на когерентность (0–100), либо (б) считать conditional log-likelihood/perplexity под independent LM.
- ActAdd (Turner et al., arXiv:2308.10248, GPT-2-XL): считают **perplexity ratio** — отношение perplexity степированных generations к перплексии несте­пированных на wedding-unrelated предложениях; в их эксперименте ratio ≈ 0.994 (деградация минимальна). Также используют ConceptNet **P@K** (probability that expected label среди top-K предсказанных токенов) как метрику сохранения off-target знаний.
- Persona Vectors (safety-research/persona_vectors, arXiv:2507.21509): **coherence score** — отдельный проход judge-модели **GPT-4.1-mini**, шкала 0–100, оценивает связность транскрипта (промпт-темплейт в исходники не зашит текстом на странице abstract/html, найти дословно не удалось — см. п. 1.4).
- GLP (arXiv:2602.06964): основная fluency-метрика — **conditional negative log-likelihood под той же LLM** (т.е. self-perplexity базовой модели на стирированных активациях) плюс производная величина **Delta LM Loss** = прирост LM-лосса (perplexity) степированной генерации относительно нестепированной baseline. Не использует внешнего judge для fluency, но подчёркивает, что рост лосса — прокси именно "схода с распределения".
- Self-scoring bias: отдельные работы про LLM-as-judge (arXiv:2410.21819, "Self-Preference Bias in LLM-as-a-Judge") фиксируют, что judge-модели систематически завышают оценку текстов с низкой perplexity и текстов, сгенерированных ими же — довод в пользу того, чтобы judge и стирируемая модель были разными весами/семействами.

**Distinct-n (dist-1/dist-2/dist-3).**
Источник: Li et al., 2016, "A Diversity-Promoting Objective Function for Neural Conversation Models" (NAACL, arXiv:1510.03055).
Точная формула:
$$\text{Distinct-}n = \frac{|\{\text{уникальные } n\text{-граммы в тексте}\}|}{N_{\text{total}}}$$
где $N_{\text{total}}$ — общее число сгенерированных n-грамм (то есть общее число токенов минус $n-1$, приблизительно равное общему числу слов). Нормировка — на суммарное количество сгенерированных токенов/слов, а не на длину словаря. Штрафует повторы; не требует референса.
Известная проблема метрики (и её улучшение): arXiv:2202.13587 "Rethinking and Refining the Distinct Metric" (ACL 2022) — показывает смещённость классического distinct-n по длине текста и предлагает исправленную версию (expectation-adjusted distinct, EAD). Стоит процитировать при обосновании выбора метрики.
Комбинированная метрика: **Fluency-Diversity Rate (FDR) = (Dist-2)² / ln(PPL)** — встречается в литературе по контролируемой генерации как единый скаляр, объединяющий perplexity и distinct-2 (источник конкретной статьи не идентифицирован надёжно при поиске — арксив id не найден, использовать с осторожностью / доперепроверить перед цитированием).

### 1.2 Ось concept strength / behavioral score

Без LLM-judge встречаются четыре подхода, все зафиксированы в найденных источниках:
1. **Частота целевых токенов / top-K вероятность целевого слова** — как в ActAdd: P@K на ConceptNet, вероятность появления целевого слова в completion.
2. **Косинус с вектором стиринга** — измеряют cosine similarity между активацией ответа (response-side proxy) и направлением стиринга/документа; используется и как ретривал-метрика, и как классификатор "содержит ли активация концепт" (бинарный классификатор через порог по cosine).
3. **Классификатор** — например, finetuned sentiment classifier (5-точечная шкала) в GLP для sentiment-стиринга — независимая модель-классификатор поверх сгенерированного текста, а не LLM-judge.
4. **Log-prob целевых слов** — conditional log-likelihood целевых токенов под GPT-3 davinci-002 logprobs упоминается как метод (общий паттерн, не привязан к одной статье с точным id).

С LLM-judge:
- GLP: sentiment scored судьёй по шкале **0–2** (упрощённая шкала для sentiment), и отдельно — five-point sentiment classifier как non-LLM альтернатива. Персона-концепты — по шкале **0–100**.
- SAE-TS (arXiv:2411.02193): **Behavioral score** (1–10) и **Coherence score** (1–10) от **GPT-4o-mini**, на 256 сэмплах по 32 токена; итоговая метрика — **normalized product Behavioral × Coherence** (0–1).
- CAA (Rimsky et al., arXiv:2312.06681): GPT-4 оценивает по шкале **1–10**, "насколько сильно проявлено целевое поведение".
- Persona Vectors: **trait expression score** 0–100 от GPT-4.1-mini (0 = нет проявления, 100 = сильное проявление), отдельно coherence score 0–100 тем же judge.

### 1.3 Формализация "схода с многообразия" (off-distribution damage)

- **Delta LM Loss / conditional NLL** (GLP, arXiv:2602.06964): прирост лосса модели на стирированной активации — их основная прокси-метрика "насколько активация ушла с многообразия", коррелирует с subjective fluency.
- **KL-дивергенция к базовому (нестепированному) распределению выходов**:
  - KL-then-Steer, arXiv:2406.15518: обучают модель минимизировать KL между steered и unsteered output distributions на безобидных входах **до** применения стиринга; лучший метод предотвращает 44% jailbreak-атак на Llama-2-7B-chat при почти неизменном MT-Bench.
  - "Could Inference-Time Interventions Preserve Alignment? Safety Cost of Steering Vectors Is Separable and Reducible" (arXiv:2608.08383): вводят **refusal-token divergence (D_R)** — частичный KL, посчитанный только по подмножеству токенов-индикаторов отказа ("I", "cannot" и т.п.), а не по всему словарю; плюс ASR (attack success rate) и FRR (false refusal rate) как метрики "цены" стиринга. Метод **CAST** учит rank-1 ablation-направление $\hat r$ и убирает его компоненту из вектора стиринга: $v^* \leftarrow v - \hat r \hat r^\top v$, решая constrained-оптимизацию (primal-dual) по трём ограничениям (safety, эффект поведения, false-refusal). На Qwen-7B стандартный стиринг поднимает mean ASR с 24.5% до 35.0% (worst-case >63%); конкретные числа KL=2.088→0.044, которые всплыли в предварительном поиске, в самом тексте статьи **не подтверждены** — не использовать без повторной проверки.
- **Mahalanobis distance к распределению активаций как OOD-метрика** — прямых работ, соединяющих Mahalanobis-OOD-детекцию именно со стирингом активаций, найти не удалось; есть только общий пласт литературы про Mahalanobis-based OOD detection в другом контексте (image OOD, напр. arXiv:2505.18032 Mahalanobis++, arXiv:2605.14413 MahaVar) — **не про activation steering**, использовать только как источник формулы Mahalanobis distance, не как прецедент применения к стирингу.
- **Reconstruction error как метрика** — напрямую не найдено отдельной работы, где reconstruction error автоэнкодера использовался бы как метрика "степени схода с многообразия" при стиринге (кроме GLP, где denoising loss косвенно играет эту роль).

### 1.4 Пайплайн safety-research/persona_vectors

Репозиторий: https://github.com/safety-research/persona_vectors (код к arXiv:2507.21509).
- Judge — **gpt-4.1-mini-2025-04-14**, оценивает (а) trait expression score 0–100, (б) coherence score 0–100. Дословный текст промпт-темплейта извлечь через доступные веб-инструменты не удалось (README не публикует его целиком, прямой путь к judge.py на raw.githubusercontent вернул 404 — файл, вероятно, лежит в другой директории репозитория, не проверено).
- Метод извлечения вектора персоны — автоматический, по natural-language описанию черты характера (не detали формулы извлечения в открытом доступе на этой странице).
- Формула стиринга: $h_\ell \leftarrow h_\ell + \alpha \cdot v_\ell$ (усиление черты), $h_\ell \leftarrow h_\ell - \alpha \cdot v_\ell$ (подавление), $v_\ell$ — персона-вектор на слое $\ell$.
- **Projection difference** ΔP — метрика для предсказания, насколько данные для дообучения сдвинут модель по направлению персоны, **до** самого файнтюнинга:
$$\Delta P = \frac{1}{|D|}\sum_{i} \big[a_\ell(x_i,y_i) - a_\ell(x_i,y_i')\big]\cdot \hat v_\ell$$
где $a_\ell(x_i,y_i)$ — активация на реальном тренировочном ответе, $a_\ell(x_i,y_i')$ — активация на ответе базовой модели, $\hat v_\ell$ — нормированный персона-вектор. Большая ΔP ⇒ датасет вероятно сдвинет персону модели при дообучении — используется как proactive-фильтр тренировочных данных.
- Также есть скрипт `cal_projection.sh` с параметром `--projection_type` (несколько вариантов проекции) — детали не раскрыты в README.

---

## 2. Способы снизить разрушительность стиринга (baseline'ы)

### 2.1 Norm-preserving / norm-matched steering

Точной формулы вида $\tilde h = \|h\|\cdot(h+\alpha v)/\|h+\alpha v\|$ как отдельного канонического "имени метода" в одном источнике не нашлось, но она эквивалентна **Renormalized CAA (CAA-r)**, формализованной в "A Geometric Account of Activation Steering through Angle–Norm Decomposition" (arXiv:2606.06735):
$$y = r\cdot\frac{x+\alpha s}{\|x+\alpha s\|}, \quad r=\|x\|$$
Та же статья вводит более общее разложение на угол и норму:
- Обычный CAA: $y = x+\alpha s$ (не norm-preserving, не per-token).
- **CAA-m (matched)**: $y=x+\alpha s$ с $\alpha$, подобранным так, чтобы $\langle y/\|y\|, s\rangle=\gamma$ (целевой угловой score); закрытая форма из Appendix C: $\alpha = r\left(\dfrac{\gamma\sqrt{1-c^2}}{\sqrt{1-\gamma^2}} - c\right)$, где $c=\langle x/\|x\|, s\rangle$.
- **Spherical steering (S)**: $y = r(\gamma s + \sqrt{1-\gamma^2}\,v)$ — строго сохраняет норму $r$, поворачивает активацию к концепт-направлению $s$ на угол, соответствующий $\gamma$; $v$ — компонента, ортогональная $s$.
- **Spherical + Norm scaling (SN)**: $y=\beta r(\gamma s+\sqrt{1-\gamma^2}v)$ — добавляет мультипликативный параметр нормы $\beta$.
- Важный вывод статьи: строгое сохранение нормы не всегда оптимально при сильном стиринге — авторы показывают, что при $\gamma=0.7$ увеличение $\beta$ с 1.0 до 1.2 улучшает perplexity примерно в 1.8×, т.е. стиринг лучше рассматривать как двухпараметрическую (угол + радиус) интервенцию, а не одномерную.

Родственная работа: **Selective Steering** (arXiv:2601.19375) — "mathematically rigorous norm-preserving rotation formulation"; **Minimizing Collateral Damage in Activation Steering** (arXiv:2605.01167) — обзорно вводит класс Norm-Preserving methods, включая **Slerp** (сферическую линейную интерполяцию) как альтернативу прямому сложению.

### 2.2 Projection-based ablation vs addition (directional ablation, feature clamping)

**Directional ablation** — Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717):
- Направление через diff-in-means: $r^{(l)} = \mu_{\text{harmful}}^{(l)} - \mu_{\text{harmless}}^{(l)}$.
- Ablation (полностью убирает компоненту направления из **каждого** слоя и **каждой** токен-позиции):
$$x' \leftarrow x - \hat r\hat r^\top x$$
- Activation addition (индукция поведения, добавление на одном слое $l$ по всем позициям):
$$x^{(l)\prime} \leftarrow x^{(l)} + r^{(l)}$$
- Проверка сохранности способностей: LM Evaluation Harness (Open LLM Leaderboard протокол) — MMLU меняется в пределах ±1.5 п.п., ARC/GSM8K — ±0.8 п.п., TruthfulQA — падение на 1–3.5 п.п. для ортогонализованных моделей.

**Feature clamping (SAE-based)**:
- "Steering Language Model Refusal with Sparse Autoencoders" (arXiv:2411.11296) — клампинг значения конкретной SAE-фичи до константы (выше — усиление, ниже — подавление поведения); значение клампа — гиперпараметр.
- "Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning" (arXiv:2503.11127) — вводит **conditional clamping** (`clamp_cond`): в отличие от обычного клампа (сравнение с нулём), сравнивает активацию фичи с задаваемым `clamp_value`; клампинг к фиксированному отрицательному значению даёт unlearning с меньшими побочными эффектами, чем масштабирование, особенно при работе с несколькими фичами одновременно.

### 2.3 CAA (Rimsky et al.)

arXiv:2312.06681, "Steering Llama 2 via Contrastive Activation Addition".
Формула вектора (mean difference по датасету троек prompt/positive/negative):
$$v_{MD} = \frac{1}{|D|}\sum_{(p,c_p,c_n)\in D}\big[a_L(p,c_p) - a_L(p,c_n)\big]$$
Применение: $h \leftarrow h + \text{multiplier}\times v_{MD}$, добавляется на всех токен-позициях после промпта пользователя, на одном выбранном слое. Оптимальный слой зависит от модели: Llama-2-7B-chat — слой 13, 13B-вариант — слои 14–15 (эффект пиковый примерно на одинаковой относительной глубине для разных типов поведения).
Оценка: GPT-4, шкала 1–10 "насколько сильно проявлено целевое поведение" + open-ended генерация и multiple-choice поведенческие датасеты.

### 2.4 ActAdd (Turner et al.)

arXiv:2308.10248, "Activation Addition: Steering Language Models Without Optimization" (изначально сентябрь 2023, обновлялась).
Формула:
$$h_A^l = h_+^l - h_-^l,\qquad h'^l = h^l + c\cdot h_A^l$$
где $h_+^l, h_-^l$ — активации на слое $l$ от контрастных промптов (напр. "Love" vs "Hate"). Достаточно **одной** пары промптов — без оптимизации, без градиентов. Коэффициент $c$ обычно по модулю < 15. Оптимальный слой — средние слои сети; для wedding-вектора на GPT-2-XL пик эффекта на слое 6, успех топик-стиринга >90% против ~2% базовой линии.
Off-target preservation: ConceptNet P@K (вероятность правильного токена в top-K) — деградация незначительна; perplexity ratio на несвязанных предложениях ≈ 0.994.

### 2.5 SAE-Targeted Steering (Chalnev, Siu, Conmy)

arXiv:2411.02193, "Improving Steering Vectors by Targeting Sparse Autoencoder Features". Код: github.com/slavachalnev/SAE-TS.
Метод: обучают линейный аппроксиматор эффекта произвольного вектора стиринга на все SAE-фичи:
$$\hat y = xM+b$$
$x\in\mathbb R^{d_{model}}$ — вектор стиринга, $M\in\mathbb R^{d_{model}\times d_{sae}}$, минимизация MSE между предсказанным $\hat y$ и наблюдаемым эффектом $y$ на 50 000 обучающих примеров.
Целевой вектор, максимизирующий эффект на фиче $j$ при минимизации побочных эффектов:
$$s = \frac{M_j}{\|M_j\|} - \lambda\frac{Mb}{\|Mb\|},\quad \lambda=1$$
(далее нормируется к единичной норме). Масштаб $\alpha$ подбирается так, чтобы cross-entropy loss стирированной модели вырос на фиксированные +0.5 над baseline (единый бюджет "порчи" модели для честного сравнения методов).
Метрики: Behavioral score (1–10) и Coherence score (1–10) от GPT-4o-mini на 256 сэмплах по 32 токена; итоговый показатель — normalized Behavioral×Coherence.
Результаты на Gemma-2-2B (итоговая метрика, среднее по 9 задач): CAA 0.217, raw SAE feature steering 0.129, **SAE-TS 0.360** (SAE-TS лучше на 7 из 9 задач; отдельные задачи — London: CAA 0.048 / SAE 0.006 / SAE-TS 0.538; Wedding: 0.177 / 0.263 / 0.543; Love: 0.304 / 0.100 / 0.423).

### 2.6 Conditional / clamped SAE steering

См. п. 2.2 — arXiv:2503.11127 (conditional clamping для unlearning) как основной источник; отдельной статьи именно про "conditional clamped SAE steering" для baseline-сравнения fluency/concept trade-off (аналогичной SAE-TS) не найдено.

### 2.7 Формализация "стиринг ломает модель, потому что уводит активацию off-distribution"

Основные найденные источники и их формализация:
1. **GLP** (см. раздел 3) — Delta LM Loss как прокси off-distribution damage; денойзинг возвращает активацию на многообразие через flow matching, что снижает LM loss (0.0513–0.0860 против 0.1976–0.2224 у SAE-baseline на Llama8B — см. раздел 3).
2. **KL-дивергенция к unsteered-распределению** — KL-then-Steer (arXiv:2406.15518) и "Safety Cost of Steering Vectors" (arXiv:2608.08383, refusal-token divergence $D_R$, ASR/FRR) — раздел 1.3.
3. **Angle-Norm Decomposition** (arXiv:2606.06735) — норма активации после стиринга как индикатор ухода с многообразия; авторы явно связывают "linear steering can substantially change the activation norm, pushing activations out of distribution and thereby degrading the model" с мотивацией для spherical/norm-aware методов.
4. Mahalanobis distance и reconstruction error как формальные метрики off-manifold степени **напрямую к activation steering в найденной литературе не привязаны** — это пробел, который стоит заполнить самостоятельно (потенциальный вклад твоей работы).

---

## 3. GLP — "Learning a Generative Meta-Model of LLM Activations"

Grace Luo, Jiahai Feng, Trevor Darrell, Alec Radford, Jacob Steinhardt.
**arXiv:2602.06964** (принята на ICML 2026). Project page: generative-latent-prior.github.io.

Это прямой предшественник задуманного метода — детали важны для честного позиционирования.

**Абстракт (перевод сути):** существующие подходы к анализу активаций (PCA, SAE) опираются на жёсткие структурные допущения. Генеративные модели могут выявлять структуру без таких допущений и работать как приор, повышающий точность интервенций. Авторы обучают диффузионные модели на миллиарде residual-stream активаций ("meta-models"), выучивающих распределение внутренних состояний сети. Диффузионный лосс гладко падает с ростом вычислений и надёжно предсказывает downstream-полезность. Применение выученного приора к стирингу улучшает беглость текста, причём выигрыш растёт по мере падения лосса. Нейроны meta-модели всё сильнее изолируют отдельные концепты по мере падения лосса (растут sparse probing scores).

**Метод.** Несмотря на слово "diffusion" в названии подхода, техническая реализация — **flow matching**: forward-процесс строит $z_t$ как линейную интерполяцию между точкой данных $z_0$ и шумом $\varepsilon$; сеть-денойзер $u_\theta(z_t,t)$ обучается аппроксимировать целевую скорость $u=\varepsilon-z_0$ (MSE-лосс на предсказании скорости). Архитектура — глубокий MLP со SwiGLU-блоками и residual-связями (не архитектура SAE); протестированы размеры 0.5B/0.9B/1.7B/3.3B параметров; ширина модели = 2× размерности активации, expansion factor gated-MLP = ещё 2×.

**Данные.** Активации Llama-3.2-1B (основные scaling-эксперименты, слой 7 — "middlemost") и Llama-3.1-8B (downstream-применения, слой 15); всего 1 млрд токенов/активаций из корпуса **FineWeb**. Гиперпараметры обучения: batch size 4096, learning rate 5e-5, cosine schedule, warmup ratio 0.01.

**Применение к стирингу (алгоритм denoising после интервенции):**
1. Применить интервенцию: $\text{acts\_edit} = \text{acts} + \alpha\cdot w$ ($w$ — вектор стиринга).
2. Стандартизировать (привести к нулевому среднему и единичной дисперсии по статистике обучающего распределения активаций).
3. Добавить шум на уровне $t_{start}$: $\text{acts\_noisy} = (1-t_{start})\cdot\text{acts\_edit} + t_{start}\cdot\text{noise}$.
4. Запустить многошаговый denoising-сэмплинг от $t=t_{start}$ до $t=0$.
5. Вернуть исходную статистику (обратная стандартизация).
Гиперпараметры: $t_{start}=0.5$, num_steps=20. То есть это ровно "проекция интервенированной активации назад на выученное многообразие" через частичный flow-matching денойзинг, а не полная генерация с нуля.

**Метрики.**
- Fluency: основная метрика — **conditional negative log-likelihood под той же LLM** (self-perplexity), производная — **Delta LM Loss** (прирост LM loss/перплексии стирированной генерации относительно нестерированного baseline). Ошибки — 95% bootstrap CI.
- Concept strength: для sentiment — LLM-judge по шкале **0–2**, либо five-point sentiment classifier (non-LLM alt.); для персона-концептов — шкала **0–100**.
- Probing: **AUC** для 113 бинарных классификационных задач (из прежней работы), логистическая регрессия L-BFGS.

**Бейзлайны сравнения:** Persona Vectors (§4.2), DiffMean (§4.3, sentiment steering), Sparse Autoencoders — конкретно **LlamaScope SAE** (He et al. 2024), а также raw layer activations и raw MLP neurons как пробинг-бейзлайны.

**Ключевые количественные результаты:**
- **Table 2, Delta LM Loss на Llama8B:** SAE — 0.1976 (Base) / 0.2224 (Instruct); **GLP — 0.0513 (Base) / 0.0860 (Instruct)** — т.е. GLP-денойзинг снижает прирост LM loss после стиринга примерно в 2.6–3.9 раза относительно SAE-стиринга.
- **Figure 5**: GLP-постобработка расширяет Парето-фронт (concept strength vs fluency) относительно чистого SAE-стиринга на 500 случайных направлениях.
- **Figure 12**: GLP расширяет Парето-фронт относительно DiffMean baseline и для позитивного, и для негативного sentiment-стиринга.
- **Table 4, 1-D probing AUC:** Llama1B GLP **0.84** [0.81, 0.87] против SAE 0.70, raw layer output 0.77, raw MLP neurons 0.79; Llama8B GLP **0.87** [0.84, 0.89] против SAE 0.76, raw layer output 0.77, raw MLP neurons 0.82. Т.е. GLP как признаковое пространство для линейного пробинга превосходит и SAE, и сырые активации/нейроны.

**Вывод для позиционирования:** GLP — это ровно "flow-matching денойзер активаций, проецирующий на выученное многообразие" в чистом виде, обученный как general-purpose meta-model (не специально под стиринг), применяемый постфактум к $h+\alpha v$ через частичный noise-and-denoise цикл. Отличия, которые стоит формализовать в своей работе: (а) чем денойзер GLP отличается от предлагаемого — архитектурно, по данным обучения, по способу интеграции в pipeline стиринга; (б) какие метрики/бейзлайны GLP не покрыл (например, нет прямого сравнения с CAA/ActAdd/norm-preserving/SAE-TS — только с Persona Vectors, DiffMean, SAE); (в) GLP не даёт формальной off-distribution метрики (Mahalanobis/reconstruction error) — использует только LM loss как прокси.

---

## 4. Denoising / manifold projection для активаций — прочие работы

- **GLP** (arXiv:2602.06964) — см. раздел 3, основной релевантный прецедент.
- **Riemannian-Manifold Steering: Geometry-Aware Generative Autoencoders for Label-Free Steering** (arXiv:2605.24942) — обучают генеративный автоэнкодер, выучивающий геометрию (риманово многообразие) активаций; стиринг ограничен геодезическими путями на многообразии, а не прямыми линиями в исходном пространстве активаций. Заявлены "projection operations onto the manifold" и информационно-геометрические метрики расстояния между состояниями активаций. Точные формулы геодезического шага и числовые результаты извлечь из PDF не удалось (контент — сжатый PDF-стрим, не распарсился); при необходимости нужно скачать и прочитать PDF напрямую, а не через WebFetch.
- **Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement** (arXiv:2509.23799) — денойзинг векторов стиринга через интерпретируемое SAE-пространство фичей (уточняет сам вектор стиринга, а не денойзит итоговую активацию после интервенции — концептуально другой шаг pipeline, чем GLP).
- **Graph-Regularized Sparse Autoencoders for LLM Safety Steering** (arXiv:2512.06655) — граф-регуляризация SAE (Laplacian графа co-активации нейронов) для более согласованных фичей; не про денойзинг/многообразие напрямую, но релевантно как альтернативный способ "структурировать" пространство активаций перед стирингом.
- Прямых работ про **денойзинг-автоэнкодер** (classic DAE, не flow/diffusion) специально для постобработки стирированных активаций — не найдено.

---

## Пробелы (что не найдено / требует доп. проверки)

1. Дословный текст judge-промпта из safety-research/persona_vectors — не подтверждён (404 на прямом пути к judge.py; README не публикует текст промпта).
2. Формула Fluency-Diversity Rate $(Dist_2)^2/\ln(PPL)$ — источник (конкретная статья/arxiv id) не идентифицирован надёжно, использовать после дополнительной проверки.
3. Числа KL=2.088 → KL=0.044 (47× снижение) из предварительного поиска — **не подтверждены** в тексте arXiv:2608.08383 при повторном fetch; не цитировать без независимой проверки первоисточника.
4. Прямая связка Mahalanobis distance / reconstruction error как формальной метрики "схода с многообразия" именно для activation steering — не найдена; похоже на открытый зазор в литературе, который твоя работа может закрыть.
5. Riemannian-Manifold Steering (arXiv:2605.24942) — точные формулы и числа не извлечены (PDF не распарсился через доступные инструменты), нужен отдельный заход с полным чтением PDF.
