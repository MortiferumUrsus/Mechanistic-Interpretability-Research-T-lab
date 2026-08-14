# Инструменты для activation steering на GPT-2 small — техническая справка

Дата сбора: 2026-08-14. Источники — официальные репозитории на GitHub (raw-файлы), PyPI JSON API,
Hugging Face, Neuronpedia docs. Там, где сведения не удалось подтвердить дословно, стоит пометка
«требует проверки в рантайме» с конкретным вызовом для проверки.

---

## 1. SAELens (github.com/decoderesearch/SAELens, ранее jbloomAus/SAELens)

Репозиторий активно поддерживается (последний push — 2026-08-10, не archived).
PyPI-пакет называется `sae-lens`, последняя версия на момент сбора — **6.49.1**.

### 1.1 Установка и жёсткая связка версий

`sae-lens==6.49.1` в `requires_dist` жёстко пинит TransformerLens:

```
transformer-lens<2.16.1,>=2.16.1   # т.е. ровно 2.16.1
transformers<6.0.0,>=4.38.1
datasets>=3.1.0
safetensors<1.0.0,>=0.4.2
```

Важно: если поставить `pip install sae-lens transformer_lens` без уточнения версий, pip обязан
разрешить `transformer_lens` до **2.16.1**, а не до последней версии TransformerLens (на 2026-08-14
последняя отдельная версия TransformerLens — 3.7.1, но она несовместима с SAELens 6.x).
Проверить после установки: `pip show transformer-lens sae-lens transformers torch`.

### 1.2 Загрузка предобученного SAE — точный вызов

Из официального тьюториала `tutorials/basic_loading_and_analysing.ipynb` (decoderesearch/SAELens):

```python
from sae_lens import SAE

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",   # слой 6, hook_resid_pre
    device=device,                      # "cuda" / "cpu"
)
```

**Важное изменение API в v6** (миграция v5→v6, см. `docs/migrating.md` в репозитории):
- в v5 `SAE.from_pretrained(...)` возвращал кортеж `(sae, cfg_dict, sparsity)`;
- в v6 возвращает **только объект `sae`** (как у HF Transformers / TransformerLens). Старый
  распаковочный синтаксис `sae, cfg_dict, sparsity = SAE.from_pretrained(...)` всё ещё работает,
  но кидает DeprecationWarning.
- Если нужны cfg_dict/sparsity отдельно — `SAE.load_from_pretrained_with_cfg_and_sparsity(...)`
  (в доках отмечено, что большинству пользователей не нужно).
- Конфиг теперь лежит в `sae.cfg` (только «существенные для запуска» поля: `d_in`, `d_sae`,
  `dtype`, `device`, ...), а второстепенные метаданные (например `prepend_bos`, `hook_name`,
  `context_size`, `dataset_path`) — в `sae.cfg.metadata`.

Раз версия sae-lens может обновиться, перед использованием стоит проверить на месте:
```python
print(sae.cfg)               # d_in, d_sae, dtype, device
print(sae.cfg.metadata)      # hook_name, hook_layer, dataset_path, context_size, prepend_bos
```

### 1.3 Релизы для gpt2-small, residual stream, слой 6

Данные вытащены дословно из `sae_lens/pretrained_saes.yaml` (main-ветка SAELens, файл на
43673 строки — тянуть его напрямую через веб-морду GitHub ненадёжно, был скачан `raw.githubusercontent.com`
и прогреплен). Ключевые releases для GPT-2 small residual stream:

**`gpt2-small-res-jb`** (Joseph Bloom, "Open Source SAEs for all residual stream layers", классика,
покрывает `hook_resid_pre` каждого слоя 0-11 + `hook_resid_post` слоя 11):
```yaml
gpt2-small-res-jb:
  repo_id: jbloom/GPT2-Small-SAEs-Reformatted
  model: gpt2-small
  config_overrides:
    model_from_pretrained_kwargs:
      center_writing_weights: true
  saes:
  - id: blocks.6.hook_resid_pre
    path: blocks.6.hook_resid_pre
    l0: 51.0
    variance_explained: 0.9
    neuronpedia: gpt2-small/6-res-jb
```
→ `sae_id` для слоя 6 = **`"blocks.6.hook_resid_pre"`**.
Конфиг с HF (`jbloom/GPT2-Small-SAEs-Reformatted/blocks.6.hook_resid_pre/cfg.json`), подтверждён дословно:
```json
{
  "d_in": 768,
  "d_sae": 24576,
  "hook_point": "blocks.6.hook_resid_pre",
  "hook_point_layer": 6,
  "model_name": "gpt2-small",
  "dataset_path": "Skylion007/openwebtext",
  "context_size": 128,
  "dtype": "torch.float32"
}
```
Т.е. **32x expansion** (768 → 24576), обучен на `Skylion007/openwebtext`, L1-регуляризация
(классический ReLU-SAE, не top-k), variance explained ~0.9 на слое 6.

**`gpt2-small-resid-post-v5-32k`** и **`gpt2-small-resid-post-v5-128k`** — это порты SAE из
**`openai/sparse_autoencoder`** (OpenAI v5), переупакованные под SAELens (обратите внимание —
`repo_id` буквально называется `jbloom/GPT2-Small-OAI-v5-*`, "OAI" = OpenAI):
```yaml
gpt2-small-resid-post-v5-32k:
  repo_id: jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs
  model: gpt2-small
  config_overrides:
    dataset_path: Skylion007/openwebtext
  saes:
  - id: blocks.6.hook_resid_post
    path: v5_32k_layer_6.pt
    l0: 32.0
    variance_explained: 0.9
    neuronpedia: gpt2-small/6-res_post_32k-oai

gpt2-small-resid-post-v5-128k:
  repo_id: jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs
  saes:
  - id: blocks.6.hook_resid_post
    path: v5_128k_layer_6
    l0: 32.0
    variance_explained: 0.9
    neuronpedia: gpt2-small/6-res_post_128k-oai
```
→ `sae_id` для слоя 6 = **`"blocks.6.hook_resid_post"`** (обратите внимание — **другой хук**,
`resid_post`, не `resid_pre`!). d_sae по названию релиза: 32k → **d_sae=32768**, 128k → **d_sae=131072**
(точное число из названия версии OpenAI, не встретилось в yaml дословно как число —
**требует проверки в рантайме**: `sae.cfg.d_sae` после загрузки). d_in=768 в обоих случаях.

Также существует **`gpt2-small-res-jb-feature-splitting`** (эксперимент по feature splitting,
только слой 8, разные `d_sae` от 768 до 98304 — не нужен для слоя 6) и набор `gpt2-small-res_sc*-ajt`
/ `res_sl*-ajt` (Neuronpedia-native SAE, покрывают только слои 2/6/10, `hook_resid_pre`,
`repo_id: neuronpedia/gpt2-small__res_*-ajt`) — тоже валидные варианты для слоя 6, но без
`l0`/`variance_explained` в конфиге (менее документированы), пример:
```yaml
gpt2-small-res_sce-ajt:
  repo_id: neuronpedia/gpt2-small__res_sce-ajt
  saes:
  - id: blocks.6.hook_resid_pre
    path: 6-res_sce-ajt
    neuronpedia: gpt2-small/6-res_sce-ajt
```

**Рекомендация под задачу steering**: для базового эксперимента разумнее всего взять
`release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre"` — это самый "канонiчный",
документированный и часто цитируемый вариант, с готовым Neuronpedia-дашбордом
(`gpt2-small/6-res-jb`) и понятной семантикой хука.

### 1.4 hook_resid_pre vs hook_resid_post — куда именно вставлять интервенцию

- `blocks.6.hook_resid_pre` — это residual stream **на входе в блок 6**, т.е. это ровно тот же
  тензор, что и `blocks.5.hook_resid_post` (residual stream идёт "насквозь", pre слоя L == post
  слоя L-1 для L>0). SAE, обученный на `hook_resid_pre` слоя 6, реконструирует то, что "видит"
  блок 6 перед своей attention-подрешёткой.
- `blocks.6.hook_resid_post` — residual stream **на выходе из блока 6**, после attn+MLP слоя 6.
- **Для интервенции (activation steering) это означает**: если вектор фичи (`sae.W_dec[i]`)
  добавляется через хук на `blocks.6.hook_resid_pre`, то интервенция происходит *до* слоя 6
  (влияет на слои 6..11 включительно). Если через `blocks.6.hook_resid_post` — интервенция
  происходит *после* слоя 6 (влияет на слои 7..11). При смешивании SAE, обученного на одном
  хуке, с интервенцией на другом хуке — семантика фич не гарантированно переносится (хотя
  numerически pre(L)==post(L-1), это один и тот же тензор в модели без residual-стрима
  изменений между слоями, так что `hook_resid_pre` слоя 6 и `hook_resid_post` слоя 5 —
  один и тот же вектор активаций; но SAE `res-jb` слоя 6 обучен именно на этом месте).

Практический вывод: если взяли `gpt2-small-res-jb` / `blocks.6.hook_resid_pre`, вставлять вектор
steering нужно хуком именно на **`"blocks.6.hook_resid_pre"`** (или эквивалентно на
`"blocks.5.hook_resid_post"` — тот же тензор), а не на `blocks.6.hook_resid_post`.

### 1.5 W_dec — декодерные векторы фич, нормировка

```python
W_dec = sae.W_dec          # torch.Tensor, shape [d_sae, d_in] = [24576, 768] для res-jb слой 6
norms = W_dec.norm(dim=-1)  # ожидается ~1.0 для каждой строки, если декодер unit-norm
```
По документации/практике SAELens (см. `training_saes.md`, обсуждения folding весов): декодер в
SAE, обученных этой библиотекой, обычно **приводится к unit norm** (constraint во время обучения:
`||W_dec[i]||_2 = 1` для каждой фичи `i`), а после обучения статистики активаций
"folds" в веса энкодера/декодера (см. `normalize_activations` конфиг, значения вида
`"expected_average_only_in"`). Но это подтверждено только на уровне общей документации/практики,
**не дословной цитатой конфига `gpt2-small-res-jb`** (поля `normalize_activations` /
`apply_b_dec_to_input` не встретились в `cfg.json`, который удалось прочитать) →
**требует проверки в рантайме**:
```python
sae.W_dec.norm(dim=-1).mean(), sae.W_dec.norm(dim=-1).std()
```
Если норма не ровно 1.0 — перед steering вектор фичи стоит вручную нормализовать
(`sae.W_dec[i] / sae.W_dec[i].norm()`) и домножать на свой коэффициент силы интервенции.

### 1.6 Neuronpedia — человекочитаемые описания фич программно

- Каждый SAE-релиз в `pretrained_saes.yaml` содержит поле `neuronpedia` для каждой фичи-хука
  (например `gpt2-small/6-res-jb`) — это `modelId/saeId`-подобный идентификатор на
  neuronpedia.org, по которому строится URL дашборда: `https://neuronpedia.org/gpt2-small/6-res-jb`.
- В самом SAELens есть мост: `sae_lens.analysis.neuronpedia_integration`, конкретно функция
  `get_neuronpedia_quick_list(sae, feature_idx_list)` — генерирует ссылку на Neuronpedia "quick
  list" для набора индексов фич (подтверждено кодом из тьюториала `basic_loading_and_analysing.ipynb`):
  ```python
  from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
  url = get_neuronpedia_quick_list(sae, test_feature_idx_gpt)
  ```
- Для получения текстовых объяснений (auto-generated explanations) программно — у Neuronpedia
  есть публичный REST API (`https://neuronpedia.org/api-doc`, "work in progress" по их же docs),
  плюс пакет `neuronpedia` на PyPI. Точный endpoint для конкретной фичи (что-то вроде
  `GET /api/feature/{modelId}/{saeId}/{index}`, отдаёт explanations/activations) **не удалось
  подтвердить дословно** через доступные источники → **требует проверки в рантайме**: открыть
  `https://neuronpedia.org/api-doc` (Swagger/OpenAPI UI) и проверить нужен ли API key (похоже что
  для чтения публичных данных — нет, но не подтверждено).

### 1.7 Проверка актуального списка releases в рантайме

Поскольку `pretrained_saes.yaml` — живой файл (его пополняют), перед тем как жёстко хардкодить
`release`/`sae_id` в коде, стоит на месте прогнать:
```python
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
import pandas as pd

d = get_pretrained_saes_directory()
df = pd.DataFrame.from_records({k: v.__dict__ for k, v in d.items()}).T
gpt2_res = df[(df.index.str.contains("gpt2-small")) & (df.index.str.contains("res"))]
print(gpt2_res)
```

---

## 2. TransformerLens (github.com/TransformerLensOrg/TransformerLens)

Активно поддерживается (последний push — 2026-08-14). PyPI-пакет `transformer-lens`.

### 2.1 Загрузка gpt2-small

```python
from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained(
    "gpt2-small",             # или просто "gpt2" — алиас в TransformerLens
    device="cuda",
    dtype=torch.float32,      # см. раздел 5 — bf16/автоматика могут быть проблемой на Turing
)
```
GPT-2 small в терминах TransformerLens: `n_layers=12`, `d_model=768`, `n_heads=12`, `d_head=64`,
`n_ctx=1024`, `d_vocab=50257` — соответствует стандартной архитектуре GPT-2 117M/124M.

### 2.2 Хук для добавления вектора в residual stream на конкретном слое

Имя хука: **`blocks.{L}.hook_resid_post`** (или `hook_resid_pre` / `hook_resid_mid` — см. раздел 1.4).

```python
def steer_hook(resid, hook):
    # resid: [batch, seq, d_model]
    resid = resid + coefficient * steering_vector   # steering_vector: [d_model]
    return resid

logits = model.run_with_hooks(
    tokens,
    fwd_hooks=[("blocks.6.hook_resid_pre", steer_hook)],
)
```

Полный список хуков в остаточном потоке одного блока: `hook_resid_pre` (вход блока) →
(attn) → `hook_resid_mid` (после attn, перед MLP) → (mlp) → `hook_resid_post` (выход блока).
Плюс есть отдельные `hook_embed`, `hook_pos_embed` перед блоком 0.

### 2.3 Контекстный менеджер хуков + генерация

```python
def steer_hook(resid, hook):
    resid = resid + coefficient * steering_vector
    return resid

with model.hooks(fwd_hooks=[("blocks.6.hook_resid_pre", steer_hook)]):
    output = model.generate(
        prompt_tokens,
        max_new_tokens=50,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        use_past_kv_cache=True,
        prepend_bos=True,
    )
```
Альтернатива без контекстного менеджера — `model.add_hook(name, hook_fn)` +
`model.reset_hooks()` / `model.remove_hook(name)` вручную; либо разовый вызов
`model.run_with_hooks(tokens, fwd_hooks=[...])` (хуки автоматически снимаются после вызова).
`model.hooks(...)` — контекстный менеджер, найден и подтверждён по практике
(LessWrong "Implementing activation steering" и другим примерам с TransformerLens), хотя
дословно в официальном API-референсе Orchestra-Research его текст не процитировал — если он
не сработает в установленной версии, эквивалент — обернуть `add_hook`/`reset_hooks` в
`try/finally`.

### 2.4 Совместимость версий (torch / transformers) — под связку с SAELens

Два разных сценария:

**А) Ставить TransformerLens отдельно, последнюю версию.**
`transformer-lens==3.7.1` (последняя на 2026-08-14) требует:
```
torch>=2.6
transformers>=5.9.0        # !! современная мажорная версия transformers, не 4.x
python>=3.10,<4.0
numpy>=1.24 (py3.10-3.11) / >=1.26 (py3.12)
accelerate>=0.23.0
```

**Б) Ставить вместе с `sae-lens` (нужно для нашего эксперимента).**
`sae-lens==6.49.1` жёстко требует `transformer-lens==2.16.1` (не последнюю!), а та, в свою
очередь (проверено через PyPI JSON для tag 2.16.1):
```
python>=3.9: torch>=2.6, transformers>=4.51, numpy<2
python==3.8: torch<2.6, transformers<4.51   # неактуально, если ставите свежий Python
```
Т.е. **при совместной установке SAELens+TransformerLens транзитивно приедет `transformer-lens==2.16.1`
и `transformers` где-то в диапазоне `[4.51, 6.0)`**, а не последний `transformers>=5.9.0` из
пункта А. Практическая рекомендация: ставить одной командой
`pip install sae-lens transformer_lens` (без явного пина версии transformer_lens) и дать
резолверу самому подобрать 2.16.1 — не пытаться руками поставить последний TransformerLens 3.7.1
рядом с SAELens, они несовместимы по пину.

Проверка после установки:
```bash
pip show transformer-lens sae-lens transformers torch | grep -E "Name|Version"
```

---

## 3. openai/sparse_autoencoder

Репозиторий существует и формально **не архивирован** (`archived: false` в GitHub API), но
**фактически заброшен**: последний `push` — **2024-07-19**, т.е. более двух лет без изменений
(на дату сбора справки, 2026-08-14). Для нового кода в 2026 году полагаться на него как на
активно поддерживаемую библиотеку не стоит.

Что там есть для GPT-2 small: набор чекпоинтов SAE версий **v4**, **v5_32k**, **v5_128k**,
обученных на нескольких точках residual stream/MLP (`resid_delta_attn`, `resid_delta_mlp`,
`resid_post_attn`, `resid_post_mlp`, `mlp_post_act`, ...) и разных слоях. Загрузка — через
собственный модуль `sparse_autoencoder.paths` (пути в облачном сторадже, тянутся `blobfile`),
пример из README:
```python
import transformer_lens
import sparse_autoencoder

model = transformer_lens.HookedTransformer.from_pretrained("gpt2")
path = sparse_autoencoder.paths.v5_32k(location="resid_post_mlp", layer_index=6)
state_dict = sparse_autoencoder.paths.blob_file... # см. README, точная функция подгрузки весов
autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
latent_activations, info = autoencoder.encode(input_tensor_ln)
```
(Дословный вызов для скачивания весов по URL из README процитировать не удалось полностью —
**требует проверки в рантайме**, открыть `README.md` репозитория и код `sparse_autoencoder/paths.py`.)

**Практическая рекомендация**: не использовать `openai/sparse_autoencoder` напрямую. Те же самые
веса OpenAI v5 (32k/128k) уже переупакованы под современный SAELens-loader — это разделы
`gpt2-small-resid-post-v5-32k` / `gpt2-small-resid-post-v5-128k` из раздела 1.3 выше
(`repo_id: jbloom/GPT2-Small-OAI-v5-*-resid-post-SAEs`). Грузить их через
`SAE.from_pretrained(release="gpt2-small-resid-post-v5-32k", sae_id="blocks.6.hook_resid_post")` —
проще, современнее и не требует `blobfile`/устаревшего кода.

---

## 4. Датасет активаций для gpt2-small (слабый интернет, 6 ГБ VRAM)

| Датасет (HF) | Что это | Размер | Комментарий |
|---|---|---|---|
| `NeelNanda/pile-10k` | первые 10k документов из The Pile, сырой текст | **33.3 MB** | Самый маленький и самый безопасный выбор при слабом интернете. Стандартный "debug/demo" корпус в экосистеме TransformerLens/SAELens. |
| `apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2` | уже токенизированный (GPT-2 tokenizer) вариант `monology/pile-uncopyrighted` (Pile без сабсетов с копирайтом, отфильтрован под токены OpenWebText) | parquet, заявлен как 10–100 MB (HF size-category) | Удобен, если хочется сразу готовые токены без своей токенизации; но нестандартный/менее заметный источник — стоит проверить `dataset_infos.json` на HF перед использованием. |
| `Skylion007/openwebtext` | полный OpenWebText, на нём обучены и `gpt2-small-res-jb`, и OpenAI v5 SAE (в `config_overrides.dataset_path`) | **много GB** (полный корпус, десятки GB) | Это "родной" корпус для SAE из раздела 1, но при слабом интернете и цели просто прогнать steering-эксперимент (а не переобучать SAE) — качать весь датасет не нужно. |

**Рекомендация под условия (слабый интернет, 6 ГБ VRAM, задача — steering, не тренировка SAE):**
брать `NeelNanda/pile-10k` для сбора активаций/тестовых промптов. Полный `Skylion007/openwebtext`
нужен только если планируется *переобучать* SAE с нуля — для чистого activation steering с уже
готовым SAE это не требуется.

```python
from datasets import load_dataset
ds = load_dataset("NeelNanda/pile-10k", split="train")
```

---

## 5. GTX 1660 Ti, 6 ГБ VRAM, sm_75 — практические ограничения

Подтверждено: GeForce GTX 1660 Ti построена на чипе **TU116 (Turing)**, compute capability
**sm_75 (7.5)**. Turing поддерживает **fp16** (через Tensor Cores начиная с этой архитектуры на
чипах с ними; у 1660 Ti конкретно нет отдельных Tensor Cores, но fp16-арифметика на CUDA cores
работает), но **bf16 аппаратно не поддерживается** — bf16 появился начиная с Ampere (sm_80+),
Turing и более старые архитектуры его не имеют.

Подводные камни для связки torch 2.6+ / TransformerLens / SAELens на этой карте:

1. **dtype: используйте `torch.float32` или `torch.float16`, НЕ `torch.bfloat16`.**
   Если код (в т.ч. дефолты какой-то библиотеки) попытается создать/скастовать тензор в bf16 на
   этой GPU — либо будет исключение, либо (в зависимости от пути) тихий фолбэк/эмуляция
   через float32 с потерей смысла ускорения. Проверка в рантайме:
   ```python
   torch.cuda.get_device_capability()   # ожидается (7, 5)
   torch.cuda.is_bf16_supported()       # ожидается False
   ```
   Для GPT-2 small (124M параметров) и SAE (768×24576 ×2 ~ 38M параметров) даже fp32 полностью
   помещается в 6 ГБ VRAM с большим запасом — гнаться за fp16 ради экономии памяти не обязательно,
   fp32 безопаснее для отладки численной стороны steering. Конфиг `gpt2-small-res-jb`
   (см. 1.3) сам обучен и хранится в `torch.float32`.

2. **torch.compile / Triton**: Triton формально поддерживает CUDA capability ≥ 7.0 (значит sm_75
   попадает), но на практике многие готовые скомпилированные kernel'ы в актуальных версиях
   Triton/torch собираются только под sm_80+/sm_90, из-за чего `torch.compile` на Turing может
   падать с ошибками отсутствующих kernel'ов или требовать компиляции на лету (медленнее,
   иногда нестабильно на Windows). Рекомендация: для эксперимента с steering `torch.compile`
   не обязателен (модель маленькая, инференс и так быстрый) — начинать **без** него, включать
   только если возникнет explicit need и после отдельной проверки, что компиляция вообще
   проходит на этой машине.

3. **FlashAttention**: FlashAttention-2 (`flash-attn` пакет от Dao-AILab) **официально не
   поддерживает Turing** (заявлена поддержка Ampere/Ada/Hopper); для Turing актуальна только
   ветка **FlashAttention 1.x**. Но это не критично: `HookedTransformer` в TransformerLens по
   умолчанию использует собственную (eager/наивную либо `torch.nn.functional.scaled_dot_product_attention`)
   реализацию внимания, а не пакет `flash-attn` напрямую — flash-attention актуален в основном
   если явно подключать HF `transformers` с `attn_implementation="flash_attention_2"`. Для GPT-2
   small (n_ctx=1024, маленькая модель) выигрыш от FlashAttention в любом случае небольшой —
   можно спокойно игнорировать и не ставить `flash-attn` вообще.

4. **CUDA/драйвер**: torch>=2.6 wheels собираются под CUDA 12.x; sm_75 поддерживается всеми
   актуальными сборками CUDA 12.x (просто без bf16/новых Tensor Core фич Ampere+). Проверить
   реальную видимость GPU и версию CUDA у установленного torch:
   ```python
   import torch
   print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
   print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
   ```

5. **VRAM 6 ГБ — бюджет памяти**: GPT-2 small в fp32 — это ~124M × 4 байта ≈ 500 MB весов;
   один SAE `gpt2-small-res-jb` слоя 6 (d_in=768, d_sae=24576) — encoder+decoder ≈
   2 × 768 × 24576 × 4 байта ≈ **151 MB**. Даже с оптимизатором/градиентами (если вдруг
   дообучать SAE) и активациями на батч это далеко от предела 6 ГБ для чистого инференса +
   steering. Узкое место скорее будет не VRAM, а **скорость скачивания весов/датасета** —
   отсюда рекомендация в разделе 4 брать маленький `NeelNanda/pile-10k`, а не полный OpenWebText.

---

## Сводка того, что нужно проверить в рантайме перед написанием кода

1. `get_pretrained_saes_directory()` — актуальный список releases/sae_id для gpt2-small (yaml
   меняется со временем, могут появиться новые/точнее задокументированные варианты для слоя 6).
2. `sae.cfg` / `sae.cfg.metadata` сразу после `SAE.from_pretrained(...)` — точные `d_sae`, `d_in`,
   `hook_name`, `dtype` для конкретно выбранного release (особенно для `v5-32k`/`v5-128k`, где
   d_sae=32768/131072 не подтверждён дословной цитатой конфига).
3. `sae.W_dec.norm(dim=-1)` — действительно ли декодер unit-norm (не подтверждено дословно для
   `gpt2-small-res-jb`, только общей практикой SAELens).
4. `pip show transformer-lens sae-lens transformers torch` после установки — какие версии
   реально зарезолвил pip (ожидание: transformer-lens==2.16.1, transformers в [4.51, 6.0)).
5. `torch.cuda.get_device_capability()` и `torch.cuda.is_bf16_supported()` на конкретной машине.
6. `https://neuronpedia.org/api-doc` — точный REST-эндпоинт и нужен ли API key для получения
   explanations программно (помимо уже подтверждённого моста `get_neuronpedia_quick_list`).
7. Если решите всё же трогать `openai/sparse_autoencoder` — открыть его `README.md` и
   `sparse_autoencoder/paths.py` напрямую (репозиторий с 2024 года не менялся, но точные имена
   функций подгрузки весов лучше проверить de visu, а не по пересказу).
