# Ремонт стиринга в GPT-2 small: диагностика передачи и три дешёвых способа починить fluency

Решение тестового задания T-Lab 2026, трек Mechanistic Interpretability.
Текст задания: [`../tasks/01-mechanistic-interpretability.md`](../tasks/01-mechanistic-interpretability.md).
Предрегистрация гипотез и протокола: [`PLAN.md`](PLAN.md). Отчёт: [`REPORT.md`](REPORT.md).

Задача: стиринг `h̃ = h + αv` вдоль направления из декодера SAE усиливает нужное свойство, но при
больших `α` ломает связность текста. Нужно уменьшить негативный эффект дешёвым способом.

## Что здесь есть

Интервенция ставится после серединного слоя, в `blocks.6.hook_resid_post`. Этот тензор побитово
совпадает с `blocks.7.hook_resid_pre`, на котором обучены SAE релиза `gpt2-small-res-jb`, поэтому
словарь признаков лежит ровно в базисе интервенции (проверяется командой `activations.py verify`).

Сила стиринга параметризуется приращением концепт-координаты `s = c · median‖h‖`, а не сырым `α`:
так сетка сравнима между признаками, и все методы сопоставляются при равной силе концепта.

| Плечо | Что делает | Обучение |
|---|---|---|
| `clean` | `h` | — |
| `naive` | `h + s·v̂` | — |
| `norm_preserving` | сохраняет норму активации | — |
| `denoise_naive` | `x + η(D(x) − x)` — буквальная формулировка задания | денойзер |
| `cds` | `x + λ·P⊥(D(x) − D(h) − s·v̂)` — контрастная коррекция с сохранением направления | денойзер |
| `mts` | сдвиг минимальной махаланобисовой нормы при том же приращении концепт-координаты | — |
| `fsr` | кламп не-целевых латентов SAE под их естественный корпусный потолок | — |
| `dirfix` | **второй раунд:** инъекция вдоль обученной поправки направления, при той же норме возмущения | поправка направления |

Первый раунд (`clean` … `fsr`) — плечи, объявленные до запусков. `dirfix` — второй раунд: метод выбран
после просмотра результатов первого, и в отчёте это сказано отдельным разделом. Поправка направления
обучается только на `FIT`-признаках и применяется к признакам, которых не видела; норма возмущения у неё
совпадает с наивной по построению, поэтому она не может выиграть, инжектируя больше.

## Установка

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install "numpy<2.1" transformer-lens sae-lens transformers datasets matplotlib pandas scipy tqdm pyyaml
.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Порядок важен: `transformer-lens` и `sae-lens` тянут с PyPI CPU-сборку torch, поэтому CUDA-сборка
ставится последней. Проверено на GTX 1660 Ti (6 ГБ, sm_75), torch 2.13.0+cu126.

## Воспроизведение

Полный конвейер одной командой: `scripts\run_all.ps1`. По стадиям:

```
cd src

# данные: тождество хуков, дамп активаций, промпты, потолки латентов SAE
python activations.py verify
python activations.py dump --n-tokens 2000000 --batch-size 16
python activations.py prompts --n-prompts 40 --prompt-len 8
python sae_stats.py --chunk 2048

# отбор признаков и разбиение FIT / DEV / TEST
python features.py select --seed 0

# обучение денойзеров только на FIT-направлениях
python train_denoiser.py --arch mlp --noise mix --cond 1 --seed 0 --steps 60000

# выбор всех инференс-ручек на DEV, без генерации и без судьи
python sweep_dev.py

# один прогон на TEST с замороженными ручками
python generate.py --split test --out gen_test.jsonl
python metrics.py --gen gen_test.jsonl --out scored_test.csv --stages ppl,keyword,sae,dist
python metrics.py --gen gen_test.jsonl --out scored_test.csv --stages judge
python pareto.py --scored scored_test.csv --concept keyword_hit

# механизм
python analysis.py transmission
python analysis.py spectral
python analysis.py surgery
python analysis.py causal
python analysis.py predictors
```

Второй раунд целиком: `scripts\run_round2.ps1`.

Проверки инвариантов: `python test_arms.py` и `python test_pareto.py`.

## Контроль утечки

Индексы признаков SAE разбиты на три непересекающихся множества. `FIT` (24000 признаков) участвует
только в синтезе обучающих возмущений, `DEV` (6) — в выборе всех гиперпараметров, `TEST` (12)
открывается один раз. Любой признак с `|cos| ≥ 0.3` к любому направлению из `DEV ∪ TEST` исключён
из `FIT`; фактический максимум по всем парам — в [`results/leakage.csv`](results/leakage.csv).
Промпты для оценки взяты из другого шарда корпуса, чем активации для обучения.

Все инференс-ручки (`λ`, `η`, `k`, усадка ковариации, `σ`) фиксируются на DEV и записываются в
`configs/frozen.yaml` до первого запуска на TEST; на TEST меняется только сила `c`. Это существенно:
свип `λ` содержит `λ = 0`, то есть сам бейзлайн, поэтому объединение точек по `(c, λ)` доминировало
бы бейзлайн тавтологически.

## Структура

```
PLAN.md          предрегистрация: гипотезы, метрики, порядок работ
REPORT.md        отчёт
reviews/         внешние ревью плана (Fable 5, Kimi K3, GPT-5.6-sol)
kb/              база знаний: 11 разборов первоисточников + PDF
configs/         features.yaml (отобранные признаки), frozen.yaml (ручки)
src/             реализация, см. таблицу команд выше
results/         csv и графики
artifacts/       чекпойнт денойзера, карточка модели, скрипт публикации
```

## Артефакт

Лучший чекпойнт денойзера, карточка модели и скрипт публикации лежат в
[`artifacts/`](artifacts/). Публикация в открытый репозиторий Hugging Face — отдельный шаг,
выполняется владельцем аккаунта:

```
python artifacts/push_to_hf.py --repo <user>/gpt2-small-steering-denoiser
```
