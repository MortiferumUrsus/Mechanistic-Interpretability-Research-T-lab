# Ремонт стиринга по признакам SAE в GPT-2 small

Стиринг языковой модели вдоль направления из декодера SAE, `h̃ = h + αv`, усиливает признак, но при
больших `α` ломает текст. Этот репозиторий изучает, как дёшево уменьшить это повреждение, в четырёх
раундах: конструкция из задания (денойзер после интервенции) и три её починки; обученная поправка
направления инъекции; воспроизведение на отложенных признаках; разбор того, что обученная поправка
делает на самом деле. Отчёт: [`REPORT.ru.md`](REPORT.ru.md) (английская версия:
[`REPORT.md`](REPORT.md)), он же одним HTML-файлом: `report.ru.html` (английский — `report.html`).
Предрегистрация: [`PLAN.ru.md`](PLAN.ru.md) (английская версия: [`PLAN.md`](PLAN.md)). Исходное
задание: [`TASK.md`](TASK.md) (перевод на английский: [`TASK.en.md`](TASK.en.md)).

Работа начиналась как решение тестового задания T-Lab 2026 (трек Mechanistic Interpretability);
четвёртый раунд добавлен после сдачи.

## Результат в четырёх строчках

1. **Денойзер стирает сигнал.** Денойзер, обученный возвращать активации на корпусное
   многообразие, удаляет от 24 до 62% стиринг-вектора, что согласуется с предсказанием в замкнутой
   форме, а любая post-hoc починка активации увеличивает нелинейную часть downstream-отклика. Ни
   одно плечо не превосходит наивный стиринг (отчёт, §2–§8).
2. **Обученная поправка направления доставляет больше концепта.** Отображение ранга 64
   `w(v) = normalise(v + M v)` (98k параметров), обученное на одном наборе признаков и применённое
   к другим при той же норме возмущения, поднимает пре-зарегистрированный endpoint на TEST с 0.534
   до 0.776 и воспроизводится на 12 отложенных признаках и на 48 свежих (§9, §10.8).
3. **Её выигрыш по перплексии на сгенерированном тексте в основном артефакт.** Половина поправки —
   одно общее направление, которое снижает энтропию следующего токена модели. Инжектированное
   отдельно, оно снижает Pythia log-PPL сильнее, чем полная поправка, при нулевом концепте и при
   этом ухудшает предсказание настоящего текста моделью; приставленное к столбцу декодера вообще
   без обучения, оно доставляет столько же концепта, сколько обученная поправка, или больше.
   Перплексия генераций под внешней моделью-оценщиком не является мерой связности, если интервенция
   способна менять уверенность модели (§10.2–§10.6).
4. **На настоящем тексте при равной доставке концепта обученная поправка всё же стоит дешевле
   наивного стиринга** — на 0.07–0.6 ната NLL при teacher forcing, в зависимости от того, как сшиты
   две кривые, и с широкими интервалами: измерение (четыре признака) устанавливает знак, а не
   величину. Вариант со штрафом на энтропию уменьшает общую компоненту и оставляет это сравнение в
   пределах шума (§10.6, §10.7).

## Постановка

GPT-2 small (TransformerLens), интервенция в `blocks.6.hook_resid_post`. Этот тензор побитово
совпадает с `blocks.7.hook_resid_pre`, на котором обучены SAE релиза `gpt2-small-res-jb`, поэтому
столбцы декодера являются стиринг-направлениями прямо в базисе интервенции, без всякой смены базиса
(тождество проверяется командой `activations.py verify`).

Сила параметризуется приращением концепт-координаты `s = c · natural_s(f)`, где
`natural_s(f) = ceiling_f · ‖W_dec[f]‖` — сильнейшая собственная активация признака на корпусе
(`src/common.py`, `natural_strength`). Естественные масштабы признаков различаются в 6.3 раза
(`natural_s` = 11.12…70.60, `results/feature_scales.csv`), и общая единица вроде
`median‖h‖ = 88.23` эту разницу скрыла бы: в ней фактические силы составляют 0.126…0.800 от `c`.
Поэтому сетка `c` сравнима между признаками, а плечи сопоставляются при равной силе концепта.

Связность измеряется прокси — log-перплексией продолжения под Pythia-410m (перплексия самой
порождающей модели вырождена: сломанная модель уверена в своём выводе), плюс dist-1/2/3, доля
повторённых 4-грамм и prompt dependence; концепт — долей попаданий по автоматически выведенному
словарю признака и активацией этого признака в SAE на сгенерированном тексте. Четвёртый раунд
добавляет измерение на тексте, который модель не писала: NLL при teacher forcing и top-1 точность
на отложенных документах под тем же стиринг-хуком (`src/capability_sweep.py`).

### Плечи

| плечо | что инжектирует или делает | обучение |
|---|---|---|
| `clean` | `h` | — |
| `naive` | `h + s·v̂` | — |
| `norm_preserving` | масштабирует результат обратно к `‖h‖` | — |
| `denoise_naive` | `x + η(D(x) − x)` — буквальная формулировка задания | денойзер |
| `cds` | `x + λ·P⊥(D(x) − D(h) − s·v̂)` — контрастная коррекция с сохранением направления | денойзер |
| `mts` | сдвиг минимальной махаланобисовой нормы при том же приращении концепт-координаты | — |
| `fsr` | кламп не-целевых латентов SAE под их корпусный потолок | — |
| `dirfix` | `h + s·w(v̂)` — обученная поправка направления при наивной норме | поправка направления (раунд 2) |
| `randrot` | контроль: поворот `v̂` на тот же угол в случайном направлении | — |
| `shared` | `h + s·normalise(v̂ + κ·d̄)`, где `d̄` — общая компонента обученной поправки | — (раунд 4) |
| `shared_only` | контроль: `h + s·d̄`, без направления признака | — |
| `residual`, `purified`, `centred`, `diffmeans`, `diffmeans_purified`, `antimanifold`, `rotate` | семейства направлений раунда 4, см. §10.1 отчёта | — |
| `cond_wiener`, `cond_denoise` | условные денойзеры, тянущие к собственному многообразию концепта (провал, §10.9) | условный денойзер |

Первый раунд (`clean` … `fsr`) объявлен до запусков. `dirfix` — второй раунд: метод выбран после
того, как результаты первого были просмотрены, и это оговорено отдельно. Поправка обучается только
на FIT-признаках и применяется к признакам, которых не видела; норма её возмущения совпадает с
наивной по построению. Третий раунд — то же плечо на двенадцати свежих признаках (`test_r3`),
удержанных вне обучения поправки, плюс контроль случайного поворота. Четвёртый раунд — плечи ниже
`randrot` в таблице, чекпойнт со штрафом на энтропию `dir_ent`, свип способностей на настоящем
тексте, свипы по сидам и рангам, 48 свежих признаков (`test_r4`) и воспроизведение на слое 10.

## Установка

```
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Порядок важен: `transformer-lens` и `sae-lens` могут заменить CUDA-сборку torch CPU-сборкой,
поэтому CUDA-сборка ставится последней. Раунды 1–3 выполнялись на GTX 1660 Ti (6 ГБ), раунд 4 — в
сессиях Kaggle T4/P100 с версиями, закреплёнными в `requirements.txt`. Точный lock-файл не
сохранялся, поэтому это рабочий рецепт установки, а не бит-в-бит идентичное окружение.

## Воспроизведение

Раунды 1–3, из корня репозитория. У каждого `scripts/*.sh` есть PowerShell-двойник `scripts/*.ps1` —
именно им выполнены прогоны, попавшие в отчёт; если интерпретатор называется не `python`, задайте
`PY`.

```
scripts/run_all.sh        # данные, отбор признаков, денойзеры (60000 шагов), один проход на TEST, механизм, раунды 2 и 3
scripts/run_round2.sh     # только раунд 2: обучить поправку, выбрать ручки на DEV, один проход на TEST
scripts/run_round3.sh     # только раунд 3: зарезервировать holdout из FIT, переобучить без него, затем генерация
```

По стадиям:

```
cd src

# данные: тождество хуков, дамп активаций, промпты, потолки латентов SAE
python activations.py verify
python activations.py dump --n-tokens 2000000 --batch-size 16
python activations.py prompts --n-prompts 40 --prompt-len 8
python sae_stats.py --chunk 4096

# отбор признаков и разбиение FIT / DEV / TEST
python features.py select --seed 0

# денойзеры, обучаются только на FIT-направлениях; замороженный вариант идёт первым
python train_denoiser.py --arch mlp --noise gauss --cond 1 --seed 0 --steps 60000
python train_denoiser.py --arch mlp --noise mix --cond 1 --seed 0 --steps 60000
python train_denoiser.py --arch linear --noise mix --cond 0 --seed 0 --steps 60000

# все инференс-ручки выбираются на DEV, без генерации
python sweep_dev.py
python select_from_dev.py         # пишет configs/frozen.yaml по DEV-оценкам; TEST генерируется только после этого

# один проход на TEST с замороженными ручками
python generate.py --split test --out gen_test.jsonl
python metrics.py --gen gen_test.jsonl --out scored_test.csv --stages ppl,keyword,sae,dist
python pareto.py --scored scored_test.csv --concept keyword_hit

# механизм
python analysis.py transmission
python analysis.py spectral
python analysis.py surgery
python analysis.py causal --lam 1.5 --shrink 0.01
python analysis.py predictors
python plots.py --scored scored_test.csv --concept keyword_hit   # results/fig_*.png, встроены в отчёт
python make_html.py                                             # REPORT.md -> report.html
python dirfix_vs_naive.py       # A/C и перплексия в одной таблице, её цитирует §9.1 отчёта
python report_numbers_check.py  # каждое число отчёта должно быть в файле, на который ссылается его абзац
```

Порядок шагов внутри третьего раунда существенен: сначала `select_round3.py` резервирует двенадцать
holdout-признаков из FIT, затем `train_direction.py` переобучает поправку уже без них (23988
направлений вместо 24000), и только после этого идут генерация, метрики и контроль случайного
поворота. Если переставить шаги, поправка окажется обученной на собственном holdout.

**Четвёртый раунд** выполнялся на Kaggle драйверами `scripts/run_exp*.py` и `scripts/run_layer.py`
(стадии выбираются ключом `--stages`; каждая стадия обёрнута так, что одна ошибка не останавливает
очередь, — а значит зелёная сводка очереди не доказывает, что стадия произвела свой артефакт).
Обвязка Kaggle лежит в `kaggle/` (см. `kaggle/SMOKE.md`). Те же стадии запускаются локально из
`src/`:

```
cd src
python anatomy.py --ckpt dir_hot                # общее направление d̄ -> checkpoints/shared_direction.pt, таблицы анатомии
python direction_report.py                      # диагностика по развёртке для каждого семейства направлений
python qq_test.py --n-docs 500 --ctx 512 --c 1.0 --features 3 --split test_r3 --out-prefix qq
python generate.py --split test_r3 --arms naive,dirfix,shared,shared_only,residual,antimanifold --c-grid 0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --out gen_expA_r3.jsonl
python concept_data.py mine                     # data/concept_positions.npz
python concept_data.py stats                    # data/concept_stats.pt, нужен плечам diffmeans
python generate.py --split test_r3 --arms naive,dirfix,shared,diffmeans,centred,purified,diffmeans_purified,rotate --c-grid 0,0.5,1.0,1.5,2.0,3.0 --out gen_expF_r3.jsonl
python capability_sweep.py                      # NLL и top-1 на настоящем тексте под стиринг-хуком -> capability_sweep.csv
python capability_sweep.py --set direction=dir_ent --out capability_ent.csv
python capability_sweep.py --arms shared --out capability_kappa_0.25.csv   # после установки kappa_shared: 0.25 в configs/expA.yaml (и так же для 0.5)
python matched_concept_capability.py --bootstrap 2000            # повреждение настоящего текста при равной доставке концепта
python matched_concept_capability.py --first-n-features 4 --bootstrap 2000 --out matched_concept_capability_4feat.csv
python matched_concept_capability.py --scored scored_expF_ent_clean.csv --capability capability_ent.csv --arm-map dirfix=dir_ent --bootstrap 2000 --out matched_concept_capability_ent.csv
python paired_by_strength.py --scored scored_expA_r3.csv --out paired_expA_r3_by_strength.csv
python dbar_cosines.py                          # results/shared_direction_cosines.csv
python matched_repetition.py --scored scored_expF_r3.csv --out-prefix matchedF
python feature_distribution.py --scored scored_expF_r3.csv --out-prefix featdistF
python layer_profile.py                         # норма по слоям, проекция на d̄, энтропия на выходе
python train_direction_ent.py train --rank 64 --gamma 1.0 --lr 3e-3 --steps 2000 --name dir_ent --ent-weight 1.0
python anatomy.py --ckpt dir_ent --tag _ent
python select_round4.py --seed 4242 --n 48      # пишет configs/features_r4.yaml; features.yaml заморожен
python train_direction.py train --rank 64 --gamma 1.0 --lr 3e-3 --steps 2000 --name dir_r4 --exclude-r4
```

`TLAB_LAYER=10` переводит весь конвейер на SAE в `blocks.10.hook_resid_pre`
(`scripts/run_layer.py --layer 10` проходит всю цепочку на одноразовой копии репозитория).

Проверки, которые проходят без GPU на свежем клоне: `python test_pareto.py` (инварианты машинерии
фронта), `python report_numbers_check.py` (каждое число отчёта есть в артефакте, на который
ссылается его абзац) и валидатор пакета разметки ниже. `python test_arms.py` проверяет инварианты
плеч, но ему нужен `data/act_stats.pt`, поэтому он запускается только после `activations.py dump`.

## Контроль утечки

Индексы признаков SAE разбиты на три непересекающихся множества. `FIT` (24000 признаков) участвует
только в синтезе обучающих возмущений, `DEV` (6) — в выборе всех гиперпараметров, `TEST` (12)
открывается один раз. Любой признак с `|cos| ≥ 0.3` к любому направлению из `DEV ∪ TEST` исключён
из `FIT`, фактический максимум по всем парам — в [`results/leakage.csv`](results/leakage.csv).
Двенадцать признаков третьего раунда дополнительно исключены из `FIT`, и поправка переобучена без
них; 48 признаков четвёртого раунда и все FIT-направления с `|cos| ≥ 0.3` к ним исключены из
чекпойнта, использованного в раунде 4 (`dir_r4`). Промпты для оценки взяты из другого шарда
корпуса, чем активации для обучения.

Одну слабость надо назвать явно. Промпты между DEV и TEST не разделены, хотя `PLAN.md` этого
требовал: `sweep_dev.py` берёт первые десять промптов `data/prompts.json`, `generate.py` — первые
тридцать, то есть все DEV-промпты входят в TEST, и треть TEST-промптов была видна при выборе
инференс-ручек. В раунде 4 для прогона на 48 признаках использован отдельный пул промптов;
построчная таблица его оценок не сохранена, поэтому это разделение зафиксировано только в логах
ядра (отчёт, §10.8).

Все инференс-ручки (`λ`, `η`, `k`, усадка ковариации, `σ`) фиксируются на DEV и записываются в
`configs/frozen.yaml` до первого запуска на TEST, на TEST меняется только сила `c`. Это
существенно: свип `λ` содержит `λ = 0`, то есть сам бейзлайн, поэтому фронт, взятый как объединение
точек по `(c, λ)`, доминировал бы бейзлайн тавтологически.

## Post-hoc разметка сохранённых генераций

У `metrics.py` есть экспериментальная стадия `judge` (LLM-судья Qwen2.5-1.5B-Instruct). Она не
запускалась и не участвует ни в одном числе отчёта. Вместо неё после завершения экспериментов один
AI-аннотатор в отдельном контексте оценил 468 сохранённых генераций третьего раунда в 180
сопоставимых ячейках, вслепую к соответствию «плечо — текст», по рубрике, замороженной до разметки.
Это не человеческая оценка, и второго аннотатора нет. Для `dirfix − naive` оценка по
`target success` равна `+0.978 [0.567, 1.378]`, а по `degeneration control` —
`−0.522 [−0.856, −0.200]`; интервалы для `coherence` и `overall quality` накрывают нуль. Метки,
протокол и ограничения:
[`evaluation/ai_annotation/summary.md`](evaluation/ai_annotation/summary.md).
Детерминированные части проверяются из корня репозитория командой

```bash
python evaluation/ai_annotation/validate_package.py --repo-root . --annotation-dir evaluation/ai_annotation
```

Ожидаемые финальные строки: `VALIDATION PASS`, 468 уникальных меток, 180 сопоставимых ячеек,
`warnings: []`. Три файла, которые валидатор хеширует как критические источники
(`configs/features.yaml`, `results/gen_r3.jsonl`, `results/gen_r3_ctrl.jsonl`), хранятся ровно в тех
байтах, по которым считались хеши, включая переводы строк CRLF, и помечены `-text` в
`.gitattributes`, чтобы любой клон воспроизводил их.

## Структура репозитория

```
README.md, README.ru.md         этот файл, английская (основная) и русская версии
REPORT.md, REPORT.ru.md         отчёт; report.html / report.ru.html — рендеринг одним файлом
PLAN.md, PLAN.ru.md             предрегистрация (гипотезы, метрики, порядок работ); русский файл — замороженный оригинал
TASK.md, TASK.en.md             задание в исходном виде (русский, дословно, хешируется пакетом разметки) и его перевод
configs/                        features.yaml (замороженные разбиения), features_r4.yaml (признаки раунда 4), frozen.yaml (ручки),
                                expA.yaml (вес смешивания для плеча shared), expF.yaml (сетка его свипа)
src/                            реализация; см. таблицы команд выше
scripts/                        конвейеры: *.ps1 — как выполнялись прогоны отчёта, *.sh — их зеркала, run_exp*.py и run_layer.py для раунда 4
kaggle/                         сборка payload, генератор ядер, помощники push/fetch, smoke-тест
results/                        csv, json, графики; таблицы раунда 4 названы по своему драйверу (expA, expF, expG, layer10);
                                results/kaggle_runs/ хранит сводку очереди каждого ядра, заметки и небольшие выходы
checkpoints/                    денойзеры, dir_hot (опубликован), dir_ent, dir_r4, shared_direction*.pt, seeds/ (свип ранг × сид)
artifacts/                      опубликованные чекпойнты с автономным loader, карточка модели (README.md / README.ru.md) и скрипт публикации
evaluation/                     post-hoc разметка одним аннотатором с item-level labels и валидатор
kb/                             заметки по литературе и PDF
reviews/                        внешние рецензии на план v1 (на русском)
```

Файлы результатов раунда 4 и что в них:

| файл | содержимое |
|---|---|
| `results/anatomy_summary.json`, `_ent.json`, `_r4.json` | анатомия `dir_hot`, `dir_ent`, `dir_r4`; по-признаковая таблица для `dir_ent` — в `anatomy_per_feature_ent.csv` |
| `results/direction_report.csv` | косинус с `d̄`, энергия в слабой половине спектра и дотягивание до логитов по семействам направлений и признакам |
| `results/layer_profile.csv` | норма резидуала по слоям, проекция на `d̄`, энтропия на выходе |
| `results/qq_summary.csv`, `qq_tokens_*.csv` | потокенная диагностика ActAdd на настоящем тексте |
| `results/scored_expA_r3.csv`, `paired_expA_r3.csv` | `shared`, `shared_only`, `residual`, `antimanifold` (κ = 0.75) |
| `results/scored_expF_r3.csv`, `paired_expF_r3.csv` | `shared`, `diffmeans`, `centred`, `purified`, `diffmeans_purified`, `rotate` (κ = 1.0) |
| `results/scored_expF_kappa.csv`, `capability_kappa_*.csv` | свип по `κ` для `shared` |
| `results/capability_sweep.csv`, `capability_ent.csv` | NLL и top-1 на настоящем тексте в зависимости от силы |
| `results/matched_concept_capability.csv`, `_ent.csv` | повреждение настоящего текста при равной доставке концепта |
| `results/scored_expF_ent_clean.csv`, `paired_expF_ent.csv` | поправка со штрафом на энтропию `dir_ent` |
| `results/matchedA_*.csv`, `matchedF_*.csv`, `featdist*_*.csv` | проверки при равных повторах и распределения по признакам |
| `results/scored_layer10.csv`, `paired_layer10.csv` | воспроизведение на слое 10 |
| `results/r4_identity_at_zero.csv` | каждое плечо раунда 4 при `c = 0` является тождеством |
| `results/kaggle_runs/tlab-mi-a-d/` | свип по сидам и рангам (`expD_eval_dev.csv`, логи обучения) и анатомия `dir_hot` |
| `results/kaggle_runs/tlab-mi-e4-score/` | прогон на 48 признаках: сводка по endpoint и парная таблица |
| `results/kaggle_runs/tlab-mi-g-e/` | прогон условного денойзера |
| `results/kaggle_runs/tlab-mi-layer10/` | воспроизведение на слое 10: анатомия на слое 10 и косинус её `d̄` с `d̄` слоя 7 |
| `results/kaggle_runs/tlab-mi-e3/` | отбор признаков раунда 4 и первый лог обучения `dir_r4` |
| `results/kaggle_runs/tlab-mi-layer4/` | разошедшийся прогон на слое 4 |

## Публикация чекпойнта

В `artifacts/` лежат `direction_correction.pt` (поправка раунда 2, идентична
`checkpoints/dir_hot.pt`), `dir_ent.pt` (вариант со штрафом на энтропию), `denoiser.pt`,
`config.json`, `model.py` (автономный loader) и карточка модели. Публикация на Hugging Face —
ручной шаг под учётной записью владельца:

```
hf auth login
python artifacts/push_to_hf.py --repo <user>/gpt2-small-steering-direction-correction --dry-run
python artifacts/push_to_hf.py --repo <user>/gpt2-small-steering-direction-correction
```

`--dry-run` печатает список файлов и целевой репозиторий, ничего не загружая.
