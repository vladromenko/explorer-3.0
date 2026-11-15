# Что доделать дальше — конкретно (16.11.2026)

Контекст и правила — в `docs/HANDOFF_KNOWLEDGE_GAP_RU.md`. Wristlab и `so101_autonomous_pick_place` не менять.
Тяжёлые прогоны — по одному: `nohup systemd-run --user --scope -p MemoryMax=6G timeout … <команда> > лог &`.
Пуш: `GH=/home/jetsonnx/.local/share/explorer3/tools/gh_2.103.3_linux_arm64/bin/gh; GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c credential.helper="!$GH auth git-credential" push -q origin main`.

## A. Без роботов (по порядку)
1. **Учитель MuJoCo для предметов разной формы** — `explorer3/sim/aligned.py`, `ScriptedFloorPick`.
   Сейчас (`docs/results/knowledge_transfer_mujoco/summary.json`) кубик 10/10, печенье 0/9, носок 1/16, всего 30%.
   Причины смотреть по эпизоду (`env.reset(seed, options={'object': 'cookie'})`): высота захвата, IK у пола,
   раскрытие захвата 30 vs ширина, ориентация (yaw предмета → поворот запястья, сустав 5).
   Цель: с полным знанием ≥80% на всех предметах; без знания (кубик 32 мм по умолчанию) заметно хуже.
   Затем `.venv-sim/bin/python scripts/knowledge_transfer_mujoco.py --episodes 60`, коммит результатов.
2. **Обучение со знанием** (разрешено владельцем): профиль `rosmaster_floor_pick_knowledge_sim` в
   `explorer3/sim/factory.py` = `partial(RosmasterFloorPick, vary_objects=True)`, эксперт со знанием,
   текст задачи с размером: «pick up the cookie, 7.0 x 7.0 cm, 1.2 cm high»; контроль — тот же текст без размера.
   Сбор показов и обучение SmolVLA — через существующие jobs (как `rosmaster_floor_pick_sim`), по одному за ночь.
   Сравнение: успех на одних seed, `scripts/benchmark_stats.py` для интервалов.
3. **Виртуальная комната**: `.venv/bin/python scripts/knowledge_gap_experiment.py --rooms 10 --conditions K0,K1,K2,K3`,
   потом отдельно `--conditions K4` (Qwen, ~36 с/эпизод). Итоги скопировать в `docs/results/knowledge_gap_sim/`.
4. `docs/RESEARCH_ROOT_RU.md` (вопрос, K0–K4, MuJoCo, уровни доказательства, что делает владелец) и
   шаги корневого эксперимента на `/today` (`explorer3/today_pages.py`).

## B. С роботами (владелец рядом)
5. **Tank, остановка по глубине** (задеплоена, не проверена): сессия `mode=policy`, `v=0.08`, к ножкам стула;
   ждать `reason='Препятствие впереди'` ≤0.25 м, `/health` → `depth_guard.front_m`. Не больше 3 с движения.
6. **«принеси кубик»**: `curl -G -X POST localhost:8030/api/swarm/command --data-urlencode "text=принеси кубик"`;
   смотреть `GET /api/swarm/status`. Геометрический захват у пола промахивается — ждём обученный захват.

## C. Владельцу
7. Блок 3: 50 показов «ROSMASTER · взять с пола» ведущей рукой (без них захват ненадёжен).
8. Места передачи SO-101 («Рой» → «Места передачи», ~10 мин) — чтобы «принеси» доходило до человека.
