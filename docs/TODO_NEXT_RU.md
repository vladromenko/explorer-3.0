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
   **Где править:** текст задачи сейчас один на профиль (`configs/sim-profiles.json` → `task`); нужен текст на эпизод:
   `explorer3/sim/capture.py` (meta, строка ~26: добавить `task` из `env.episode_parameters['object']`),
   `explorer3/sim/import_episode.py` (передать в `EpisodeWriter`), и чтение задачи в `explorer3/datasets.py`
   при сборке LeRobot-датасета (per-episode task). Без этого SmolVLA не видит знание.
3. **Виртуальная комната**: `.venv/bin/python scripts/knowledge_gap_experiment.py --rooms 10 --conditions K0,K1,K2,K3`,
   потом отдельно `--conditions K4` (Qwen, ~36 с/эпизод). Итоги скопировать в `docs/results/knowledge_gap_sim/`.
4. (`docs/RESEARCH_ROOT_RU.md` готов.) Добавить
   шаги корневого эксперимента на `/today` (`explorer3/today_pages.py`).

## B. С роботами (владелец рядом)
5. **Tank, остановка по глубине** (задеплоена, не проверена): сессия `mode=policy`, `v=0.08`, к ножкам стула;
   ждать `reason='Препятствие впереди'` ≤0.25 м, `/health` → `depth_guard.front_m`. Не больше 3 с движения.
6. **«принеси кубик»**: `curl -G -X POST localhost:8030/api/swarm/command --data-urlencode "text=принеси кубик"`;
   смотреть `GET /api/swarm/status`. Геометрический захват у пола промахивается — ждём обученный захват.

## C. Владельцу
7. Блок 3: 50 показов «ROSMASTER · взять с пола» ведущей рукой (без них захват ненадёжен).
8. Места передачи SO-101 («Рой» → «Места передачи», ~10 мин) — чтобы «принеси» доходило до человека.

## D. Усиление научной части (по обсуждению с владельцем 16.11)
Схема владельца: ROSMASTER получает «найди и привези X» → понимает, что не знает ГДЕ и КАК → зовёт Tank
(исследует, находит) и SO-101 (учится брать X) → знания/навык переходят в ROSMASTER → он выполняет.
Сейчас SO-101 только МЕРИТ предмет; «научился и передал навык» ещё не реализовано.
9. **SO-101 учится брать предмет**: пробные захваты на столе (скопировать исполнитель захвата из
   `so101_autonomous_pick_place` в `explorer3/so101_vendor/`, с его проверками безопасности; не вызывать оригинал),
   результат — точка и ориентация захвата В СИСТЕМЕ ПРЕДМЕТА + ширина губок, подтверждённые успешным подъёмом.
10. **Перенос знания в ROSMASTER**: перевод «захват в системе предмета» в позу руки ROSMASTER
    (`rosmaster_arm.pick(knowledge=…)` уже принимает ширину/высоту — добавить угол запястья и точку).
    Сравнить три варианта передачи: только размер · точка+ориентация захвата · общая модель на данных обоих
    роботов (co-training SO-101 + ROSMASTER, кросс-эмбодимент).
11. **Калиброванное «мне нужна помощь»** (идея KnowNo, Ren et al. 2023, conformal prediction): вместо порогов —
    оценка неуверенности (видимость/место/успех захвата из истории `knowledge.record_use`), порог по калибровочной
    выборке с гарантией успеха. Сравнить: никогда не звать · всегда звать · правила · Qwen · калиброванный.
12. **Цена помощи**: учитывать время/занятость помощника; главный график статьи — доля запросов помощи против
    успеха задачи (кривая), на виртуальной комнате и MuJoCo, затем 20 реальных прогонов.
13. **MuJoCo в той же логике**: эпизод = ROSMASTER без знания пробует → провал/неуверенность → «SO-101» в
    симуляции даёт знание → повтор; обучение SmolVLA на данных с текстом знания (п.2) и без.

## Состояние на конец сессии 16.11 (~03:30)
- Сделано (п.2 частично): профили `rosmaster_floor_pick_knowledge_sim` / `_noknowledge_sim`, текст задачи на эпизод
  («pick up the cookie, 6.9 x 7.1 cm, 1.3 cm high») доходит до датасета и до проверки в MuJoCo (`sim/evaluate.py`).
  Для показов берутся только предметы, которые захват берёт сверху (`GRASPABLE`: cube, bottle, block, ball).
- Не проверено: `MUJOCO_GL=egl .venv-sim/bin/python -m pytest tests/test_sim_aligned.py` (последний прогон оборвался).
- Следующее: `.venv/bin/python scripts/sim_rehearsal.py --skills rosmaster_floor_pick_knowledge_sim,rosmaster_floor_pick_noknowledge_sim --policies smolvla`
  (ночь), затем сравнить успех двух моделей.
- `docs/results/knowledge_transfer_mujoco/` перезаписывается прогоном с исправленной проверкой удержания — проверить и закоммитить.
