# Переиспользование научных компонентов

Состояние реализации; пустые проверки не обозначают готовность.

| Компонент | Тип | Версия / лицензия | Использование и проверка | Ограничение |
|---|---|---|---|---|
| Menagerie robotstudio_so101 | upstream | `370c439…`; Apache-2.0 | `sim.models.so101_model`, настоящие шаги, камера, геометрический перенос кубика | Локальная калибровка не подтверждена |
| MuJoCo Studio WebViewer | upstream | 4.0.1; Apache-2.0 | `sim.worker`: `launch_web.launch`, `ViewerHandle.set_model/sync`; Playwright видит canvas и управление | SO/ROS arm/оба шасси UI проверены; геометрия ROS из официального архива; общий мир с тремя scripted задачами проверен; совместное исполнение двух сохранённых ACT и exact replay проверены |
| Playground SO101PickCube + RSL-RL | upstream | Playground 1.0.1 / Apache-2.0; RSL-RL 5.4.2 / BSD-3-Clause | Официальный `mujoco_playground.rl.rsl_rl.train`: CPU step/train/save/new-process reload | Пока smoke; lift ≠ place |
| MuJoCo/Gymnasium + SB3 PPO | upstream integration | MuJoCo 4.0.1; Gymnasium 1.4.0 / MIT; SB3 2.9.2 / MIT | `sim.ppo` использует официальный PPO на native dynamics | 4 конечных прогона по 16384 шага выполнены на ранних моделях; 0/20 у каждого; не навык |
| Playground sysid | upstream + adapted data loader | 1.0.1; Apache-2.0 | `sim.sysid`: оригинальные `build_parameters`, `to_model_sequences`, MuJoCo optimizer; 2 реальные траектории fit, 1 отложенная | 3 итерации; hold-out residual ухудшился 0.00566 → 0.00595 rad; параметры не применены. Время записи UART — proxy, не MCU acknowledgement; нули суставов требуют проверки |
| Fleet-DAgger | upstream subset + ported integration | `ifl_benchmark` d35f0f8; MIT/Ryan Hoque | `vendor/fleet_dagger` — точные Random/TD и assign_humans; `fleet_allocation.FleetAllocator`; 20 шагов conformance каждого метода против отдельного исходника PASS | Один оператор; pure allocation и один оператор в общей сцене проверены; online запросы, bounded corrections и ручная передача в следующий round подключены; новые training rounds не запускались; TD получает risk/novelty извне; C.U.R. не заявлен |
| Visual/dynamics randomization | protocol_adaptation | Tobin 1703.06907 / Peng 1710.06537; собственный код | `sim.variation`, seeded параметры и snapshot/restore; контакты nominal/varied проверены без обучения | Диапазоны assumed, эффект на качество ACT пока не измерен |
| AutoIntervene-adapted | ported_algorithm, partial | 2608.07065v1 §III; собственный код | `autointervene.Memory/SupportMonitor/calibrate`; настоящий ACT выдаёт frozen features и chunk; queue/snapshot/reload проверены без обучения | Математика и дисковый roundtrip проверены на явно маркированных fixtures; подготовка памяти/порогов из frozen plan подключена к UI; online switching в общем мире проверен на fixtures; на новых trained/held-out данных не запускалась. Использован ACT ResNet18 вместо авторского backbone; не воспроизведение чисел статьи |
| RESCUE/Kintsugi branch protocol | protocol_adaptation, partial | собственная интеграция, не воспроизведение статьи | `sim.branching`: три сопоставимые ветви, бюджеты, повтор и импорт проверенного сегмента | Browser/replay/readback PASS; не исследование эффекта человека |
| DART-inspired | unverified | дополнительная серия | Не запускался | После основных сценариев |

Артефакты текущей проверки: runtime `validation/corrective-a/`. Три scripted-expert записи — именно исполнения MuJoCo с контактами, не fixtures и не человеческие демонстрации. После 50 шагов ACT save/reload прошли, три rollout на seeds 10000–10002 завершились failure/task_deadline. Это проверенный тракт с отрицательным результатом короткого обучения.

ROHE реализована по формуле (2) статьи Fleet-DAgger: (M/N) × суммарная награда / (1 + сумма квадратов Frobenius allocation). `plotting/plot.py` в авторском репозитории строит также penalized-reward график — он не подменяет определение статьи. «Успешные задачи / минуты помощи», включая switch/reset, выводится отдельно; simulated expert не считается измеренным временем человека.

Новые модели: `policy_catalog`, `training.train_command`, `sim.policy_worker` используют официальный LeRobot 0.6.1 ACT/SmolVLA/Diffusion. SmolVLA base revision5e8d12a… и SmolVLM revision7b375e… закреплены. SmolVLA inference/snapshot на имеющемся synthetic checkpoint проверен, качество навыка не заявляется; новые обучения запрещены пользователем. YOLOE и SAM3 — отдельный perception слой, не замена моторной политике.

## Агентный слой · 14.11.2026

Это работающая адаптация протоколов поверх существующих компонентов. Перенос
чужого исходного кода и адаптация метода различаются явно. Научное превосходство
или результаты обучения из приведённых статей проекту не приписываются.

| Метод / первичный источник | Код и лицензия источника | Реальная точка интеграции | Граница |
|---|---|---|---|
| [RoboOS](https://arxiv.org/abs/2505.03673), [RoboOS-NeXT](https://arxiv.org/abs/2510.26536) | [FlagOpen/RoboOS](https://github.com/FlagOpen/RoboOS), Apache-2.0 | `cognitive.registry`, типизированный TaskGraph, роли, broker и временные evidence: план отделён от роботных навыков и памяти | Архитектурная адаптация, не установка RoboBrain/облачного RoboOS. Никакого объявления новой архитектуры |
| [PhysMem](https://arxiv.org/abs/2602.20323) | [verification.py](https://github.com/haoyangli16/PhysMem/blob/main/physmem/learning/verification.py), [hypothesis.py](https://github.com/haoyangli16/PhysMem/blob/main/physmem/core/hypothesis.py); MIT согласно README | `store.add_case/verify_case`: опыт → candidate → независимые episode IDs → supported_in_scope / contradicted; тесты исключают самоподтверждение и повторный подсчёт | Адаптация логики verification/promotion с более строгой привязкой артефактов. Upstream исходники не скопированы, вероятности уверенности не выдумываются, raw experience не удаляется |
| [RoboMemory](https://arxiv.org/abs/2508.01415) | [SP4595/RoboMemory-Code](https://github.com/SP4595/RoboMemory-Code), Apache-2.0 | Разделение evidence/facts/events/cases; FTS5 retrieve + замороженный снимок памяти реально добавляет выбор прямого наблюдения вместо ограниченного поиска | Адаптация memory→planning цикла; тяжёлые embedding/VLM сервера статьи не устанавливаются; это не полный upstream benchmark |
| [ActiveAlign / Look Before You Act](https://arxiv.org/abs/2610.23378) | Статья, CC BY 4.0 по arXiv; код не перенесён | `run.execute_graph`: свежий вид цели допускает продолжение, но успех камеры, успех получателя и общий исход измеряются отдельно; сохраняются оба эпизода и совместимость камеры/политики/геометрии | Протокольная адаптация проверки продолжения; VLA adaptation/training запрещены и не выполнены |
| [smolagents ToolCallingAgent](https://huggingface.co/docs/smolagents/v1.26.0/en/reference/agents) | `smolagents==1.26.0`, Apache-2.0 | `cognitive.model`: настоящий ToolCallingAgent, единственный закрытый proposal tool, максимум две генерации с одной коррекцией формата | CodeAgent отсутствует; ToolCallingAgent не исполняет моторные/обучающие команды |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Apache-2.0, revision `89644892e4d85e24eaac8bacfd4f463576704203`, hashes в `configs/cognitive-assets.lock.json` | Локальный inference на Homebrain, исходный checkpoint без обновления весов | Первый тракт получает измерения роботных RGB как текст; прямое VLM понимание изображения этим тестом не заявляется |

`sim.cooperation` использует один прежний FleetWorld, тот же ArmModel IK и DockExpert.
Детектор — явно названный OpenCV color/shape baseline только для фиксированного
оранжевого маркера в номинальной сцене; он не получает simulator segmentation IDs
или истинную позу цели. DockExpert остаётся privileged scripted teacher. Истинные
симуляционные distance/yaw/stable используются отдельно для оценки исхода.
