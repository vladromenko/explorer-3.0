# Переиспользование научных компонентов

Состояние реализации; пустые проверки не обозначают готовность.

| Компонент | Тип | Версия / лицензия | Использование и проверка | Ограничение |
|---|---|---|---|---|
| Menagerie robotstudio_so101 | upstream | `370c439…`; Apache-2.0 | `sim.models.so101_model`, настоящие шаги, камера, геометрический перенос кубика | Локальная калибровка не подтверждена |
| MuJoCo Studio WebViewer | upstream | 4.0.1; Apache-2.0 | `sim.worker`: `launch_web.launch`, `ViewerHandle.set_model/sync`; Playwright видит canvas и управление | SO/ROS arm/оба шасси UI проверены; геометрия ROS из официального архива; общий мир с тремя scripted задачами проверен; learned fleet ещё не готов |
| Playground SO101PickCube + RSL-RL | upstream | Playground 1.0.1 / Apache-2.0; RSL-RL 5.4.2 / BSD-3-Clause | Официальный `mujoco_playground.rl.rsl_rl.train`: CPU step/train/save/new-process reload | Пока smoke; lift ≠ place |
| MuJoCo/Gymnasium + SB3 PPO | upstream integration | MuJoCo 4.0.1; Gymnasium 1.4.0 / MIT; SB3 2.9.2 / MIT | `sim.ppo` использует официальный PPO на native dynamics | 4 конечных прогона по 16384 шага выполнены на ранних моделях; 0/20 у каждого; не навык |
| Playground sysid | upstream + adapted data loader | 1.0.1; Apache-2.0 | `sim.sysid`: оригинальные `build_parameters`, `to_model_sequences`, MuJoCo optimizer; 2 реальные траектории fit, 1 отложенная | 3 итерации; hold-out residual ухудшился 0.00566 → 0.00595 rad; параметры не применены. Время записи UART — proxy, не MCU acknowledgement; нули суставов требуют проверки |
| Fleet-DAgger | upstream subset + ported integration | `ifl_benchmark` d35f0f8; MIT/Ryan Hoque | `vendor/fleet_dagger` — точные Random/TD и assign_humans; `fleet_allocation.FleetAllocator`; 20 шагов conformance каждого метода против отдельного исходника PASS | Один оператор; pure allocation и один оператор в общей сцене проверены; interactive learning rounds ещё не подключены; TD получает risk/novelty извне; C.U.R. не заявлен |
| Visual/dynamics randomization | protocol_adaptation | Tobin 1703.06907 / Peng 1710.06537; собственный код | `sim.variation`, seeded параметры и snapshot/restore; контакты nominal/varied проверены без обучения | Диапазоны assumed, эффект на качество ACT пока не измерен |
| AutoIntervene-adapted | ported_algorithm, partial | 2608.07065v1 §III; собственный код | `autointervene.Memory/SupportMonitor/calibrate`; настоящий ACT выдаёт frozen features и chunk; queue/snapshot/reload проверены без обучения | Математика и дисковый roundtrip проверены на явно маркированных fixtures; подготовка памяти/порогов из frozen plan подключена к UI; на новых trained/held-out данных не запускалась, полного round ещё нет. Использован ACT ResNet18 вместо авторского backbone; не воспроизведение чисел статьи |
| RESCUE/Kintsugi branch protocol | protocol_adaptation, partial | собственная интеграция, не воспроизведение статьи | Native snapshot/restore уже детерминирован | Три сопоставимые ветви/бюджеты ещё не завершены |
| DART-inspired | unverified | дополнительная серия | Не запускался | После основных сценариев |

Артефакты текущей проверки: runtime `validation/corrective-a/`. Три scripted-expert записи — именно исполнения MuJoCo с контактами, не fixtures и не человеческие демонстрации. После 50 шагов ACT save/reload прошли, три rollout на seeds 10000–10002 завершились failure/task_deadline. Это проверенный тракт с отрицательным результатом короткого обучения.

ROHE реализована по формуле (2) статьи Fleet-DAgger: (M/N) × суммарная награда / (1 + сумма квадратов Frobenius allocation). `plotting/plot.py` в авторском репозитории строит также penalized-reward график — он не подменяет определение статьи. «Успешные задачи / минуты помощи», включая switch/reset, выводится отдельно; simulated expert не считается измеренным временем человека.

Новые модели: `policy_catalog`, `training.train_command`, `sim.policy_worker` используют официальный LeRobot 0.6.1 ACT/SmolVLA/Diffusion. SmolVLA base revision5e8d12a… и SmolVLM revision7b375e… закреплены. SmolVLA inference/snapshot на имеющемся synthetic checkpoint проверен, качество навыка не заявляется; новые обучения запрещены пользователем. YOLOE и SAM3 — отдельный perception слой, не замена моторной политике.
