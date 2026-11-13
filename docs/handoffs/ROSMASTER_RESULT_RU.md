# ROSMASTER — результат отдельного направления (ведётся по ходу)

Задание: docs/handoffs/ROSMASTER_NEXT_DIALOG_RU.md (принадлежит основному диалогу). Глобально stage=A, hardware_connections=false; аппаратных сессий без подтверждения пользователя не открывалось.

## 2026-11-13 16:50 — исходное состояние (только чтение, SSH)
- ROSMASTER explorer 192.168.0.177, uptime 4 ч 22 мин; батарея 12.2 В.
- Владелец serial /dev/ttyUSB0: прежний rosmaster-bridge (PID 2507); explorer-control/watchdog/camera active; explorer3-rosmaster-agent и -camera inactive; rosmaster-console (8090) inactive.
- Через прежний мост /arm_servo_state: все 6 значений конечные (NaN нет), ~390 сообщ./с; поза сложена [92.7, 193.2, -43.2, 15.7, 89.4, 33.0].
- Развёрнутый агент совпадал с репозиторием (runtime 64217e53, leased_core 9c50830c, control_adapter 227b5b7e, agent 3c0a0329).

## 2026-11-13 17:05 — обрыв «Controller rejected ARM: EXPIRED»
Журнал ~/explorer3/logs/control-sessions.jsonl (сессия 185 с, 5522 команды): 29 отмен руки контроллером при НЕПОДВИЖНОЙ руке, удерживаемой на нижнем пределе локтя (цель 0°, измерено −0.2°), затем ARM отклонён как EXPIRED → LeasedCore отключал сессию (прощался только NOT_READY).
Исправлено (без ослабления STOP/watchdog):
- цели руки держатся на 1° внутри пределов (NativeIO.move_arm, позы, выравнивание зеркала);
- EXPIRED на ARM/ARM_ENABLE считается тем же переходным состоянием отмены, что NOT_READY; время (1.5 с) и частота (30/10 с) по-прежнему ограничены, прочие отказы (RANGE и т.д.) отключают сессию.
- Новый тест test_arm_expired_rejection_is_transient_like_not_ready; scripts/rosmaster_robot_tests.sh — воспроизводимый прогон тестов агента на роботе (FakeBoard, без приводов): 27 PASS.
- Установлено в ~/explorer3 (агент не запускался), резерв maintenance/backup-20261113-165119.

## 2026-11-13 17:20 — запись: причины окончания и происхождение (§8, §9)
Уже было исправлено основным диалогом и проверено: таймаут испытания → failure/task_deadline; неизвестный measured_response_kind → unknown; для кадров ROSMASTER (action_time_kind=command_in_force_at_next_agent_snapshot) send_latency = null, а не 0 (explorer3/timing.py).
Добавлено в explorer3/rosmaster_session.py: termination_reason различает operator_success/operator_failure/operator_abort, operator_stop (STOP, скрытая вкладка, потеря heartbeat), robot_stop (агент сам остановил), hardware_or_network_error, emergency (не перезаписывается последующим обрывом), task_deadline; в provenance каждого сегмента domain=real, source=human|policy, agent_code_sha256 четырёх файлов агента. Тест tests/test_rosmaster_recording.py проверяет эти поля (fake agent, 7 PASS вместе с test_rosmaster_operator).
Исследовательская запись ROSMASTER идёт только через собственный агент (/sample.jpg); explorer3/leader8090.py — отдельная совместимость с Console 8090, rosmaster_session её не импортирует.

## 2026-11-13 17:45 — параметры для симуляции (только чтение через прежний мост)
Полностью: docs/handoffs/rosmaster_measured_params.json (раздельно измерено / опубликовано / предположено).
- Камера Orbbec: 640×480 rgb8, 13.9 к/с; fx=fy=480.71, cx=322.3, cy=245.6; поле зрения 53.06° вертикаль / 67.4° горизонталь. В sim было 60° → исправлено на 53.06° в models.py и mobile.py (только ROSMASTER), тесты test_rosmaster_sim_contacts 2/2 PASS.
- Камера на руке (arm4, DCW2_ON_ARM4) — sim совпадает с CubeBot; камера URDF производителя на шасси — у этого робота не используется.
- Одометрия /odom_raw 49.7 Гц: скорость — измеренные энкодеры (только при encoder_measurement_valid); поза — интеграл измеренной скорости (дрейф), не интеграл команд.
- IMU 100 Гц, батарея 2 Гц (12.2 В), лидары 2×6.1 Гц (666 точек, 12 м), обратная связь руки ~65 Гц на сустав, все 6 конечны.
- Пределы суставов 1–5 URDF = калибровка контроллера; захват описан разными шкалами (URDF палец −88.2…0°, контроллер 30…170°).
Не измерено (нужна разрешённая запись движения): динамика приводов и захвата, динамика mecanum, задержка команда→колесо/сустав.

## 2026-11-13 18:00 — допуск этапа для ROSMASTER
Найдено: панель ROSMASTER не проверяла глобальный допуск (stage=A, hardware_connections=false) и могла забрать робот/включить управление. Исправлено: действия acquire/enable/prepare/mirror_on/leader/record_start/trial_start/preset/pair_capture проверяют local_work.hardware_allowed('rosmaster'); при этапе A — понятный отказ, STOP и чтение состояния не блокируются. Исключение только для ROSMASTER: configs/rosmaster-hardware-enabled.json {"enabled": true} (Tank/SO-101 им не открываются; tests/test_rosmaster_gating.py). Проверено вживую после перезапуска UI (без активных заданий/управления): «Включить» → отказ этапа A, агент на роботе остался inactive. Исключение НЕ создано — ждёт согласия пользователя.
