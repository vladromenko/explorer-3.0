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
