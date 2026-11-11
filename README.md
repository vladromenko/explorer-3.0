# Explorer 3.0

Новый независимый проект на Homebrain (Jetson Orin NX 16 GB). Этап A прошёл программную приёмку; ожидается физический блок B. Подробности — STATE.md.

Private repository: https://github.com/vladromenko/explorer-3.0

Локальный адрес UI: http://127.0.0.1:8030 (запуск и перезапуск проверены).
Доступ с Mac через существующий SSH: `ssh -N -L 8030:127.0.0.1:8030 jetsonnx@192.168.0.168`, затем открыть http://127.0.0.1:8030 на Mac.
Приложение слушает только loopback: канал управления не публикуется в интернет. Mac только браузер/терминал; все вычисления и файлы на Homebrain.

Запуск: `./start.sh`. Остановка: `./stop.sh`. Отдельная user-служба `explorer3-ui.service`.
Повторный запуск/остановка идемпотентны. Открытие страницы и рестарт никогда не разрешают движение.
Установка службы: `./scripts/install_service.sh`. Автоматическое включение на загрузке пока не настроено.

Код в `~/Projects/explorer-3.0`. Локальные данные, веса, калибровки, SQLite и журналы в `~/.local/share/explorer3`, вне Git.
Старые базы/датасеты/checkpoints не используются. Зависимости закреплены; PyTorch/CUDA системы не изменяются.

Последовательность: A Homebrain → B SO-101 → C Tank → D ROSMASTER → E совместная приёмка → F сбор и исследование.
Tank и ROSMASTER для A не нужны. USB-подключение не равно проверке питания или разрешению движения.
Homebrain от сети; SO-101 от существующих подходящих блоков; машины от штатных батарей. Напряжения/источники не менять.
Неподтверждённый заряд отображается как неизвестный. Никаких придуманных процентов или одометрии.

LeRobot 0.6.1, upstream tag 7e241bd630a3719a56157a497ce5d08f244784f1. NiceGUI 3.19.0, Quasar и CSS локальные, без CDN.
Источники API: https://github.com/huggingface/lerobot/tree/v0.6.1 и https://nicegui.io/documentation/section_foundations.
Происхождение копий — docs/PROVENANCE.md. Научный протокол — docs/RESEARCH_PROTOCOL.md.

Непроверенные физические функции заблокированы. Software smoke использует отдельный synthetic fixture и не означает физический PASS.

Проверено offline: UI и локальные assets; 3 synthetic video episodes → ACT 50 steps → новый процесс reload/inference; SmolVLA forward/backward/save/reload и официальный trainer 1 step. 4 contract-теста и pip check PASS. Физические проверки не выполнялись.

Ограничения: viewer настоящих эпизодов, replay/rollout и длительное обучение с validation/resume ещё требуют доводки на следующих этапах; выключенные кнопки не означают реализованную функцию.
