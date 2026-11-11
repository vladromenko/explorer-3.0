# Explorer 3.0 — состояние

Обновлено: 2026-11-12, Europe/Moscow.
Текущий этап: **C — подготовка подключения только Tank. Этап B завершён.**.
Не начинать заново. Пользователь разрешил передачу управления, телеперацию, три показа и возврат. Старую систему http://192.168.0.168:8020 изменять или исправлять запрещено.

## Проект и доступ

- Homebrain, Jetson Orin NX 16 GB, aarch64, Python 3.12.3. Новый код: /home/jetsonnx/Projects/explorer-3.0.
- Private GitHub: https://github.com/vladromenko/explorer-3.0. Новый репозиторий без старой истории; CLI авторизован, push работает.
- UI: http://192.168.0.168:8030, локально http://127.0.0.1:8030. Один NiceGUI/FastAPI сервер и SQLite. Разрешены loopback и LAN 192.168.0.0/24; проверяется Origin.
- ./start.sh и ./stop.sh, user-служба explorer3-ui.service. Старт/рестарт не открывает приводы и не запускает движение.
- Данные, веса, SQLite, калибровки и журналы: /home/jetsonnx/.local/share/explorer3, вне Git.
- Старая so101-ai-learning-ui.service восстановлена: active, handoff.json was_active=true/restored=true. HTTP :8020 отвечает, teleoperation/recording idle, hardware_open=false, устройств не занимает. Пользователь проверил прежнюю телеперацию и нажал STOP; 706 samples, hardware_open=false, error=null, stop_errors=[]; другие старые службы не изменялись.

## A — завершено

- Новая .venv: LeRobot 0.6.1, upstream tag 7e241bd630a3719a56157a497ce5d08f244784f1; NiceGUI 3.19.0. Lock: configs/requirements.lock.txt; pip check PASS.
- torch 2.11.0+cu130 / torchvision 0.26.0+cu130 установлены только в новую .venv. Системная CUDA/JetPack и старые окружения не изменены. Предупреждение PyTorch о CC8.7 сохраняется; реальные CUDA train/reload выполнены.
- ACT synthetic: 3 эпизода, 36 кадров 64×64, официальный trainer 50 шагов batch 8, новый процесс reload/inference PASS.
- SmolVLA public base/tokenizer в собственном cache: forward/backward/save/reload PASS; официальный trainer 1 шаг batch 1 и новый процесс reload PASS.
- Offline: отдельный cgroup с запретом внешней сети и разрешённым LAN; UI, dataset, train/reload/inference PASS. Глобальная сеть не менялась. Web assets локальные, CDN нет.
- Доказательства: runtime validation/{act-pass,smolvla-pass,smolvla-trainer-pass,offline-pass,ui-assets}.json; docs/stage-a-validation.json.
- Старые датасеты/checkpoints не использованы. Низкоуровневый hardware.py скопирован в новый проект с собственными путями. Собственные калибровки; происхождение и хеши в docs/PROVENANCE.md и runtime/provenance.

## B — завершено, проверенные факты

- Read-only probe: калибровки обеих рук совпали, шесть приводов каждой отвечают в Position mode, wrist 640×480 YUYV около 30.17 Hz. Camera controls, нули, лимиты и питание не менялись.
- Пользователь подтвердил плавное Leader→Follower и STOP. Ограниченный teleop probe: 28 samples, закрытие без ошибок.
- Первый preview был редким; затем общий WebSocket перегружался изображениями. Две прерванные попытки сохранены и excluded=1: 3455b038595e439baa67aa91f5443c0d (202), dd9b4139e42f4895b67f683b0ce84a03 (87). Точная причина старых остановок не доказана; serial/camera ошибок в логе не было.
- Исправление только Explorer: отдельный HTTP MJPEG, JPEG вне event loop, последний кадр и backpressure; график 4 Hz. Убран stop на window.blur, сохранены hidden/pagehide/disconnect и heartbeat timeout 1 s. Причины остановки сохраняются.
- Реальная камера → HTTP: 60 кадров, 14.32 fps, max gap 72.28 ms, event-loop max gap 83.07 ms, без моторов. Пользователь затем подтвердил плавное видео и следование без отключений.
- Пользователь проверил уход на другую вкладку: Follower остановился и после возвращения сам не возобновился. Сервер idle, thread_active=false, error=null, причина «Браузер: вкладка скрыта», 255 кадров без записи. runtime validation/so101-hidden-tab-stop.json. Физический разрыв сети отдельно не выполнялся.
- Три настоящих success: 451048674bfe4b579bcf3c6ea800c3aa (1117 кадров), cc209320e3af486b83e74791977dd913 (922), 47c74f2756df42edb42514658a43e04a (597). Всего 2636 кадров, около 30 Hz. LeRobot readback, causal sidecar, finite action/state, первое/среднее/последнее видео PASS. Повторы камеры честно отмечены; p95 age <32 ms. runtime validation/episode-{id}.json.
- Viewer настоящих эпизодов загружен: HTML 200, mp4 range 206; синхронный cursor/video/action/state. Пользователь подтвердил live preview; автоматизированная визуальная проверка браузером недоступна.
- Десять компактных тестов PASS. 24 source/service хеша прежней системы повторно совпали после возврата. Найден и исправлен зависающий при остановке HTTP MJPEG: поток отслеживает shutdown сервера. Реальный restart с открытым MJPEG 0.285 s, Result=success, приводы не открывались.

## Сейчас и следующий шаг

ACT smoke job be7262c82f354190b82eead78129f6cf завершён: 50 шагов, effective batch 8, seed 1000, save PASS, новый процесс reload/inference PASS [1,6], без команд моторам. Первый вызов inference 0.411 s — не замер устойчивого цикла. docs/stage-b-training.json; runtime validation/so101-act-pass.json.
Три выбранных эпизода объединены официальным LeRobot merge_datasets без перекодирования MP4: хеши совпали, 2636 кадров/3 эпизода; aborted не использованы. runtime validation/so101-training-input.json.
B завершён: docs/stage-b-validation.json и runtime validation/stage-b-pass.json. Пользователь подтвердил прежнее управление и STOP; старые код/служба неизменны. Карточка дальнейших SO-101 показов в README: ориентир 50 всего, три пилотных сохранены; массовый сбор сейчас не требуется.

Точное следующее физическое действие: включить только Tank от штатной батареи, подключить доступный Ethernet к Mac по уже подготовленной схеме (подключённый кабель не переставлять). Не менять AP/Wi-Fi/Internet Sharing. Пока не двигать Tank. SO-101 после STOP можно выключить; ROSMASTER пока не нужен. Затем read-only определить реальный маршрут Homebrain→Tank и модель/SDK; не предполагать модель по SSID HW-9E015B81.

| Профиль | Запись | Обучение на настоящих показах | Физический навык |
|---|---|---|---|
| so101_pick_place | проверена, 3 success | ACT smoke 50 шагов + reload проверены | не проверен |
| tank_dock | не проверена | не проверено | не проверен |
| rosmaster_dock | не проверена | не проверено | не проверен |
| rosmaster_pick_place | не проверена | не проверено | не проверен |

## Ограничения и дальнейшие этапы

- Smoke checkpoint не является готовым навыком. Первый ориентир 50 качественных показов на профиль, включая подходящие пилотные; не гарантия успеха. Не ждать 50 для перехода к C.
- Replay/rollout и длительный training с validation/resume ещё требуют реализации/приёмки. Полное обучение из UI пока не объявлено готовым. До него проверить группировку сессий/расстановок и заморозить split.
- ROSMASTER не исследован и не запрашивался. Начата C — только Tank, сохранить готовящуюся Ethernet-схему через Mac. D — только ROSMASTER. E — все вместе только после раздельной приёмки.
- Старый неудачный dataset не удалён: неоднозначные кандидаты ~/.cache/huggingface/lerobot/local/cubes_to_box_v1_20261031_001153, ~/Projects/so101_ai_learning/data/ds_e214e7ffaf654b0b, ~/Projects/so101_ai_learning/data/ds_045fe3a18fd84445. До удаления нужен точный выбор; никаких масок. Новая просьба не менять старую систему имеет приоритет.
- Исследовательский протокол docs/RESEARCH_PROTOCOL.md; оценщик и эксперимент — F. Новизна относительно RESCUE не проверена, утверждать её нельзя.
