# Explorer 3.0 — состояние

Обновлено: 2026-11-12, Europe/Moscow.
Текущий этап: **C — Tank подключён по Wi-Fi, Ethernet отключён; подготовка изолированного драйвера. Этап B завершён.**.
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

Точное следующее физическое действие (уточнённая карточка): включить только Tank от штатной батареи и перенести компьютерный конец его Ethernet-кабеля из Mac в Ethernet Homebrain. Если уже подключён к Homebrain — не переставлять. Не менять AP/Wi-Fi/Internet Sharing. Пока не двигать Tank. SO-101 после STOP можно выключить; ROSMASTER пока не нужен.

Найден результат прежней настройки сети в чате «Настройка SSH доступа к роботу»: Tank Ethernet 192.168.150.1, Mac .2, Homebrain .3; на роботе ранее был запущен контейнер MentorPi (текущая модель/состояние пока НЕ проверены). Локально подтверждены SSH alias hiwonder-tank (pi, собственный id_ed25519) и NetworkManager hiwonder-tank-wired: enP8p1s0, autoconnect=yes, static 192.168.150.3/24, never-default=yes. Сейчас Ethernet DOWN, маршрута до Tank по LAN нет. SSH Homebrain→Mac 192.168.0.129 не авторизован (publickey denied); запросили только физическую перестановку кабеля для использования уже готового прямого профиля. Сеть не меняли. После подключения проверить маршрут, SSH и модель/SDK read-only; не определять модель по SSID HW-9E015B81.

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


## C — подключение и новый запрос Wi-Fi

Пользователь перенёс Ethernet Tank → Homebrain. enP8p1s0 автоматически получил 192.168.150.3/24, маршрут к .1 прямой. SSH pi@192.168.150.1 с существующим ключом выполнен успешно. Tank: Raspberry Pi 5 Model B Rev 1.1, Debian 12 host; Docker MentorPi (ros:humble, network=host), ROS 2 Humble, драйверы в /home/ubuntu/ros2_ws/src/driver внутри контейнера. Сохранённая конфигурация /home/pi/docker/tmp/.typerc: MACHINE_TYPE=MentorPi_Tank, VERSION=|V1.0.2|2026-01-14|, камера aurora. Это прочитанная конфигурация, ещё не полная аппаратная приёмка. Старые службы/драйверы на Tank не менялись, команды движения не отправлялись.

Пользователь дополнительно попросил настроить Wi-Fi-доступ к Tank с Homebrain и Mac, чтобы отключить Ethernet. Ответ: показы сейчас не нужны; позже три коротких подъезда и возврат штатного управления. Ethernet не отключать до проверки нового Wi-Fi. Требуется сохранить штатную возможность управления, подготовить возврат конфигурации, не публиковать сетевые секреты.

После двух успешных read-only SSH инспекций Tank перестал отвечать по 192.168.150.1, ещё ДО любых изменений. Ethernet link остаётся UP, ARP FAILED, TCP 22/9090/8080 недоступны. Второй read-only вызов с чтением wifi.py не выполнился; фактическая логика wifi.service пока неизвестна. Запрошена конкретная проверка: кабель по-прежнему Tank↔Homebrain, питание включено; если кабель уже снят — вернуть до проверки Wi-Fi. Не утверждать, что Wi-Fi готов. На Homebrain активная Wi-Fi сеть Digma_049F00; пароль не выводился и не передавался.


## C — Wi-Fi готов, кабель снят (2026-11-12)

- Пользователь подтвердил Ethernet; восстановлен SSH по link-local IPv6 fe80::749e:2dc7:3e4e:310. Диагноз: IPv4 на eth0 исчез, активен DHCP без сервера. Заводской wifi.py при переключении удаляет /etc/NetworkManager/system-connections/*; не исправляли оригинал.
- С сохранёнными резервными копиями применён штатный /etc/wifi/wifi_conf.py (root:root, 0600): WIFI_MODE=2, домашний SSID и пароль из действующего профиля Homebrain. Секреты не выводились и не коммитились. Первая попытка штатного подключения не увидела SSID и вернулась в AP, удалив override. После полного scan сеть обнаружена; повторное подключение прошло. Не заявлять, что первая попытка была успешной.
- Проводной резерв восстановлен отдельным профилем /usr/lib/NetworkManager/system-connections/explorer3-tank-wired.nmconnection: 192.168.150.1/24, never-default, autoconnect. Он пережил повторное переключение wifi.service; заводской код/служба не правились. Поддерживаемый каталог подтверждён документацией NetworkManager. На Homebrain прежний hiwonder-tank-wired .3 сохранён; временный .149.3 убран при переподключении, основной Wi-Fi/VPN не менялись.
- Wi-Fi Tank: Digma_049F00, текущий DHCP IP 192.168.0.136; mDNS raspberrypi.local. SSH с Homebrain PASS, маршрут туда и обратно по Wi-Fi. Пользователь снял Ethernet; Tank eth0 DOWN/carrier=0, новый SSH по Wi-Fi снова PASS.
- Homebrain SSH aliases hiwonder-tank/tank → raspberrypi.local; отдельный hiwonder-tank-wired → 192.168.150.1. Проверяется тот же SSH host key. Backup ~/.ssh/config хранится вне Git в runtime/provenance/ssh-config-before-tank-wifi.
- На Mac использовать ssh pi@raspberrypi.local в той же домашней сети. Его public key на Tank подтверждён; фактический вход с Mac не выполняли (Homebrain→Mac SSH недоступен). Старый alias hiwonder-tank на Mac всё ещё проводной; файл на Mac не менялся.
- Бэкапы на Tank: /home/pi/explorer3/network/backup-20261111T211708Z и backup-20261111T212410Z (root-only). Четыре хеша исходных wifi.py, wifi_conf.py, wifi.service и start_node.sh совпали. Возврат AP подготовлен /home/pi/explorer3/network/restore-factory-ap.sh: только при подключённом Ethernet, сохраняет override и запускает штатный Wi-Fi менеджер. Сам возврат AP этим скриптом пока не выполнялся; штатный fallback AP наблюдался после первой неудачной STA-попытки.
- wifi.service остаётся enabled/active. Холодная перезагрузка Tank с новым Wi-Fi ещё не проверена. Не отключать Wi-Fi ради проверки без необходимости. Доказываем рабочую связь после снятия кабеля, не весь boot-path.
- Доказательства: runtime validation/tank-wifi.json, docs/tank-network-validation.json. Питание/камера/управление/записи Tank ещё НЕ приняты. Следующий шаг — read-only изучить и скопировать реально используемый драйвер/SDK; движения требуют отдельного конкретного физического блока C.
