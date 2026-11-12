# Происхождение

Скопирован только низкоуровневый SO-101 I/O из `/home/jetsonnx/Projects/so101_ai_learning/src/so101_lab/hardware.py`.
SHA256 исходника: `c1bb47ffe68f994717a28d25a8bde2670dbc999c20429311d31ad7f630486075`.
Рабочая копия: explorer3/hardware.py. Исходник сохранён локально вне Git.
Авторство: существующий пользовательский проект so101_ai_learning; отдельная лицензия не обнаружена; публичное распространение не разрешено этим документом.
Изменения: собственный каталог данных/калибровок, wrist-only, собственный resource lease. Backend/UI/trainer/данные не копировались.
Оригинал не изменён. Это копия используемого службой I/O; физическая работоспособность копии проверена на этапе B (три настоящих показа; прежнее управление восстановлено).
LeRobot устанавливается из PyPI как upstream dependency, Apache-2.0; не копируется из старой venv.
Калибровки: собственные файлы вне Git, robot ID и моторы сохранены, хеши и источники в локальном calibration/manifest.json.
PyTorch/torchvision: новые установки тех же версий 2.11.0+cu130/0.26.0+cu130, без изменения системной CUDA.

Публичный SmolVLA base: revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6.
SmolVLM2-500M-Video-Instruct: revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467.
Взяты из стандартного локального Hub-кэша (не из старых проектов), развёрнуты обычными файлами в собственный cache.
Хеши всех snapshot-файлов и ResNet18 сохранены в локальном cache/manifest.json.


Tank: установленные Hiwonder ROS2 driver/controller и ros_robot_controller изучены read-only; отдельные Board SDK и Aurora930 executable/library скопированы внутрь /home/ubuntu/explorer3/vendor. Авторы — Hiwonder / Deptrum; лицензия на распространение SDK не установлена (package.xml содержит TODO), бинарные пакеты в Git не включены. Собственные glue/safety/HTTP файлы находятся в agents/tank. Оригиналы не исправлялись. Manifest с исходными путями и SHA256 — локальный runtime/provenance/tank-vendor-manifest.json. Геометрия/знаки/лимиты воспроизведены из реально установленного controller, коррекции прочитаны из robot_correction_factors.yaml без изменения. Новый namespace /explorer3/tank; ROS-домен и дистрибутив сохранены. Новая venv использует неизменяемые системные ROS/OpenCV, не старое окружение приложения. Источники документации и ограничения — docs/TANK.md.

После отказа первой физической проверки дополнительно скопирована полная низкоуровневая цепочка controller/ros_robot_controller с необходимыми конфигами, всего 12 файлов. Manifest source/copy SHA256: runtime/provenance/tank-native-chain/native-manifest.json. Скопированные оригиналы вне Git; авторские заголовки сохранены. Изменены только пути конфигов на собственные. Own native_driver.py исполняет оригинальные классы/конструкторы/ROS callbacks; прежняя собственная функция кинематики удалена. Зависимости сообщений ROS используются как неизменяемые установленые библиотеки; старый веб-backend не используется. Два теста с памятью вместо serial не считаются физической приёмкой.


### Локальный адаптер выбранной Console8090 (2026-11-12)

agents/rosmaster/control_adapter.py содержит два метода `_manual_step` и `_mirror_step`, скопированные без изменений из снимка mirror-fix-20261112/candidate/cubebot/web/server.py, соответствующего выполненному откату. Исходный SHA256 и отдельные хеши методов — control-adapter-provenance.json. Native CartesianJog, MirrorMapper, ArmModel и конфигурация переиспользуются из собственного native/cubebot; старый веб-backend не импортируется. Обёртка только разделяет профили базы/руки, проверяет вход и истечение команд. Нет изменения знаков/нули/лимитов, сглаженного приведения или отклонённого partial-feedback патча. Копия start-agent.sh исправлена локально на известный ROS_DOMAIN_ID=30 и без MoveIt; на робота не установлена. Живой ROS runtime и handoff остаются отдельной незавершённой аппаратной интеграцией.
