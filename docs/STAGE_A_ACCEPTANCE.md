# Этап A — сводка программной приёмки

13 ноября 2026. Область: Homebrain; без подключений к роботам. Новые обучения
отложены прямым указанием пользователя. Это сводка программной подготовки,
не заявление об обученных навыках, калиброванном G или завершённом исследовании.

| Сценарий | Свидетельство | Результат / предел |
|---|---|---|
| SO-101 и ROSMASTER arm: физика захвата | `validation/corrective-a/contacts-final`, `STATE.md` | По 3/3 scripted переносов, проникновение проверено на каждом подшаге; реальные приводы не калиброваны |
| Tank на Homebrain | `browser-tank/result.json`; закреплённые источники в model manifest | Chromium + настоящий MuJoCo: open/move/pause/reset/record/save PASS, внешних запросов 0; оригинальная T1 геометрия, planar dynamics ещё не идентифицирована |
| ROSMASTER программные сценарии | `docs/handoffs/ROSMASTER_RESULT_RU.md` и приложенные acceptance JSON | Соседний диалог: UI, fake-agent, запись, испытание и пилот; не заменяет физическую приёмку |
| Запись → LeRobot readback | SO fcaf4239, ROS b55eaec1; `browser-branches/result.json` | Чтение PASS; scripted sim, не человеческие показы |
| Общий мир и политики | `fleet-real-policies-final/result.json`, `browser-fleet-policies/result.json` | Два настоящих старых ACT одновременно; очереди помощи, отдельные камеры/частоты, STOP и повторное открытие отчёта PASS |
| AutoIntervene-adapted | `tests/test_autointervene.py`, `tests/test_fleet_execution.py` | Алгоритм и online switching проверены численно; новая held-out калибровка не выполнена без новой обученной политики |
| R5: три сопоставимые ветви | `browser-branches/result.json`, `tests/test_sim_branching.py` | Snapshot/replay, одинаковые бюджеты, censored не failure, просмотр и импорт сегмента PASS |
| Общий оценщик G | `tests/test_gain_preparation.py`, `browser-gain/result.json` | Признаки до решения, план, разбиение, JSON inference, shadow подключение; подходящих новых данных и обученного G пока нет |
| Ручное обучение/fine-tune | `browser-training-draft/result.json`, `docs/TRAINING_PREPARATION.md` | План создан/скачан/открыт, jobs не созданы; обновления весов не запускались |
| Дополнительные модели/зрение | `browser-vision/result.json`, `diffusion-inference/result.json`, STATE SmolVLA | Реальные YOLOE/SAM3 и сохранение/восстановление inference проверены; не навык управления |
| Сбой/отмена/восстановление | native tests, browser-fleet-policies | STOP отменяет конечный процесс; после отмены policy workers не остались; snapshot сохраняет RNG/chunks/TD историю; изменённые веса/калибровка/источники отклоняются |
| Offline и cold restart | browser reports external=[], `/api/health` после restart | Главный UI запущен с `70a8792`, PID1271092; обе новые страницы HTTP200, stage A/local_only, armed=false, compute_job=null |

Относительные пути `browser-*`, `fleet-*` выше находятся в runtime
`/home/jetsonnx/.local/share/explorer3/validation/corrective-a/`.
Проверки разных подсистем выполнены в соответствующих зафиксированных версиях;
это не повтор всех старых тестов на одном новом HEAD. Новые затронутые сценарии
проверены повторно, результаты не приписываются аппаратуре.

## Граница перехода

Для физической части порядок остаётся B ROSMASTER → C Tank → D SO-101.
Tank **не нужно включать, чтобы закончить программную часть A или начать B**.
Его USB/STOP-инцидент и независимая остановка требуют отдельного блока C;
прошлые успешные проверки не закрывают этот инцидент.

Аппаратный допуск автоматически не открыт. Обучение моторных политик,
обучение/калибровка G, улучшение после повторных раундов и проверка переноса
остаются явно отложенными. Программная подготовка этих маршрутов не означает,
что исходное требование A3/A4 о результативном обучении выполнено после запрета
запуска. Продолжение B не должно выдавать эти отложенные результаты за готовые.
