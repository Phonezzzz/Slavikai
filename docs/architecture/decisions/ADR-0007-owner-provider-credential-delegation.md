# ADR-0007 — Ключи провайдеров владельца закрыты по умолчанию

- **Статус:** принятое продуктовое решение
- **Дата:** 24.09.2026
- **Владелец решения:** владелец продукта SlavikAI
- **Затронутые области:** identity/principal, credentials, policy, model access,
  provider egress, STT/TTS, automation, audit

## Контекст

Принятый deployment рассчитан на одного owner и допущенных members за
Cloudflare Access; bearer automation является отдельным auth lane. Текущие
provider keys могут поступать из environment или общего для приложения
`config/api_keys.json`. Изменение ключей через UI Settings требует роли owner,
но `server/http/common/ui_settings.py::_resolve_provider_api_key` принимает
provider и источник ключа, а не principal или решение о делегировании.
Проверенный путь UI chat вызывает этот resolver при настройке scoped Agent;
STT тоже вызывает его напрямую. Этот source trace указывает на Gap с Target,
но сам по себе не доказывает поведение всех routes или live exploit. Значения
секретов не читались.

## Решение

1. Сохранённый или настроенный для owner provider key по умолчанию может
   использовать **только owner**. Member или automation principal не получает
   права на него из факта доступа к SlavikAI, общей session, выбора model route
   или назначения task.
2. Owner может выдать право использования явно названному principal по
   отдельному правилу. Правило связывает как минимум credential и route или
   provider, назначение, principal, срок действия/отзыв и применимые лимиты.
   Policy/Credential authority проверяет правило на каждом использовании или
   при выдаче ограниченного credential handle. Model Access не считает key
   availability доказательством route eligibility.
3. Делегирование даёт **использование**, но не чтение/экспорт raw secret, право
   изменять ключ, общее разрешение на инструменты или принятие результата.
   Provider request и audit record сохраняют effective principal, ссылку на
   owner/delegation и выбранный route, не записывая сам ключ.
4. То же правило действует для chat models, embeddings, STT/TTS и других
   provider-backed capabilities. Ключ из process environment должен иметь
   явную классификацию owner/credential и не обходить principal check.
   Точный storage/handle protocol и migration определяются отдельно.

## Почему

Право owner менять Settings не означает согласие оплачивать или раскрывать
данные каждого member/automation run. Credential имеет собственные scope и
lifetime, отличные от session и model selection. Явное делегирование
разрешает нужное совместное использование с отзывом и учётом egress.

## Рассмотренные варианты

- Все допущенные members и automation используют общий ключ: отклонено
  владельцем продукта; допуск к SlavikAI не даёт общего права на private
  provider account.
- Каждый principal обязан принести свой ключ: слишком жёстко, поскольку
  владелец разрешил узкое явное делегирование.
- Достаточно запретить members менять Settings: недостаточно, потому что
  read/use path потребляет уже сохранённый ключ без изменения настроек.

## Последствия

- OQ-CD-05 закрыт для Target. Разделы 8/9/18/19 должны проверить все пути
  получения ключа и egress, определить action-time principal/delegation
  checks, отзыв, ротацию и безопасное внедрение без утечки секретов.
- Проверенный application-level resolver и отдельные UI chat/STT пути не
  передают principal/delegation в функцию выбора ключа. Это Current State Gap
  для полного route audit и отдельного implementation PR. Сам ADR не меняет
  runtime и статус claims registry.
