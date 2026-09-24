# ADR-0010 — Замена ревизии фиксируется без отдельного final

- **Статус:** принятое продуктовое решение
- **Дата:** 24.09.2026
- **Владелец решения:** владелец продукта SlavikAI
- **Затронутые области:** task/run lifecycle, communication, verification,
  artifacts, recovery, persistence

## Контекст

Lifecycle snapshot `6040640` называет `superseded` terminal состоянием
принятой task revision. ADR-0002 задаёт четыре user-facing terminal causes
(`completed`, `failed`, `cancelled`, `aborted`) и один logical final для
соответствующей terminal task revision. Если считать supersession пятой
причиной final, старая и новая ревизии могут породить competing finals.

## Решение

1. Когда новая принятая ревизия заменяет старую, старая получает **явный
   `superseded` status и lineage link**, но **не получает отдельный logical
   final только из-за замены**. Это закрытие ревизии для дальнейшего
   исполнения, а не пятая terminal cause в ADR-0002.
2. Authoritative Lifecycle фиксирует old/new revision IDs, actor/основание,
   момент принятия новой ревизии и status transition. Communication может
   показать status/correction о замене с ссылкой на новую ревизию, но не
   публикует `final_success`, `final_failure`, `final_cancelled`,
   `final_aborted` или `final_partial` для старой только по этому событию.
3. Уже доставленный до замены final и его evidence/artifact refs сохраняются
   как исторический факт; его не удаляют и не переписывают. Замена после
   final видна как новая lineage revision и, если нужно, явная correction.
4. Незавершённые effects и результаты старой ревизии сохраняют свои identity
   и provenance. Новая ревизия использует их только через явную ссылку,
   policy и criteria revalidation. Замена не является cancellation effect и
   не доказывает отсутствие внешнего side effect.

## Почему

Владелец продукта выбрал status + lineage без отдельного final старой
ревизии. Это сохраняет один актуальный путь ответа пользователю и не
маскирует уже выданный final или частичную работу. Статус исполнения
ревизии и user-facing terminal outcome имеют разные semantics.

## Рассмотренные варианты

- Публиковать пятый final class `final_superseded`: отклонено владельцем
  продукта.
- Переименовывать замену в `cancelled` или `aborted`: отклонено, потому что
  это иные причины завершения по ADR-0002.
- Молча заменить старую ревизию без lineage: отклонено из-за потери
  provenance, recovery и объяснения для пользователя.

## Последствия

- OQ-CD-03 закрыт на уровне продукта. ADR-0002 и Communication contract
  должны явно ограничить правило «один final» четырьмя user-facing terminal
  causes; `superseded` остаётся status/lineage transition вне final mapping.
- Lifecycle snapshot `6040640` нельзя интегрировать как есть: кроме этой
  границы, в его logical-task terminal enum отсутствует `aborted`. Нужна
  согласованная версия Lifecycle contract и проверка её с Verification.
- Runtime должен будет атомарно связывать ревизии и защищать stale workers,
  evidence и delivery. ADR не утверждает, что такой mechanism существует.
