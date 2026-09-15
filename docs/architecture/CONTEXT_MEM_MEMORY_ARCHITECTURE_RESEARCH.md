# context-mem как архитектурный reference для памяти SlavikAI

Дата исследования: 2026-09-06.

**Статус: исследование и кандидатная концептуальная модель, не принятый архитектурный контракт.**
Документ не меняет ARCH_CANON, SOURCE_OF_TRUTH или статусы runtime claims. Он не содержит плана реализации, PR-последовательности или предложений по миграции. Существующие классы, таблицы и ограничения реализации SlavikAI не используются как границы целевого дизайна.

Исследован context-mem на коммите **2a55af0a4bf3467df89f1315a74bb2e15ad903f7**; [package.json][package] указывает версию 4.0.0; продуктовые заявления сверены с [README][readme]. Все внешние ссылки ниже закреплены на этом коммите. Архитектурный контекст SlavikAI прочитан на **472f3a0eb8f1903429655c46af3791b54beac504**. Исходники reference изучались в отдельном временном каталоге, без установки пакета, IDE hooks, моделей и сервисов.

**Основной вывод:** наиболее полезна декомпозиция «сохранённые свидетельства → поддерживаемое знание → производные представления → выбранный контекст». Однако context-mem не реализует эту декомпозицию последовательно: его temporal, provenance, privacy, compression и retrieval механизмы местами расходятся между собой. Для personal agent стоит обсуждать принципы отдельно от готовности этой реализации.

### Метод и границы доказательств

- Прочитаны README, wiki schema, существенные разделы Context Protocol RFC, документация методологии benchmark; прослежены ingest, storage schema, search, knowledge handlers, sessions, compression, maintenance, privacy, wiki/synthesis, import и global store.
- Проверены вызывающие пути Kernel, MCP handlers и Claude Code hooks. Наличие функции отдельно от её подключения к runtime не считается подтверждением работы всей возможности.
- Прочитаны релевантные тесты и benchmark harness. Полный upstream test suite, benchmark datasets, embedding runtime и LLM-сервисы **не запускались**. Числа качества из README не воспроизводились.
- На SQLite в оперативной памяти применены все 18 SQL-блоков схемы reference и выполнены проверки отдельных запросов. Они подтвердили удаление pinned context-наблюдения через TTL, исключение архивного исторического факта из поисковых кандидатов и уникальность content_hash вне границ сессии. Это проверки SQL-семантики, не сквозной запуск TypeScript-сервера.
- Тело canonicalize из SearchFusion выполнено в Node: запросы «память проекта» и «любимая музыка» дают одинаковый пустой ключ. Это проверка функции ключа, не замер всего поиска.
- Отрицательные выводы означают отсутствие полноценного механизма в изученных schema и execution paths данного коммита; они не претендуют на аудит всех интеграций и будущих версий.

### Правила и контекст исследования

Применены обязательные документы AGENTS: SOURCE_OF_TRUTH, ARCH_CANON, DevRules, dev_workflow, COMMAND_LANE_POLICY, ROUTING_POLICY, STOP_RESPONSES; дополнительно прочитаны Architecture.md, архитектурный README, runtime_contract_claims.json и CONTRIBUTING.md. В исходном docs/architecture/ находились ARCH_CANON.md, Architecture.md и README.md.

Для проектирования существенны: personal agent как продукт; единый Chat как conversational entrypoint; Agent/Desktop как контуры исполнения; principal isolation; provider-neutral tool loop; отдельные approvals; различение current/target/legacy. Runtime sandbox и approvals нельзя обходить посредством памяти. Исследование не запускает runtime и не принимает решения за пользователя.

Текущий контракт подтверждённой записи памяти обозначен ниже явно. Обсуждение другого будущего capture policy не является его скрытым изменением. Потребность в автоматическом сборе рассматривается по существу, а не отвергается из-за текущего кода. [Канон SlavikAI](ARCH_CANON.md), [текущая архитектура](Architecture.md), [иерархия контрактов](../SOURCE_OF_TRUTH.md), [claims](../runtime_contract_claims.json).

## 1. Architecture of context-mem

### 1.1. Границы системы и реальные потоки

Это локальная TypeScript/Node система с SQLite, MCP interface, HTTP bridge, интеграционными hooks и необязательными embeddings/LLM. Memory-функции соседствуют с execution tools и средствами координации агентов; для SlavikAI такая комплектация не является архитектурным требованием. SQLite открывается с WAL и foreign_keys; это транзакционное хранилище, а не механизм проверки истинности содержимого. [Kernel][kernel], [storage][storage], [types/defaults][types].

~~~text
MCP observe / HTTP observe / conversation import
                  |
              Pipeline
                  |
      budget -> privacy/redaction -> hash dedup
                  |
      entity + importance + topic heuristics
                  |
      pinned / optional LLM / content summarizer
                  |
          SQLite observation
          /       |          \
   topic links  entities    embedding
          \       |          /
             projections
        wiki / synthesis / snapshots

Kernel.observe -> additional automatic knowledge extraction
Dreamer        -> stale/archive/compress/consolidate maintenance

MCP unified search
  + observations -> BM25/trigram/edit-distance/[vector] -> fusion/[LLM judge]
  + knowledge    -> FTS/trigram/scan -> relevance adjustments
  + content      -> chunks FTS
  + topics       -> topic-related observations
                  |
        merged ranked response
                  |
   session primer / per-prompt hook / explicit retrieval
~~~

Квадратные скобки означают optional механизм. Это несколько связанных путей: не каждый вход проходит через все перечисленные стадии. Например, MCP observe вызывает Pipeline непосредственно, а дополнительное auto-extraction находится в Kernel.observe. Unified search объединяет разные поисковые ветви; он не делает каждый тип памяти автоматически векторным. [Pipeline][pipeline], [Kernel][kernel], [MCP core][mcp-core], [unified search][search-tools].

### 1.2. Storage и knowledge model

| Объект | Что реально хранится | Архитектурное значение и ограничение |
|---|---|---|
| observations | content, summary, metadata, embeddings, indexed_at, privacy_level, session_id, hash, correlation_id, importance, pinned, compression_tier, access/usefulness fields | В одной строке смешаны исходное сохранённое содержимое, производные представления и изменяемая статистика |
| knowledge | category, title, content, tags, shareable, source_type, relevance/access, stale/archived, valid_from/to, superseded_by | Документное знание с временными полями; нет обязательного typed claim с предметом, предикатом, объектом и набором evidence references |
| entities / relationships | Имя, тип, metadata, aliases/canonical_id; направленные типизированные связи | Граф сущностей существует; происхождение и временная истинность каждой связи не являются обязательной частью схемы |
| topics / observation_topics | Темы, счётчики, связи observation–topic, confidence | Тематическая навигация; не доказательство независимого подтверждения знания |
| content_sources / content_chunks | Идентификатор источника, hash строки source, фрагменты и FTS | Отдельный документный индекс; версия содержимого источника не входит в ключ дедупликации |
| events / snapshots / session_chains | События, один заменяемый snapshot на session, parent/chain/project, summary | Восстановление работы; не полный неизменяемый журнал всех ревизий задачи |
| decision_trails / working_fingerprints | Сохранённая реконструкция trail и fingerprints | Специализированные производные артефакты |
| FTS, embeddings, token_stats, error_log | Индексы и операционная статистика | Технические representations/metrics, а не отдельная эпистемическая память |

Основная схема содержит 18 последовательных версий. ObservationType ограничен code/error/log/test/commit/decision/context. KnowledgeCategory в типах включает также answer/summary, однако публичный save_knowledge принимает pattern/decision/error/api/component. source_type различает explicit/inferred/observed, но не заменяет подтверждение авторства и согласия. [Схема][schema], [типы][types], [knowledge tools][knowledge-tools], [content store][content-store].

### 1.3. Capture и сохранённые свидетельства

Pipeline сначала обрабатывает privacy, затем считает SHA-256 **очищенного** текста. При совпадении content_hash возвращается существующее наблюдение, прежде чем создаются новые session/time/source metadata. Это экономит место, но объединяет разные события с одинаковым содержимым. Повторная проверка «сервис работает» через месяц должна быть новым свидетельством времени проверки, даже если текст идентичен. Можно дедуплицировать payload, сохраняя каждое occurrence отдельно. [Pipeline][pipeline], [уникальный индекс][schema].

Незакреплённые observations могут получить summary уже при ingest. При этом content сохраняется отдельно, а compression_tier первоначально равен verbatim. Поэтому verbatim tier не означает «summary отсутствует» и не гарантирует, что именно content попадёт в выдачу. Перед Pipeline integration hook ограничивает текст 50 000 символами; raw в БД означает то, что дошло до границы сохранения, а не гарантированно полный исходный tool response. [Pipeline][pipeline], [capture hook][capture-hook].

Автоматический capture в Claude Code привязан к конкретному matcher Bash/Read/Write/Edit/Grep/Glob. Для Write/Edit он может брать предложенный input content/new_string. Это не универсальное свидетельство успешного внешнего действия. Наличие post-tool hook само по себе не устанавливает, что получен проверенный конечный эффект. [Hooks][hooks], [capture hook][capture-hook].

### 1.4. Retrieval

**Lexical:** SQLite FTS5 с porter unicode61 индексирует summary/content; отдельный content-only FTS поддерживает verbatim recall. BM25Search выполняет варианты AND, phrases, entities, sanitized query, relaxed AND, OR+synonyms, individual keywords/synonyms и temporal расширения. Результаты разных запросов объединяются по ID с максимальным score, затем нормализуются. Это несколько запросов к одному FTS, а не восемь независимых источников доказательств. [BM25][bm25], [схема][schema].

**Vector:** optional @huggingface/transformers, nomic-embed-text-v1.5, 768 float32, cosine similarity. Embedding строится преимущественно по summary; для короткого текста await, для длинного — отложенное вычисление. VectorSearch сканирует максимум **5000 последних** строк с embeddings, затем сортирует по cosine; ANN-индекса здесь нет. В DEFAULT_CONFIG vector отсутствует в plugins.search: нужен явный выбор и доступная зависимость. [Embedder][embedder], [vector search][vector], [Pipeline][pipeline], [Kernel][kernel].

**Fusion/reranking:** разные plugins вызываются через Promise.all. Есть weights по intent, бонус пересечения стратегий, recency/access и IDF/bigram сопоставление сниппетов. Это эвристические score, не вероятности правильности. Kernel настраивает LLMJudge с **0.6 retrieval / 0.4 LLM**, 800 символами на candidate и maxCandidates=20; defaults самого класса — 0.5/0.5, 1500 и 30. README описывает вариант 50/50, который нельзя выдавать за единственную конфигурацию runtime. [Fusion][fusion], [judge][judge], [Kernel][kernel].

**Unified surface:** observations, knowledge, content и topics извлекаются отдельно и объединяются сортировкой relevance_score. semantic/hybrid/temporal observation modes идут через fusion; temporal добавляет referenceDate. Часть filters применяется уже после ограниченной выдачи. Knowledge wrappers исключают valid_to != NULL после первичного retrieval; это не полноценная проверка текущего interval, поскольку future valid_from отдельно не учитывается. [Unified search][search-tools].

**Cache:** SearchFusion имеет 30-секундный cache на 100 записей. Ключ — нормализованный текст запроса, без opts, limit, referenceDate или версии хранилища. Удаление пунктуации через ASCII-ориентированное регулярное выражение также удаляет кириллицу. Это конкретный дефект correctness; идея кеша сама по себе полезна. [Fusion][fusion].

### 1.5. Temporal, supersession и provenance

save_knowledge устанавливает valid_from отдельно после save. При force и найденных «противоречиях» старым записям выставляются valid_to=Date.now и superseded_by. Но resolve_contradiction(action=supersede) вместо этого архивирует старую запись, добавляет tag и optional graph relationship; merge создаёт новую и архивирует обе исходные. Общей атомарной операции knowledge revision с единой временной семантикой нет. [Knowledge tools][knowledge-tools].

temporal_query сначала вызывает обычный KnowledgeBase.search, исключающий archived, затем фильтрует ограниченную выборку по valid_from <= at < valid_to. valid_from=NULL интерпретируется как 0. Отсюда:
- архивные исторические факты могут быть недоступны;
- неизвестное начало действия превращается в «верно с начала эпохи»;
- отбор top-k до temporal filtering может пропустить подходящий факт;
- нет отдельного transaction/recorded time для вопроса «что мы считали верным тогда».

time_travel считает записи по indexed_at/created_at, а время архивации приблизительно выводит из last_accessed. Это обзор сохранившихся данных по времени, не точное восстановление прежнего знания. [KnowledgeBase][knowledge-base], [temporal handler][knowledge-tools], [time travel][time-travel].

correlation_id, session_id, source, source_observation и source_id в trail дают начальные provenance-связи. Однако buildTrail добавляет события за предыдущий час той же сессии, затем code/context/error observations за сутки **без фильтра по сессии**. Confidence растёт от числа evidence items; близость не доказывает причинность. Dreamer тоже называет временную последовательность decision → problem → milestone causal chain. Это полезная навигационная гипотеза, которую нельзя предъявлять как зафиксированную причину решения. [Decision trail][trail], [Dreamer][dreamer].

### 1.6. Compression, session context и lifecycle

При ingest Kernel регистрирует **15** deterministic summarizers, включая Python traceback; в README часто указано 14. Они выбираются по содержимому в порядке специфичности. JSON summarizer, например, извлекает структуру и типы; error summarizer оставляет сообщение и несколько frames. Это целевое сокращение информации, а не универсальная семантически без потерь компрессия. [Kernel][kernel], [JSON summarizer][json-summary], [error summarizer][error-summary].

Adaptive tiers: verbatim до 7 дней, light до 30, medium до 90, затем distilled; importance >= 0.8 замедляет переход на один tier, pinned отключает сжатие. Light выбирает первые/keyword sentences, medium использует существующий summary или короткий prefix, distilled — несколько keyword facts либо первое предложение. Dreamer изменяет **summary**, сохраняя content, но не пересчитывает embeddings в той же операции. Один цикл смотрит 500 старейших незакреплённых observations; продвижение frontier сверх уже обработанных старейших строк не обеспечивается этой выборкой. [Adaptive compressor][compressor], [Dreamer][dreamer].

Отдельно LifecycleManager удаляет observations старше 30 дней по DEFAULT_CONFIG, кроме типов decision/commit; cap — 50 000. Проверки pinned в DELETE нет. Следовательно, default TTL может удалить запись раньше, чем она дойдёт до 90-дневного distilled tier, а auto-pin по milestone не гарантирует retention. max_db_size_mb объявлен, но cleanup не использует его как отдельное условие ограничения размера. [Lifecycle][lifecycle], [defaults][types].

SessionManager сохраняет snapshot с приоритетами и номинальным лимитом 16 KiB; фактически trimToFit сравнивает serialized.length с 16 384, то есть считает JavaScript code units, а не UTF-8 bytes. После 7 дней restore оставляет P1 и stats. Session chains связывают соседние сессии проекта; startup hook выбирает полное/облегчённое восстановление по временным порогам. Wake-up primer распределяет по умолчанию 700 оценочных tokens: 15% profile, 40% knowledge, 30% recent, 15% entities. Per-prompt и post-tool hooks добавляют небольшие snippets с rate limit/cooldown. Это механизмы continuity, но не единый selector, знающий фактический context window каждого provider. [Sessions][sessions], [wake-up][wake-up], [prompt hook][prompt-hook], [session hook][session-hook].

BudgetManager суммирует tokens_in из token_stats; это учёт ingest/tool usage, а не точное измерение текущего prompt после удаления старых сообщений. overflow modes в коде — warn/aggressive_truncation/hard_stop. getTokenEstimate использует limit 1 000 000; оценка примерно четыре символа/байта на token в разных местах. Это нельзя напрямую переносить в многопровайдерный русский Chat. [Budget][budget], [types][types].

### 1.7. Knowledge maintenance и wiki

Dreamer стартует в Kernel, работает с 30-секундной первоначальной задержкой и далее примерно каждые пять минут. Он помечает знание stale после 30 дней без доступа, архивирует не-explicit после 90, ищет похожие записи, сжимает summaries, объединяет тему в pattern и повышает relevance при обращениях из нескольких сессий. Это детерминированное обслуживание, не автономное доказательное переосмысление. promotionScan выдаёт candidates/logs; сам по себе этот метод не переносит знание в global store. duplicateScan требует globalStore, который в показанном Kernel-конструкторе Dreamer не передан. [Dreamer][dreamer], [Kernel][kernel].

GlobalKnowledgeStore — отдельная SQLite в пользовательском home. promote очищает title/content через privacy engine и создаёт новую global запись с source_project(s). Это cross-project повторное использование в локальном профиле, не модель обмена между principals с независимыми правами. [Global store][global].

VaultSync пишет Markdown entity/topic/session/knowledge/index/answer pages; log.md дополняется. SynthesisEngine собирает до 50 последних неприватных observations, имеет debounce 2 секунды и минимум 3 observations. Без LLM получается временная группировка выдержек; с LLM предыдущая страница и новые observations передаются synthesizer. Это две разные степени synthesis, а не один гарантированно интеллектуальный механизм. [Vault][vault], [Synthesis][synthesis].

README обещает сохранение ручных правок, но refreshEntity/Topic/Session/Knowledge используют полную перезапись. Session page выводит summaries/короткие excerpts, а не весь raw payload. Synthesis page может использовать предыдущую страницу как вход, но также перезаписывается; dependency manifest и сохранение человеческих аннотаций на уровне БД не обеспечены. Поэтому из SQLite можно восстановить проекцию, однако не гарантированно прежний LLM-текст, ручные правки или весь Markdown event log. [Vault][vault], [templates][templates], [Synthesis][synthesis].

### 1.8. Privacy и проверяемость заявлений

PrivacyEngine удаляет private fragments, редактирует известные secrets, email/IP и custom regex; ошибка processing блокирует observation ingest. Положительно, что очищенный текст попадает в обычный Pipeline до hashing, extraction и embedding. Но это не общий mandatory boundary всех write/read paths: KnowledgeBase.save и ContentStore.index сами не прогоняют PrivacyEngine, а их MCP handlers не обеспечивают ту же обработку. В Pipeline private classification не запрещает LLM summarize и entity extraction оставшейся части текста; в обычном BM25/vector нет общего WHERE, исключающего private. Vault session/topic и synthesis имеют explicit private filter, но это локальные защиты. Удалённые теги не возвращаются; проблема — отсутствие единой семантики privacy для всех surviving content/metadata и производных. [Privacy][privacy], [Pipeline][pipeline], [knowledge tools][knowledge-tools], [content store][content-store], [search tools][search-tools].

README/документы содержат сильные заявления о качестве. Их следует читать как отчёт автора по конкретным harness, не как доказательство пригодности к personal agent:
- опубликованный retrieval hit допускает **хотя бы одну** правильную evidence session, даже для multi-evidence вопроса;
- LoCoMo ingest добавляет dataset-provided summaries/observations/events;
- BenchKernel имеет собственные ingest/search paths, прямые INSERT и отличающуюся обработку embeddings; рядом существует RealKernelBench, поэтому нужно указывать harness конкретного результата;
- savings характеризуют размер представления, не корректность последующего ответа или действия.

Эти оговорки не опровергают полезность механизмов; они ограничивают переносимость метрик. [Методология][bench-method], [BenchKernel][bench-adapter], [RealKernelBench][bench-real], [LoCoMo harness][bench-locomo].

## 2. Strong architectural ideas

1. **Разделение сохранённого материала и удобного представления.** Можно иметь подробное evidence и компактные views, не выбирая между «хранить всё в prompt» и «забыть навсегда». Для агента это основа объяснимости. В context-mem этот принцип виден в content/summary, SQLite/vault, но требует более строгой реализации. [Pipeline][pipeline], [Vault][vault].
2. **Retrieval в несколько шагов.** Сначала найти небольшой набор, затем получить точный оригинал по ID. Экономия контекста должна сопровождаться возможностью раскрыть доказательство, а не только более коротким summary. [MCP core][mcp-core], [search tools][search-tools].
3. **Дополняющие lexical и semantic методы.** Имена, команды, пути и версии часто требуют exact/lexical matching; смысловые парафразы требуют другого канала. Независимое получение кандидатов сильнее одной универсальной метрики. Конкретные weights reference не являются принципом. [BM25][bm25], [vector][vector], [Fusion][fusion].
4. **Явная временная изменяемость знания.** valid_from/to и supersession обозначают правильную потребность: «раньше верно» не равно «ошибка». Даже неполная реализация делает эту проблему видимой. [Knowledge tools][knowledge-tools].
5. **Continuity как отдельный продукт памяти.** Незавершённые намерения, ограничения задачи и проверенные шаги нужны при возвращении в работу; они не обязаны становиться вечными facts. [Sessions][sessions], [wake-up][wake-up].
6. **Структурно осведомлённое сокращение.** Для stack trace, JSON, документа и истории действий полезны разные representations. Эффективность нужно оценивать по сохранению нужного смысла, а не проценту удалённого текста. [Summarizers][summarizers].
7. **Inspectability через человекочитаемую wiki.** Пользователь должен понимать, что агент о нём и проектах помнит. Markdown — один удачный способ, но не обязательный authoritative формат. [Wiki schema][wiki-schema], [Vault][vault].
8. **Обслуживание памяти как постоянная ответственность.** Старение, дубликаты, broken references, устаревшие indexes и потеря provenance не устраняются одним хорошим ingest. Dreamer делает эту потребность явной, хотя его эвристики нельзя принимать как критерии истинности. [Dreamer][dreamer].
9. **Локальный baseline и optional дорогие компоненты.** Lexical retrieval и просмотр evidence не должны исчезать при недоступности LLM/embedding provider. Это полезная автономность, если деградация наблюдаема и не снимает privacy/policy ограничения. [Kernel][kernel].

## 3. Weak or questionable ideas

### 3.1. Смешение истинности, авторитета, популярности и свежести

computeAuthority использует source_type, session breadth, access density и recency; computeConfidence — источник, возраст, обращения, распространённость по сессиям и conflict flags. Частота retrieval здесь влияет на доверие. Но многократное извлечение одного ошибочного утверждения не даёт независимых подтверждений. Наблюдение надёжного tool может быть сильнее предположения модели и сильнее высказывания пользователя о состоянии машины; фиксированный порядок explicit > inferred > observed не универсален. [KnowledgeBase][knowledge-base], [Dreamer][dreamer].

Для target следует разнести: author/source authority, evidence quality, extraction uncertainty, factual freshness, task relevance, user importance, usage history. Последние два могут менять отбор контекста, но не превращать hypothesis в fact.

### 3.2. Similarity как contradiction и хронология как causality

Текстовая близость — способ найти пары для анализа. Она не различает дубликат, дополнение, изменение условий, смену предпочтения и несовместимое утверждение. force-save, автоматически закрывающий похожие записи, слишком силён для такой диагностики. Аналогично «после решения возникла ошибка» не означает «ошибка вызвана решением». [KnowledgeBase][knowledge-base], [knowledge tools][knowledge-tools], [trail][trail].

Альтернатива — typed statements с subject/predicate/scope/time и явно записанные evidence links. LLM может предлагать интерпретацию и conflict candidates; авторизация и фиксация причинной связи требуют отдельного основания.

### 3.3. Несогласованные lifecycle гарантии

Pinned защищает от progressive compression, но не от TTL. Default TTL 30 дней конфликтует с ожидаемым долгим переходом к distilled. Старое «не читали» превращается в stale, а архивирование не согласовано с historical retrieval. Это недостаток единого lifecycle contract, а не аргумент против forgetting или hot/warm/cold. [Lifecycle][lifecycle], [Dreamer][dreamer], [compressor][compressor].

### 3.4. Производное знание может стать собственным подтверждением

Автоматическое извлечение и consolidation записывают knowledge без обязательного списка evidence IDs. Search/ask могут сохранить synthesis обратно в knowledge. Если результат модели позже выглядит как независимый источник, возникает цикл самоподтверждения. Перечень snippets на отдельной answer page не заменяет обязательный ancestry для каждого claim. [Kernel][kernel], [Dreamer][dreamer], [search tools][search-tools].

### 3.5. Ограничения глубины истории и языка

Последние 5000 vectors, первые 50 knowledge candidates в некоторых проверках, 50 observations для synthesis и агрессивная recency bias приемлемы как локальные пределы прототипа. Для многолетней памяти они могут систематически скрывать редкие старые сведения. Кеш с потерей кириллицы, English keyword taxonomy и regex имён показывают, почему нужны отдельные многоязычные требования. [Vector][vector], [KnowledgeBase][knowledge-base], [Synthesis][synthesis], [Fusion][fusion], [entities][entities], [topics][topics].

### 3.6. Разрастание интерфейса без общего контракта

Много tools не означает более полноценную память. Прямые DB reads в hooks, разные knowledge/query handlers, local/global stores и wiki paths увеличивают число мест, где должны совпадать temporal/privacy правила. Unified interface — хорошая идея, но aliases и общая кнопка search сами по себе не унифицируют семантику. Для SlavikAI стоит обсуждать несколько ясных memory operations и единые guards, не копировать каталог инструментария. [Search tools][search-tools], [prompt hook][prompt-hook], [global store][global].

### 3.7. Wiki одновременно как cache и место редактирования

Если человек пишет содержательные правки в производную страницу, эти правки уже являются авторским источником. Полная перегенерация может их уничтожить; чтение предыдущей LLM synthesis может перенести старую ошибку в новую. Нужен выбор между regenerable view, отдельным слоем аннотаций и полноценным authored knowledge document. Нельзя обещать все три свойства без контракта владения. [Vault][vault], [Synthesis][synthesis].

### 3.8. Защитные механизмы нельзя делать best-effort

Для embeddings или красивой wiki приемлема потеря secondary feature при ошибке с диагностикой. Для privacy, authorization, temporal exclusion отозванного знания и удаления личных данных такой режим неприемлем. Reference содержит много non-critical catch и локальные privacy checks; переносить этот стиль на personal memory enforcement нельзя. [Pipeline][pipeline], [search tools][search-tools], [knowledge tools][knowledge-tools].

## 4. Relevance to SlavikAI

Ниже каждая идея оценивается по потребностям полноценного personal agent. «Базовая» означает необходимость самого свойства, а не выбор алгоритма reference. Возможная роль — предложение для обсуждения, не принятое решение.

### 4.1. Raw observations и immutable evidence

- **Проблема:** после сжатия, ошибки модели или изменения внешнего источника нужно установить, что действительно наблюдалось.
- **Принцип:** версионированное свидетельство отделено от интерпретации.
- **context-mem:** content + metadata + hash; private redaction до записи; summary рядом; dedup и TTL ограничивают полноту истории. [Pipeline][pipeline], [Lifecycle][lifecycle].
- **Роль/необходимость:** базовая evidence memory для Chat, tools, Desktop, документов и мультимодальных наблюдений.
- **Риски/альтернативы:** payload-level dedup допустим, event-level dedup по тексту опасен. «Immutable» должно означать неизменность принятой версии в пределах retention policy, а не запрет пользователю удалить личные данные. Для недопустимых к сохранению raw нужны redacted evidence и честная отметка полноты.

### 4.2. Authoritative source и derived representation

- **Проблема:** agent начинает доверять summary больше, чем исходнику.
- **Принцип:** authority относится к происхождению и предмету утверждения; derived view не приобретает её автоматически.
- **context-mem:** SQLite primary, Markdown derived; исходный content и summary разделены, но ancestry LLM synthesis неполон. [Vault][vault], [Synthesis][synthesis].
- **Роль/необходимость:** базовое разделение источника, сохранённой копии, claim и retrieval view.
- **Риски/альтернативы:** primary DB authoritative для «что сохранено», но не для «что сейчас происходит в мире». Для environment facts текущий проверяемый API/система может иметь приоритет над любой старой памятью.

### 4.3. Canonical knowledge и типизация

- **Проблема:** сырая история неудобна для устойчивого знания, но одно общее поле fact смешивает несовместимые смыслы.
- **Принцип:** durable knowledge имеет тип, область применимости и статус принятия.
- **context-mem:** categories и source_type, преимущественно coding taxonomy; текст knowledge допускает synthesis/answers. [Types][types], [KnowledgeBase][knowledge-base].
- **Роль/необходимость:** базовая модель fact/preference/decision/constraint/environment/observation плюс hypothesis, inference, procedure, commitment.
- **Риски/альтернативы:** строгая типизация полезна для жизненного цикла, но вся информация не обязана быть атомарной тройкой. Нужны и утверждения, и документы с выделенными claims.

### 4.4. Provenance / evidence chain

- **Проблема:** невозможно объяснить вывод и пересмотреть его после отзыва источника.
- **Принцип:** любое производное утверждение имеет проверяемые входы и историю преобразований.
- **context-mem:** source/session/correlation metadata, отдельные source IDs, частично эвристический decision trail. [Pipeline][pipeline], [trail][trail].
- **Роль/необходимость:** базовая evidence ancestry от источника до claim, summary и ответа.
- **Риски/альтернативы:** не требовать раскрытия внутренних рассуждений LLM. Достаточны evidence references, публичное обоснование решения, версии преобразований и explicit assumptions. Общий предок означает зависимое, а не независимое подтверждение.

### 4.5. Temporal facts и интервалы

- **Проблема:** «что сейчас», «что было тогда» и «что мы знали тогда» дают разные ответы.
- **Принцип:** valid time отделён от recorded time.
- **context-mem:** valid_from/to и indexed_at/created_at есть, но coherent bitemporal semantics нет; null/time filtering трактуются упрощённо. [Knowledge tools][knowledge-tools], [time travel][time-travel].
- **Роль/необходимость:** базовая поддержка смены предпочтений, адресов, проектов, решений и environment state.
- **Риски/альтернативы:** неизвестное время не равно epoch. Нужны timezone, interval uncertainty и различение scheduled future change/подтверждённого события. Полная bitemporal детализация может быть разной для разных классов данных.

### 4.6. Supersession, corrections и conflicts

- **Проблема:** обновление затирает историю или сохраняет два несовместимых «текущих» значения.
- **Принцип:** change, correction, retraction и disagreement — разные операции.
- **context-mem:** force-save, archive/tag/graph supersede, merge и keep_both с разной семантикой. [Knowledge tools][knowledge-tools].
- **Роль/необходимость:** базовая история ревизий и нерешённых конфликтов.
- **Риски/альтернативы:** новый факт не всегда отменяет старый: разные scopes/времена могут сосуществовать. Merge должен сохранять обе evidence chains; user correction не должна исчезать в очередной synthesis.

### 4.7. Stale knowledge

- **Проблема:** правильное в прошлом знание используется для сегодняшнего действия.
- **Принцип:** freshness определяется изменяемостью предмета и временем последней проверки.
- **context-mem:** stale выводится из отсутствия доступа 30 дней, recency — из возраста/обращений. [Dreamer][dreamer], [KnowledgeBase][knowledge-base].
- **Роль/необходимость:** базовый refresh policy, особенно перед Desktop actions.
- **Риски/альтернативы:** чтение не освежает факт; старое решение может оставаться правильным годами. Нужны verified_at, expiry/refresh rules и source version, отдельно от last_accessed.

### 4.8. Hot / warm / cold memory

- **Проблема:** миллионы сохранённых событий невозможно постоянно держать в context.
- **Принцип:** скорость доступа, representation detail и retention — отдельные оси.
- **context-mem:** primer/snapshot/short snippets и persistent SQLite позволяют увидеть несколько уровней доступа; explicit hot/warm/cold storage contract отсутствует. Compression tiers не являются такой иерархией. [wake-up][wake-up], [sessions][sessions], [compressor][compressor].
- **Роль/необходимость:** базовое разделение активного рабочего набора, подготовленных summaries/indexes и подробного архива; физический tiering зависит от масштаба.
- **Риски/альтернативы:** cold не означает менее достоверное. Редкое знание может быть критическим; выбранное cold evidence может временно стать hot без изменения своего статуса.

### 4.9. Cache и derived indexes

- **Проблема:** повторный retrieval дорог, но cached result может пережить correction или изменение прав.
- **Принцип:** кеш и индекс воспроизводимы, версионированы и никогда не расширяют доступ.
- **context-mem:** FTS triggers, embedding BLOB, 30-секундный query cache; source-only dedup в ContentStore может оставить старую версию документа. [Schema][schema], [Fusion][fusion], [content store][content-store].
- **Роль/необходимость:** indexes базовые; cache полезен по измеряемой стоимости.
- **Риски/альтернативы:** ключ включает principal, scope, filters, time, knowledge/policy revision и модель embedding. Удаление/отзыв доступа инвалидирует выдачу немедленно, а не только по TTL. Следует хранить coverage/freshness индекса.

### 4.10. Session memory и cross-session continuity

- **Проблема:** новая сессия теряет цель, ограничения и незавершённые обязательства.
- **Принцип:** task continuity существует отдельно от истории сообщений и долговременного профиля.
- **context-mem:** prioritized snapshot, parent session chain, auto-restore thresholds, precompact recovery. [Sessions][sessions], [session hook][session-hook].
- **Роль/необходимость:** базовое рабочее состояние длительной задачи, в том числе между providers и Chat/Agent/Desktop.
- **Риски/альтернативы:** ближайшая по времени сессия не обязательно нужная задача. Нужны explicit task/session relationships, branch/fork lineage и concurrency semantics. Старый session approval не переносится через summary.

### 4.11. Automatic memory capture

- **Проблема:** manual-only capture теряет события и создаёт нагрузку на пользователя.
- **Принцип:** автоматизация сбора не равна автоматическому признанию истинности и не равна бесконтрольной retention.
- **context-mem:** post-tool hooks, Kernel auto-extraction decision/error/commit/frequent-file, Dreamer consolidation. [Capture hook][capture-hook], [Kernel][kernel], [Dreamer][dreamer].
- **Роль/необходимость:** автоматический сбор разрешённых evidence и образование candidates очень полезны для personal agent; конкретная политика durable writes открыта.
- **Риски/альтернативы:** raw recording, transient candidates, принятие canonical claim и cross-project sharing требуют разных разрешений. Возможны per-item confirm, заранее выданный scoped opt-in или смешанная политика. Текущий запрет auto-writes SlavikAI не отменяется этим исследованием.

### 4.12. Explicit / confirmed capture

- **Проблема:** агент записывает чужую цитату, шутку или своё предположение как предпочтение пользователя.
- **Принцип:** прямое высказывание, просьба запомнить и подтверждение конкретной записи — разные события.
- **context-mem:** caller передаёт source_type=explicit, save_knowledge доступен как tool; это не самостоятельный consent mechanism. [Knowledge tools][knowledge-tools].
- **Роль/необходимость:** базовый inspect/confirm/edit/reject опыт для персонального профиля и спорных утверждений.
- **Риски/альтернативы:** нельзя запрашивать одно и то же согласие постоянно. Scope заранее выданного согласия должен быть понятен, отзываем и не выводиться из содержимого источника.

### 4.13. Lexical / BM25 retrieval

- **Проблема:** vector similarity теряет точные имена, ошибки, даты и идентификаторы.
- **Принцип:** точный и лексический поиск — самостоятельная сильная сторона.
- **context-mem:** multi-query FTS5, phrase/entity/synonym варианты, separate verbatim content index. [BM25][bm25], [schema][schema].
- **Роль/необходимость:** базовый канал как для project knowledge, так и личных документов.
- **Риски/альтернативы:** English stemming/словари не обеспечивают русский поиск. Выбор анализатора, identifiers, морфология и multilingual relevance нуждаются в собственных оценках; восемь стратегий не обязательны.

### 4.14. Vector retrieval

- **Проблема:** вопрос и evidence описывают один смысл разными словами.
- **Принцип:** semantic candidate generation дополняет lexical.
- **context-mem:** local nomic embeddings, scan последних 5000 записей, обычно embedding summary. [Vector][vector], [embedder][embedder].
- **Роль/необходимость:** высокая для personal conversations и разнородных источников; конкретный vector backend не фиксируется.
- **Риски/альтернативы:** summary-only embedding теряет детали. Возможны несколько representations одного evidence, exact scan для малого scope или ANN для большого. Нельзя ограничивать весь semantic recall только недавней историей.

### 4.15. Hybrid retrieval и reranking

- **Проблема:** один канал выдаёт слишком узкие либо шумные кандидаты.
- **Принцип:** объединение независимых candidate generators и отдельный reranking.
- **context-mem:** weighted fusion, intent/IDF/recency/access; optional shared LLM judge. [Fusion][fusion], [judge][judge].
- **Роль/необходимость:** высокая; lexical+vector — обоснованный кандидат, entity/temporal paths дополняют его.
- **Риски/альтернативы:** нужно сравнить score calibration, rank fusion и trained reranker; LLM reranking добавляет latency/privacy расходы. Reranker не может извлечь отсутствующий candidate или исправить неправильные ACL.

### 4.16. Relevance / importance / recency / feedback

- **Проблема:** полезный context не равен просто самым новым похожим записям.
- **Принцип:** несколько сигналов выбора должны оставаться объяснимыми.
- **context-mem:** type/keyword importance, auto-pin, age/access ranking, usefulness по последующему изменению упомянутого файла. [Importance][importance], [Fusion][fusion], [feedback][feedback].
- **Роль/необходимость:** базовое различение пользовательской важности, полезности для задачи и свежести.
- **Риски/альтернативы:** file touch не доказывает пользу, отсутствие feedback — unknown. Prefer explicit outcome/feedback и причинно связанный task result; модель рейтинга не должна учиться считать часто повторяемую ошибку истиной.

### 4.17. Runtime context selection и token budgeting

- **Проблема:** даже хорошие найденные документы не помещаются в prompt и конкурируют с текущей задачей.
- **Принцип:** selection — отдельная constrained optimization с обязательными элементами.
- **context-mem:** fixed primer shares, snapshot priorities, snippet limits, usage counters. [wake-up][wake-up], [sessions][sessions], [budget][budget].
- **Роль/необходимость:** базовый provider-aware context assembly, отдельно от persistence.
- **Риски/альтернативы:** учитывать instructions, tool schemas, messages, output reserve и ожидаемые tool returns. Сначала обязательные ограничения и decision state, затем набор достаточных evidence, а не просто top-k. Неуспех укладки должен быть наблюдаемым.

### 4.18. Adaptive compression и protected/pinned knowledge

- **Проблема:** подробные материалы вытесняют нужные сведения; обычная truncation обрывает исключения и условия.
- **Принцип:** выбор representation по задаче, структурная компрессия, возможность раскрытия оригинала.
- **context-mem:** summarizers, four age tiers, pinned bypass, keyword rules. [Compressor][compressor], [Pipeline][pipeline], [summarizers][summarizers].
- **Роль/необходимость:** высокая для очень длинных взаимодействий; protected constraints базовые.
- **Риски/альтернативы:** pin означает что именно — сохранить evidence, оставить без потерь, предпочитать при retrieval или всегда включать? Это четыре разных свойства. Age-only compression хуже task-aware; отрицания, числа, units, uncertainty, identities и provenance нельзя терять.

### 4.19. Topic synthesis

- **Проблема:** нужно понять развитие темы через сотни событий.
- **Принцип:** обзор объединяет свидетельства, но сохраняет различия времени, мнений и нерешённых вопросов.
- **context-mem:** regex topic taxonomy, deterministic timeline или LLM synthesis из последних 50 observations; Dreamer также создаёт pattern из summaries. [Topics][topics], [Synthesis][synthesis], [Dreamer][dreamer].
- **Роль/необходимость:** высокая для проектов, отношений, длительных целей и личных тем.
- **Риски/альтернативы:** topic overview должен перечислять evidence dependencies и coverage period. Для многолетней истории нужны иерархические views, а не вечная пересуммаризация предыдущего текста без проверки источников.

### 4.20. Knowledge graph и entity extraction

- **Проблема:** разные имена относятся к одному объекту, а один текстовый ярлык — к разным.
- **Принцип:** устойчивые identity и typed relations позволяют связывать знания вне lexical similarity.
- **context-mem:** curated tech aliases, path/CamelCase/ALL_CAPS/person regex, typed graph и metadata. [Entities][entities], [graph][graph], [schema][schema].
- **Роль/необходимость:** identity базовая; ширина graph retrieval зависит от полезности конкретных связей.
- **Риски/альтернативы:** нужны disambiguation, context-dependent aliases, multilingual names, раздельные люди/аккаунты/устройства. LLM extraction может предлагать edges, но uncertainty и provenance сохраняются. Не обязательно превращать каждый token и эпизод в graph node.

### 4.21. Decision history

- **Проблема:** агент повторяет отвергнутые действия и забывает, на каких условиях был сделан выбор.
- **Принцип:** решение хранит автора, варианты, rationale, область действия и evidence на момент принятия.
- **context-mem:** decision observations/knowledge, event timeline, heuristic explain_decision, ADR-style narrative. [Trail][trail], [narrative][narrative].
- **Роль/необходимость:** базовая для personal agent, действующего во внешней системе.
- **Риски/альтернативы:** реконструкция может предлагать plausible context, но должна маркироваться. Decision и authorization раздельны: предпочтение или прежнее решение не разрешает сегодняшнее опасное действие.

### 4.22. Human-readable wiki / Markdown

- **Проблема:** пользователю трудно проверить и исправить скрытую память.
- **Принцип:** знание должно иметь читаемое представление и ссылки на evidence.
- **context-mem:** vault directories, backlinks, index, synthesis, answer pages, Obsidian-friendly links. [Wiki schema][wiki-schema], [Vault][vault].
- **Роль/необходимость:** inspectability базовая, Markdown как формат — полезный, но необязательный выбор.
- **Риски/альтернативы:** derived view + отдельные annotations либо authored documents с versions. Автоматический markdown export расширяет поверхность утечки и должен подчиняться тем же scope/retention правилам.

### 4.23. Privacy / redaction

- **Проблема:** личная память может раскрыть чувствительные данные через поиск, модель, export или ошибочные связи.
- **Принцип:** identity, consent, audience, egress и data minimization проходят через все этапы.
- **context-mem:** локальное хранилище, regex redaction, private flags/частичные filters; optional external LLM. [Privacy][privacy], [Pipeline][pipeline], [LLM factory][llm-factory].
- **Роль/необходимость:** базовая защита principals, личной и проектной информации, допустимых provider destinations.
- **Риски/альтернативы:** «локально» не равно «приватно», redaction не равна ACL. Тотальное удаление email/IP может мешать legitimate personal use; нужны scoped sensitive fields или ссылки на защищённые записи без копирования secrets в context.

### 4.24. Garbage collection / forgetting / maintenance

- **Проблема:** бесконечная история увеличивает стоимость, утечки и число устаревших производных.
- **Принцип:** контролируемое forgetting с учётом зависимостей, а не только возраста.
- **context-mem:** TTL/count cleanup, private cleanup on stop, archive, progressive compression, loss predictor и background scans. [Lifecycle][lifecycle], [Dreamer][dreamer], [pressure][pressure].
- **Роль/необходимость:** базовая long-term stewardship: удалить по просьбе, ослабить retrieval, архивировать или пересчитать — разные действия.
- **Риски/альтернативы:** удаление evidence требует обхода dependent claims/views/indexes/caches/exports. Archive обратим, физическое удаление — нет. После удаления нельзя оставлять доступную synthesis с тем же приватным содержимым и обещать, что оно забыто.

## 5. Gaps for a personal agent

Ниже перечислено то, что не установлено как полноценный сквозной механизм reference и понадобится определить SlavikAI сверх него.

| Недостающая способность | Зачем personal agent |
|---|---|
| Principal + scope + audience на каждом объекте и query path | Закрытая owner/member группа всё равно требует реальной изоляции, включая graph, caches, exports и summaries; local project directory её не заменяет |
| Эпистемическая модель | «Пользователь сказал», «tool показал», «модель предположила», «подтверждено» и «устарело» должны различаться независимо от типа контента |
| Claim-level provenance и dependency tracking | Объяснять основание, отслеживать независимость evidence, каскадно отзывать производные |
| Coherent valid/recorded time и revision history | Восстановить как прошлое мира, так и прошлое убеждений; учитывать запоздалые исправления |
| Reliable action episodes | Разделять intent, requested action, attempted execution, tool result, verified outcome и rollback; кодовый diff покрывает лишь один частный случай |
| Source freshness / environment identity | PID, окно, файл, аккаунт и устройство должны иметь устойчивую identity/version; прошлый success не гарантирует настоящее состояние |
| Consent/capture/retention policies | Разделить сбор telemetry, долговременное хранение, inference профиля и раскрытие другому provider |
| Durable goals, commitments и task continuity | Сессия не является всей жизнью задачи; обязательства могут пережить Chat, provider и перезапуск |
| Multi-provider context contract | Реальный token budget, modality limits, unavailable features и разрешённый egress зависят от provider/model |
| Многоязычная и мультимодальная память | Русская речь, transliteration, OCR, изображения, audio/video artifacts требуют исходников и версии извлечённого текста |
| Память о людях и отношениях | Имена неоднозначны; разные люди имеют разные preferences/consents. Технологический gazetteer не решает эту задачу |
| Governance человеческих правок | Редактирование памяти должно создавать attributable correction/annotation, а не теряться при wiki refresh |
| Long-horizon retrieval и paging | Миллионы событий, редкие старые facts, полный набор evidence для multi-hop вопроса; finite recent scans недостаточны |
| Context manifest и abstention | Нужно объяснять, на каком знании построен ответ, какие ограничения/конфликты не разрешены и когда доказательств недостаточно |
| Memory poisoning / instruction separation | Веб-страницы, письма и документы не получают право менять инструкции, preferences или approvals просто из-за попадания в память |
| Retraction / erasure / backup policy | Забывание должно учитывать копии, derived artifacts и восстановление из backup; отдельного DELETE основной строки мало |
| Measurable maintenance correctness | Работа фонового обслуживания должна иметь coverage, retry state, drift metrics и bounded budgets, а не бесконечно смотреть одни старые записи |
| Независимая оценка качества | Нужны exact evidence recall, all-evidence coverage, temporal correctness, attribution, retention/erasure, multilingual и downstream action correctness |

Это перечень требований целевой модели, а не предложения отдельных PR. Он не утверждает, что SlavikAI уже реализует перечисленное. Важные исходные принципы проекта — principal isolation, единый tool gateway, независимые approvals и personal-agent scope — подтверждены архитектурными документами, а устройство текущих memory DB не ограничивает возможные решения. [ARCH_CANON](ARCH_CANON.md), [Architecture](Architecture.md).

## 6. Candidate target memory model

**Это архитектурная гипотеза для обсуждения.** Она описывает виды информации, отношения и контракты. Она не задаёт классы, имена будущих файлов, БД, libraries или порядок внедрения.

### 6.1. Четыре независимые оси

Любая информация описывается сразу по нескольким осям:

1. **Смысл:** evidence, claim, preference, decision, constraint, task state, procedure, derived view.
2. **Область:** principal, personal/project/organization, task/session, environment/device, audience.
3. **Эпистемическое состояние:** reported/observed/inferred, candidate/accepted/disputed/retracted; происхождение и качество подтверждения.
4. **Жизненный цикл:** valid/recorded time, freshness, retention, protection, hot/warm/cold availability.

Не следует кодировать всё одним enum «тип памяти». Например: предпочтение пользователя может быть подтверждённым, проектным, действующим только в командировке и редко извлекаемым. Это всё ещё preference; cold storage не превращает его в менее истинное утверждение.

### 6.2. Виды информации

| Вид | Содержание | Что нельзя подразумевать автоматически |
|---|---|---|
| Observation / evidence | Кто, где, когда и каким способом получил конкретный payload или результат | Наблюдение утверждения не доказывает его истинность |
| Reported fact | Кто-то сообщил утверждение о мире | Авторитет говорящего зависит от предмета; цитата не становится заявлением текущего пользователя |
| Verified fact / accepted claim | Утверждение, принятое по установленному evidence/verification policy | Не вечная истина; содержит scope, время и условия проверки |
| Preference | Выбор пользователя, явный либо предположенный | Preference не является authorization и не обязана действовать во всех задачах |
| Decision | Автор выбора, варианты, rationale, ограничения, evidence, время и последствия | Выбор не означает исполнения; отвергнутая альтернатива не является ошибочным фактом |
| Constraint | Обязательное ограничение задачи/пользователя с источником полномочий и scope | Текст constraint в чужом документе не становится доверенной инструкцией |
| Environment state | Наблюдение состояния устройства, приложения, процесса, файла, аккаунта | Состояние быстро меняется; нужны identity/version и freshness |
| Hypothesis / inference | Предположение или производный вывод с основаниями | Уверенность модели и повторение не равны проверке |
| Task / commitment | Цель, обещание, deadline, статус, блокер, выполненные и проверенные шаги | Завершение сессии не завершает задачу; planned не равно done |
| Procedure / experience | Способ действия, prerequisites, область успешного применения, результаты | Прошлый успех не разрешает новый запуск и не гарантирует успех в другой среде |
| Source document / artifact | Версия документа, снимок страницы, аудио, изображение, tool artifact | Извлечённый OCR/transcript не равен первичному payload |
| Synthesis / context view | Сжатое представление выбранных inputs | Не новый независимый evidence; не повышает authority своих предков |

Canonical knowledge — это управляемые утверждения, решения и preferences с versioned state. «Canonical» означает согласованный способ фиксировать принятое знание и спорные состояния, **не монолитную единственную правду**. Disputed claims остаются видимыми как disputed.

### 6.3. Логические слои

~~~text
                   PRINCIPAL / SCOPE / CONSENT / POLICY
             действует на ingest, derivation, retrieval, export
                                    |
  External sources                  |             User statements
  tools / files / apps / web         |             corrections / decisions
              \                     |                    /
               +------- capture boundary ---------------+
                                    |
                        EVIDENCE / EPISODIC RECORD
                 source versions, occurrences, tool outcomes
                         /                  \
                        /                    \
              CLAIM CANDIDATES          TASK / SESSION STATE
              links + uncertainty       goals, steps, blockers
                        |                    |
                acceptance policy           |
                        |                    |
                  KNOWLEDGE RECORD          |
            revisions, conflicts, decisions |
                  \             |           /
                   \            |          /
                    DERIVED REPRESENTATIONS
             summaries / topic views / wiki / FTS / vectors
                                    |
                       RETRIEVAL + CONTEXT SELECTION
                  authorized, temporal, sufficient evidence
                                    |
                     BOUNDED RUNTIME CONTEXT PACKAGE
                                    |
                    shared provider-neutral agent runtime
                                    |
                       gateway / approvals / verification
                                    |
                           new observed outcomes
~~~

Policy/approval record образует отдельную авторитетную область. Memory может ссылаться на решение об approval для audit, но не восстанавливать действующее разрешение из истории диалога. Graph является представлением отношений между этими слоями, а не альтернативным source of truth, который разрешено обновлять в обход knowledge lifecycle.

### 6.4. Что authoritative, а что derived

**Evidence record authoritative для факта наблюдения:** система видела такой ответ tool в такое время при таких параметрах. Если tool сообщил ошибочное значение, запись наблюдения всё равно корректна; claim о мире может оказаться ложным.

**Версия внешнего источника authoritative в своей области:** например, проверенный ответ приложения об объекте и его version. Сохранённая копия помогает отвечать о прошлом; для текущего изменяемого состояния может потребоваться новый запрос к источнику.

**Knowledge record authoritative для принятой системой версии утверждения:** кто и на каких основаниях принял, оспорил или исправил запись. Он не должен стирать историю evidence.

**User decision/confirmation authoritative для выраженного намерения пользователя:** кто подтвердил что именно, для какой области и срока. Это не равнозначно тому, что любое высказывание пользователя о внешнем мире достовернее измерения.

**Policy store authoritative для разрешений:** текущие scopes, revocation, once/session/persistent semantics. Память о разрешении — только audit reference.

**Derived:** embeddings, FTS, search cache, inferred entity mentions, сжатые summaries, topic synthesis, wiki views и runtime context. Для каждого нужны source revision/IDs, generator/model/policy version, coverage и статус актуальности. Human-authored document/annotation является отдельным источником; смешивать его с полностью regenerable view нельзя.

Простой критерий: после удаления derived artifact его можно пересоздать из разрешённых surviving inputs. Если теряется уникальная авторская информация, это не только cache и требует собственного хранения.

### 6.5. Capture, принятие знания и согласие

Предлагается различать четыре независимых решения:

| Решение | Пример |
|---|---|
| Разрешено ли вообще зафиксировать событие? | Сохранить факт исполнения tool с минимальными metadata; не сохранять секретный clipboard payload |
| Разрешено ли удерживать подробное evidence после run? | Срок хранения screenshot отличается от подтверждённой preference |
| Можно ли превратить информацию в durable accepted claim? | Выделить preference из разговора и подтвердить её, либо применить заранее заданный scoped policy |
| Где можно использовать/раскрывать это знание? | Только personal scope, конкретный project или provider с допустимым egress |

~~~text
observed input
    |
policy permits recording? ---- no ----> transient processing / discard
    |
   yes
    |
source occurrence + permitted payload / redacted representation
    |
structured candidate + source spans + uncertainty
    |
    +--> insufficient evidence ------------> unresolved candidate
    +--> conflict -------------------------> disputed set / review
    +--> explicit confirmation ------------> accepted revision
    +--> previously authorized capture rule -> accepted revision in its scope
~~~

Последняя ветвь — **кандидатное будущее решение**, а не существующая возможность SlavikAI. Действующий контракт требует confirm/edit_and_confirm и запрещает скрытые Memory writes. Чтобы обсуждать automatic recording как часть будущего personal agent, нужно сначала определить, является ли consent на историю/evidence отдельным от consent на canonical knowledge. Недопустимо просто переименовать запрещённую запись в «cache» и считать противоречие снятым.

Extraction из текста может быть модельной и недетерминированной; commitment результата должен получать структурированный candidate. Доверенный runtime выводит principal и author из проверенного источника события, а не из строки, написанной извлекающей моделью.

**Automatic capture нужен, но granular:** повторное наблюдение сохраняется как occurrence; identical payload можно разделить между occurrence. Extracted candidate не обязан немедленно попадать в профиль. Инициативное обнаружение полезной информации не даёт права сохранять её вечно или отправлять наружу.

### 6.6. Время, исправление и конфликт

Минимальные концепты времени:

- **event/observed time:** когда событие произошло или было измерено;
- **recorded time:** когда система получила и зафиксировала свидетельство/ревизию;
- **valid interval:** когда утверждение относится к миру;
- **verified time + freshness policy:** когда состояние проверяли и достаточно ли этого для текущей задачи;
- **precision/timezone/uncertainty:** точный timestamp, день, интервал или неизвестное время.

valid interval удобно понимать как [valid_from, valid_to), но открытые/неизвестные границы должны иметь различимые значения. Доходящее позже свидетельство о прошлом не должно задним числом менять ответ на вопрос «что агент знал раньше».

**Смена предпочтения:**

~~~text
1 июня: пользователь подтвердил «предпочитаю утренние встречи»
  claim A: valid from 1 июня, recorded 1 июня

10 июля: пользователь сообщил «с 1 июля предпочитаю встречи после обеда»
  evidence B: observed/recorded 10 июля
  claim B: valid from 1 июля
  revision: A больше не действует с 1 июля

«Что предпочитает 5 июля, по нынешним сведениям?» -> после обеда
«Что мы знали 5 июля?»                           -> утренние встречи
~~~

Это change/supersession, а не объявление июньской записи ложной. Если пользователь затем поясняет, что речь шла только о командировках, создаётся correction scope; глобальную preference нельзя автоматически заменить узкой.

**Исправление ошибки:**

~~~text
evidence: OCR прочитал номер как 18
candidate/claim A: номер 18, с ссылкой на OCR и изображение

user correction + original image review: номер 13
new claim B; A retracted/corrected; причина и actor сохранены
derived summaries/indexes/context caches A -> invalidated
~~~

Оригинал OCR остаётся свидетельством ошибки распознавания, пока его хранение разрешено. Знание о номере исправляется. Нет необходимости переписывать исходное изображение или выдавать старую ошибку за прошлое корректное состояние.

**Нерешённый конфликт:** две версии от разных источников сохраняются с условиями, временем и evidence. В runtime context попадает конфликт или явно обоснованный выбор, а не молча победившая запись с большим access_count. Similarity только помогает найти потенциальную пару.

### 6.7. Provenance как граф зависимостей

Для source/evidence: origin locator и version/hash, actor/source identity, capture method, event/recorded time, session/task/run/tool-call correlation, completeness, redaction policy и allowed audience.

Для claim/revision: evidence IDs и точные spans/regions, transformation type, extractor/model version, acceptance actor/policy, relation к предыдущей revision, confidence/uncertainty с понятной семантикой.

Для decision: explicit rationale и alternatives, constraint versions, evidence, decision maker, applicability, supersession/revocation и ссылки на последующие verified outcomes.

Для derived view/context: input revision IDs, coverage, generated time, generator version, truncation/omission markers, authorization scope. Повторная synthesis не создаёт независимого evidence. Наблюдение самого ответа модели фиксирует «модель ответила X», а не «X подтверждён внешним миром».

~~~text
source revision S
    -> evidence occurrence E
        -> extracted claim C (candidate)
            -> acceptance/revision R
                -> synthesis V
                    -> runtime context K
                        -> answer/action rationale A

correction of R -> invalidate/rebuild V, K caches
erasure of E    -> inspect every dependent C/R/V/A copy allowed to retain
~~~

Это dependency graph, не требование единственного graph database. Ручная annotation связывается с нужной revision и хранится как authored source.

### 6.8. Hot / warm / cold и правила retention

| Уровень | Что находится | Lifecycle |
|---|---|---|
| Hot working context | Текущая цель, ограничения, последние observations, selected evidence, pending decisions | Один run или активная часть задачи; строго ограничен бюджетом |
| Warm prepared memory | Task checkpoints, актуальные topic/entity views, recent claims, summaries и indexes | Обновляется при source/claim revision и смене scope; воспроизводимая часть может удаляться |
| Cold evidence/history | Разрешённые подробные источники, прошлые revisions, старые episodes и archived decisions | Доступ по запросу; retention по классу данных, consent и зависимостям |
| Policy/identity boundary | Действующие полномочия и право доступа | Не определяется популярностью и не вытесняется ranking |

Hot/warm/cold — доступность, а не новая эпистемическая таксономия. Accepted claim может находиться в cold; hot hypothesis остаётся hypothesis.

Раздельные защиты:
- retention-protected: не удалять автоматически;
- exactness-protected: не подменять summary там, где нужна точная формулировка;
- retrieval-preferred: повышать приоритет в применимом scope;
- context-mandatory: включать при выполнении соответствующей задачи.

Ни одна защита не даёт доступа чужому principal и не блокирует явный пользовательский запрос на забывание. Если mandatory context не помещается, runtime обязан показать ограничение, сузить задачу или использовать подходящий контекстный бюджет; молчаливое выбрасывание ограничений недопустимо.

### 6.9. Retrieval: от вопроса к достаточным доказательствам

Предлагаемый conceptual flow:

1. **Определить query frame:** principal, task/project, разрешённые источники, язык, intent, current/historical/as-known-at, freshness и допустимый provider egress.
2. **Выбрать eligible corpus:** permissions, deletion/retraction state и temporal scope применяются до выдачи кандидатов и до передачи внешнему reranker.
3. **Получить кандидатов независимо:** lexical/exact, vector, entity/relationship, temporal, task continuity. Расширение запроса не должно менять исходные ограничения.
4. **Объединить representations:** одна evidence revision, найденная через FTS, vector и wiki, не считается тремя подтверждениями.
5. **Rerank:** query relevance, evidence quality, applicable freshness, user importance, diversity и стоимость представления. Confidence остаётся отдельным полем.
6. **Дополнить связанный evidence:** пройти к первоисточнику, необходимым соседним событиям, decision constraints и конфликтующим revisions.
7. **Проверить достаточность:** multi-hop вопрос требует весь набор доказательств, а не один удачный hit. При отсутствии ответа сохраняется unknown/abstention.
8. **Вернуть evidence package:** claims + source refs + status/time/scope + варианты компактного/полного представления.

Graph expansion должен быть bounded: ограничение глубины, объёма и разрешённых relation types. Исторический запрос не должен автоматически предпочитать newest; вопрос о текущем состоянии — не должен игнорировать freshness только потому, что старый текст очень похож.

Отказ vector/LLM reranker допускает явную деградацию качества, но не обход temporal/privacy filters. Если источник сейчас доступен и состояние изменяемо, retrieval может привести к запросу нового observation; это tool action с обычными ограничениями runtime.

### 6.10. Runtime context и многопровайдерность

Retrieval даёт набор кандидатов. Context selection формирует конкретный пакет для конкретного запуска.

~~~text
model input budget
  - trusted instructions and applicable constraints
  - current user request and task state
  - tool schemas / protocol overhead
  - output reserve and expected tool-observation reserve
  = available budget for memory representations
~~~

Резервы выбираются по capability profile модели, а token estimate проверяется её tokenizer либо консервативным измеряемым оценщиком. Число символов/4 не является универсальной оценкой русского текста, кода, изображений и structured tool data.

Приоритеты selection:
- действующие ограничения, незавершённый decision/approval state и текущая цель;
- проверенные свежие результаты для текущего шага;
- необходимые facts/preferences/decisions с provenance;
- unresolved conflicts и caveats;
- только затем optional summaries, соседние темы и background context.

Внутри бюджета выбирается набор, покрывающий вопрос, без дублирования одного источника. Для каждого элемента доступны representations: exact span, structured claim, short summary, document overview. Compression обязана сохранять предмет, отрицания, единицы измерения, условия, temporal qualifiers и source refs. Запись uncertainty или «не проверено» нельзя удалять ради экономии tokens.

**Context manifest** фиксирует включённые revision IDs, причины выбора, token cost, актуальность, исключённые конфликтные/недоступные сведения и применённые сокращения. Его подробный audit хранится по отдельной retention policy; user-facing объяснение остаётся коротким.

Memory interface остаётся provider-neutral; choice of embedding/reranker/summarizer может отличаться от основного conversational provider. У каждой стадии собственный egress policy. Переход на другой provider не должен превращать недоступный private context в разрешённый или менять принятые facts.

Chat/Agent/Desktop могут читать одни и те же scoped сведения с разными требованиями свежести и capability boundaries. Desktop не получает отдельную «истину» о пользователе; его environment observations просто требуют более строгой привязки к реальному состоянию.

### 6.11. Пример действия в реальной системе

Память сообщает: «в прошлом запуске процесс сервиса работал с PID 1234». Это historical environment observation.

Для сегодняшней остановки сервиса нужны:
- текущая identity нужного сервиса/процесса, в том числе защита от переиспользованного PID;
- актуальный scope/approval для действия;
- наблюдение текущего состояния;
- корреляция исполнения и проверки результата.

~~~text
remembered state -> freshness check -> live observation
                                            |
                                 authorized action attempt
                                            |
                                  verification observation
                                            |
                          accepted outcome / failure / unknown
~~~

Если команда вернула exit code 0, но результат не проверен, память записывает successful command return, а не «сервис остановлен». Отмена/rollback также образуют события. Summary следующей сессии не должна сгладить различие между attempted и verified.

### 6.12. Долговременное обслуживание

Maintenance разделяется по полномочиям:

- **Воспроизводимые производные:** rebuild indexes, обновление summaries, cache eviction, проверка broken refs и coverage.
- **Изменение epistemic state:** предложить stale/conflict, потребовать новый evidence, принять correction по разрешённой политике. Возраст/частота обращения не являются доказательством.
- **Retention:** archive, redact, forget, physical deletion и обработка backups/exports по установленным правилам.

Для forgetting используется dependency closure: evidence → claims → synthesis → indexes/caches → readable exports. Если evidence удалено, но отдельный accepted claim разрешено сохранить, он должен честно отражать недоступность первичного источника. Нельзя оставлять sensitive synthesis только потому, что она считается derived.

Полезные показатели: доля claims с evidence, unresolved conflicts, freshness coverage environment facts, доля stale indexes, recall старых редких сведений, all-evidence coverage, correction propagation latency, erasure coverage, стоимость/ошибки context selection. Фоновые операции bounded, повторяемы, имеют progress cursor и диагностику; их выключение не должно уничтожать доступ к authoritative record.

## 7. Questions we still need to decide

Это решения уровня продукта и архитектуры. Таблица не задаёт порядок реализации.

| Вопрос | Почему важен | Основные варианты | Обоснованное направление / что пока неизвестно |
|---|---|---|---|
| **Что можно сохранять автоматически?** | Полная manual-only модель теряет continuity; unrestricted capture создаёт скрытый профиль и чувствительную историю | Только confirm каждой записи; scoped opt-in по типу/источнику; evidence автоматически, canonical claims после review; смешанная политика | Разделить recording, retention, canonical acceptance и sharing. Конкретные defaults требуют решения владельца; действующий confirm contract пока остаётся |
| **Что означает canonical?** | Нужно исключить превращение популярных summaries в истину | Свободные documents; атомарные typed claims; сочетание authored docs и extracted claims | Сочетание claims и документов с единой provenance/revision моделью; гранулярность атомизации пока открыта |
| **Какие типы first-class?** | От них зависят freshness, retention и применение | Минимальный универсальный claim; богатая ontology; небольшой базовый набор + расширения | Раздельные observation/fact/preference/decision/constraint/environment/task/inference; не фиксировать полный enum до обсуждения сценариев |
| **Нужна ли полная bitemporal история?** | Поздние corrections и исторические решения требуют двух времён | Valid-only; valid+recorded на всех claims; bitemporal только для значимых категорий | Различение двух времён обосновано; глубина истории и хранения transient events открыта |
| **Как задаются неизвестные даты и будущие изменения?** | Нельзя приписывать фактам epoch или считать планы свершившимися | Null с отдельной семантикой; interval uncertainty; planned/effective states | Явно различать unknown/open/scheduled; точный temporal vocabulary требует решения |
| **Кто разрешает конфликт?** | Последний текст может быть неверен, а разные scopes могут сосуществовать | Только человек; deterministic authority policy; LLM adjudication; комбинация | LLM предлагает пары/объяснения; scope/time/source правила и explicit review решают спорные случаи. Universal source ranking не обоснован |
| **Как вычислять confidence?** | Нельзя путать уверенность extractor с правдоподобием факта | Одно число; несколько calibrated scores; qualitative evidence states | Несколько независимых измерений; confidence не повышается от retrieval count. Числовая калибровка требует собственного корпуса |
| **Что означает pin/protection?** | Иначе pin обещает одно, а GC делает другое | Один universal pin; separate retention/exactness/retrieval/context flags | Разделить виды защиты и показывать их смысл; приоритеты переполнения обязательного context требуют продуктового решения |
| **Как долго хранить evidence?** | Многолетняя история конфликтует с privacy и объёмом | Вечно; TTL по всему; retention по source/type/sensitivity/dependencies | Scoped retention плюс явное забывание. Конкретные сроки, backup/export policy и redaction defaults пока не определены |
| **Может ли принятый claim пережить удаление evidence?** | Полное erasure и полезный личный профиль имеют разные ожидания | Всегда удалять descendants; сохранять разрешённое с утраченным provenance; запрашивать выбор | Нужна явная семантика команды «забудь». Нельзя автоматически сохранять sensitive пересказ удаляемого материала |
| **Какие области памяти разделяем?** | Personal/project/shared knowledge имеют разные audience | Только личное; personal+projects; отдельный shared group scope | Principal-scoped default и explicit sharing. Административная роль owner сама по себе не должна молча означать просмотр памяти members; доступ требует отдельного контракта |
| **Wiki — view или editable source?** | Иначе ручные correction теряются или расходятся с БД | Read-only projection; annotations; authored documents; двустороннее редактирование | Projection + отдельные authored annotations — наименьшая двусмысленность; потребность в полной двусторонней правке ещё не установлена |
| **Насколько широк knowledge graph?** | Граф может помочь identity или стать дорогой неточной копией всего | Entities/aliases; typed claim relationships; graph всех эпизодов | Устойчивые identities и необходимые связи — обоснованы. Полный graph-first retrieval требует задач, где он даёт выигрыш |
| **Как выбирается retrieval/reranker?** | Нужны русский язык, редкая старая память, ограниченная latency | Lexical; hybrid; rank fusion; local cross-encoder; LLM judge | Lexical+semantic как кандидат, затем evidence-based comparison. Конкретные модели/weights не выбраны |
| **Когда делать live refresh?** | Desktop action на старой памяти может быть ошибочным | TTL по типу; source-version invalidation; refresh перед рискованным действием; сочетание | Environment checks связывать с риском и identity/version; не перепроверять immutable исторические документы без причины |
| **Кто формирует runtime context?** | Нужен баланс полноты, бюджета и provider capabilities | Fixed primer; LLM-only selection; deterministic constraints + ranked selection + tool-driven expansion | Последний вариант обоснован: hard constraints локальны, семантика может быть модельной. Token reserves и latency targets пока открыты |
| **Какие providers могут видеть память?** | Optional LLM stage тоже раскрывает personal data | Local-only; per-provider policy; redacted external processing; user-approved exceptions | Per-stage egress policy, одинаковая для embeddings/reranking/synthesis/generation. Список разрешённых destinations — решение пользователя |
| **Как обрабатываются concurrent sessions?** | Две задачи могут одновременно исправить одно знание или иметь разные permissions | Последняя запись побеждает; serial acceptance; revision conflicts; optimistic concurrency | Сохранять revision identity и обнаруживать конфликт; task/session lineage отдельно от время-близости. Детали shared editing открыты |
| **Что считать качеством памяти?** | Высокий session retrieval hit может соседствовать с неверными ответами и действиями | Только recall; E2E QA; task success; комплексная оценка | Нужны все-evidence, provenance, temporal/correction/privacy и verified action outcomes. Пороговые значения определяются собственными сценариями |
| **Что разрешено maintenance без человека?** | Автоматическая чистка может менять substantive knowledge | Только indexes; ещё stale candidates; canonical merge/archive по policy | Rebuild derived — отдельная область от изменения accepted knowledge. Полномочия на merge/forget требуют явного решения |

Наиболее важный узел обсуждения: **какие записи требуют отдельного подтверждения, что считается допустимым автоматическим evidence capture и как correction/forget распространяются по всем производным**. Без этого невозможно честно определить semantics canonical memory, wiki и background maintenance.

## 8. Key takeaways

1. **Сохранять evidence отдельно от его representations.** Сжатый context должен раскрываться до разрешённого оригинала с provenance.
2. **Отделить авторитет источника от статуса хранения.** SQLite может быть primary record, но это не делает каждое содержимое фактом о мире.
3. **Temporal memory требует coherent revisions.** valid_from/to полезны только вместе с едиными correction/supersession правилами и различением «было верно» / «тогда было известно».
4. **Hybrid retrieval заслуживает обсуждения как принцип.** Точные identifiers и семантические парафразы требуют дополняющих каналов; веса reference не переносимы автоматически.
5. **Session continuity — самостоятельный слой.** Цели, блокеры, ограничения и verified progress не следует растворять в долговременном профиле.
6. **Compression должна управлять представлением, а не доверием.** Возраст и token pressure не должны уничтожать отрицания, provenance и обязательные ограничения.
7. **Wiki полезна для проверки памяти человеком.** Её ценность не требует превращать Markdown в единственный source of truth; авторские правки нуждаются в отдельном контракте.
8. **Automatic capture нужно обсуждать по стадиям.** Сбор evidence, proposal claims, accepted knowledge и sharing имеют разные риски и permissions.
9. **Maintenance — часть архитектуры с первого определения модели.** Forgetting, stale detection, corrections, cache invalidation и evidence dependencies должны быть согласованы.
10. **Personal agent требует большего, чем хороший recall.** Главные дополнения: principals/scopes, эпистемические типы, точный outcome действий, multilingual context, provider egress и возможность объяснить/исправить/удалить память.

### Источники и навигация по доказательствам

Все reference-ссылки ведут на конкретный коммит context-mem. Ссылки на классы и handlers — основание для описания реализации; предлагаемые принципы и модель SlavikAI являются авторским архитектурным анализом.

- Устройство и ingest: [Kernel][kernel], [Pipeline][pipeline], [storage schema][schema], [SQLite storage][storage], [types/defaults][types], [MCP core][mcp-core], [hooks][hooks], [capture hook][capture-hook], [conversation import][import].
- Знание, время и происхождение: [KnowledgeBase][knowledge-base], [knowledge handlers][knowledge-tools], [decision trail][trail], [time travel][time-travel], [global store][global], [graph][graph].
- Поиск: [unified handlers][search-tools], [BM25][bm25], [vector][vector], [embedder][embedder], [fusion/cache][fusion], [LLM judge][judge], [content store][content-store].
- Context/lifecycle: [sessions][sessions], [wake-up][wake-up], [budget][budget], [adaptive compression][compressor], [Dreamer][dreamer], [LifecycleManager][lifecycle], [importance][importance], [feedback][feedback], [pressure predictor][pressure].
- Wiki/privacy: [wiki schema][wiki-schema], [Context Protocol RFC][protocol], [VaultSync][vault], [templates][templates], [SynthesisEngine][synthesis], [entities][entities], [topics][topics], [PrivacyEngine][privacy], [LLM provider selection][llm-factory].
- Проверки reference: [temporal tests][temporal-tests], [lifecycle tests][lifecycle-tests], [fusion tests][fusion-tests], [vault tests][vault-tests], [synthesis tests][synthesis-tests]. Прочитаны; полный test suite не запускался.
- Метрики: [benchmark methodology][bench-method], [BenchKernel][bench-adapter], [RealKernelBench][bench-real], [LoCoMo harness][bench-locomo]. Опубликованные проценты не использованы как собственные результаты.

Локальные архитектурные источники: [ARCH_CANON.md](ARCH_CANON.md), [Architecture.md](Architecture.md), [SOURCE_OF_TRUTH.md](../SOURCE_OF_TRUTH.md), [runtime_contract_claims.json](../runtime_contract_claims.json), [DevRules.md](../agent/DevRules.md). При расхождении с этой кандидатной моделью действуют канонические документы, пока отдельное архитектурное решение не принято.

[kernel]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/kernel.ts
[pipeline]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/pipeline.ts
[schema]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/storage/migrations.ts
[storage]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/storage/better-sqlite3.ts
[types]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/types.ts
[mcp-core]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/mcp-server/tools/core.ts
[hooks]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/hooks/hooks.json
[capture-hook]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/hooks/context-mem-hook.js
[import]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/conversation-import.ts
[knowledge-base]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/knowledge/knowledge-base.ts
[knowledge-tools]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/mcp-server/tools/knowledge.ts
[trail]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/decision-trail.ts
[time-travel]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/time-travel.ts
[global]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/global-store.ts
[graph]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/knowledge-graph.ts
[search-tools]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/mcp-server/tools/search.ts
[bm25]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/search/bm25.ts
[vector]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/search/vector.ts
[embedder]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/search/embedder.ts
[fusion]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/search/fusion.ts
[judge]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/search/llm-judge.ts
[content-store]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/storage/content-store.ts
[sessions]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/session.ts
[wake-up]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/wake-up.ts
[budget]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/budget.ts
[compressor]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/adaptive-compressor.ts
[dreamer]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/dreamer.ts
[lifecycle]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/lifecycle.ts
[importance]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/importance-classifier.ts
[feedback]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/feedback-engine.ts
[pressure]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/pressure-predictor.ts
[wiki-schema]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/docs/llm-wiki-schema.md
[protocol]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/docs/context-protocol-v1.md
[vault]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/vault.ts
[templates]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/vault-templates.ts
[synthesis]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/synthesis.ts
[entities]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/entity-extractor.ts
[topics]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/topic-detector.ts
[privacy]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/privacy/privacy-engine.ts
[llm-factory]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/llm-factory.ts
[temporal-tests]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/tests/core/temporal-facts.test.ts
[lifecycle-tests]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/tests/core/lifecycle.test.ts
[fusion-tests]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/tests/plugins/search/fusion.test.ts
[vault-tests]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/tests/core/vault.test.ts
[synthesis-tests]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/tests/core/synthesis.test.ts
[bench-method]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/docs/benchmarks/methodology.md
[bench-adapter]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/benchmarks/lib/kernel-adapter.js
[bench-real]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/benchmarks/lib/real-kernel-bench.js
[bench-locomo]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/benchmarks/locomo.js
[summarizers]: https://github.com/JubaKitiashvili/context-mem/tree/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/summarizers
[json-summary]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/summarizers/json-summarizer.ts
[error-summary]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/plugins/summarizers/error-summarizer.ts
[prompt-hook]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/hooks/user-prompt-hook.js
[session-hook]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/hooks/session-start-hook.js
[narrative]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/src/core/narrative-generator.ts

[package]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/package.json
[readme]: https://github.com/JubaKitiashvili/context-mem/blob/2a55af0a4bf3467df89f1315a74bb2e15ad903f7/README.md
