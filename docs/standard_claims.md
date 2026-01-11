# Standard claims (VALUE-M4): что гарантируем и чего НЕ гарантируем

Назначение: зафиксировать **проверяемые** обещания HETERO-2 как evidence pipeline (Pfizer-ready facts).
Это не маркетинг: каждый пункт “гарантируем” должен быть либо контрактом, либо тестом (или обоими).

См. также: [CONTEXT](../CONTEXT.md), [Release checklist](95_release_checklist.md), [Artefacts registry](artefacts_registry.md), [Contracts policy](10_contracts_policy.md).

---

## Канонические контракты (frozen)

- `hetero_scores.v1` — `docs/contracts/hetero_scores.v1.md`
- `hetero_audit.v2` — `docs/contracts/hetero_audit.v2.md`

---

## Что гарантируем (и чем это доказано)

### 1) Evidence pack: обязательные файлы и целостность

При сборке evidence pack с включёнными `manifest.json`, `checksums.sha256` и `evidence_pack.zip`:

- В pack присутствуют обязательные файлы: `summary.csv`, `metrics.json`, `index.md`, `manifest.json`, `checksums.sha256`.
- `checksums.sha256` содержит SHA256 для файлов evidence pack (включая `manifest.json`).
- `evidence_pack.zip` содержит эти файлы (без “тихих пропусков” по именам).

Proof:
- checklist: `docs/95_release_checklist.md`
- tests: `tests/test_standard_claims_guardrails.py`, `tests/test_hetero2_evidence_pack_checksums_contract.py`

### 2) Frozen schema_version для входных/выходных JSON

- External scores принимаются только при `schema_version == hetero_scores.v1`.
- Audit внутри пайплайна фиксирован как `schema_version == hetero_audit.v2` (в т.ч. семантика `gate/slack/verdict`).

Proof:
- contracts: `docs/contracts/hetero_scores.v1.md`, `docs/contracts/hetero_audit.v2.md`
- tests: `tests/test_hetero_scores_v1_freeze_contract.py`, `tests/test_hetero_audit_v2_separation.py`, `tests/test_standard_claims_guardrails.py`

### 3) Истина артефактов (release/registry)

Для VALUE-вех “истина артефакта” определяется только так:

1) GitHub Release asset (zip),
2) SHA256 на asset,
3) запись в `docs/artefacts_registry.md` (URL + SHA256 + команда + outcome),
4) зелёные CI-гейты `ci/test`, `ci/test-chem`, `ci/docker` на целевом SHA.

Proof:
- `docs/artefacts_registry.md`
- `docs/95_release_checklist.md`
- `docs/90_lineage.md`

---

## Что НЕ гарантируем

- “Научную истинность” или универсальность метрик (slack/gate/verdict) вне оговорённого контекста и входов.
- Что модель/скоринг “оптимальны”, “лучшие”, “находят лучшие молекулы” или имеют клиническую валидность.
- Что любой набор данных/SMILES будет обработан без SKIP/ERROR: мы гарантируем **явную маркировку**, а не отсутствие проблем.
- Что результаты подходят для любых downstream-решений без экспертной интерпретации.

---

## Политика изменений (чтобы не было “тихой смены смысла”)

- Любая смена семантики публичных полей = breaking change → новая версия `schema_version`.
- Замороженный контракт меняется только через: контракт-док + backward-compat тест(ы) + указатели в `CONTEXT.md` и `docs/99_index.md`.

См. `docs/10_contracts_policy.md`.

