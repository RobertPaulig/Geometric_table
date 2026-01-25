# Индекс документов и REF-ов (`docs/99_index.md`)

Назначение: **единый обязательный индекс** “что читать/цитировать/использовать” в репо.
Каждый REF указывает на **реальный файл** (source of truth) или явно помечен как “производный артефакт, если присутствует”.

## Core (тома)

- **REF-VOL1-SPEC** — `docs/name3.md`  
  Зачем: Том I (Конституция/ТЗ) как читаемый source of truth в Markdown.  
  Производные артефакты (если присутствуют): `docs/name3.tex`, `docs/name3.pdf`.

- **REF-VOL2-DEVLINE** — `docs/name4.tex`  
  Зачем: Том II / devline (R&D) как source of truth в LaTeX.  
  Производный артефакт (если присутствует): `docs/name4.pdf`.

## Repo-управление (обязательное)

- **REF-ENTRYPOINT** — `CONTEXT.md`, `docs/README.md`  
  Зачем: единая точка входа и порядок загрузки контекста.  
  Где используется: всегда (онбординг / дисциплина документации / протокол).

- **REF-CONFIG-LAYERS** — `docs/30_config_layers.md`  
  Зачем: конституция слоёв A/B/C (PhysicsKernel / FeatureExtractor / Learner) и канонический `experiment.yaml`.  
  Где используется: ACCURACY-A* и любые эксперименты, где нужен единый воспроизводимый конфиг.

- **REF-ACCURACY-A3-PHASE-CHANNEL-SPEC** — `docs/specs/accuracy_a3_phase_channel.md`  
  Зачем: спецификация фазового/магнитного канала (A3.0) + инварианты/guardrails (tests-only).  
  Где используется: планирование ACCURACY-A3.* (сперва контракт и инварианты, затем реализация под флагом).

- **REF-ACCURACY-A3.3-CONTRACT** — `docs/specs/accuracy_a3_3_phase_rho_pivot.md`  
  Зачем: контракт исполнения ACCURACY-A3.3 (phase→ρ pivot): phase входит только через ρ, датчики вещественности/нормировки ρ, nested-выбор Φ (train-only).  
  Где используется: любые PR/запуски ACCURACY-A3.3 (обязателен к чтению до изменений кода).

- **REF-ACCURACY-A3.4-CONTRACT** — `docs/specs/accuracy_a3_4_rho_plus_current.md`  
  Зачем: контракт исполнения ACCURACY-A3.4 (ρ+J): Φ фиксирован (π/2), добавляется gauge-invariant current-конденсат J и единственный новый DOF κ (nested train-only).  
  Где используется: любые PR/запуски ACCURACY-A3.4 (обязателен к чтению до изменений кода).

- **REF-ACCURACY-A3.5-CONTRACT** — `docs/specs/accuracy_a3_5_edge_coherence_condensate.md`  
  Зачем: контракт исполнения ACCURACY-A3.5 (ρ + edge coherence): Φ фиксирован (π/2), добавляется edge-coherence конденсат |K_ij| и единственный новый DOF κ (nested train-only).  
  Где используется: любые PR/запуски ACCURACY-A3.5 (обязателен к чтению до изменений кода).

- **REF-BACKLOG** — `docs/04_backlog.md`  
  Зачем: склад гипотез/задач (сырьё, не “истина”).  
  Где используется: постановка R&D-экспериментов и планирование работ.

- **REF-ROADMAP** - `docs/ROADMAP.md`  
  Зачем: управляющий документ (VALUE-first) + обязательные вехи (milestones).  
  Где используется: приоритезация, привязка PR к Roadmap-ID, контроль готовности.

- **REF-VALUE-M3-ACCEPT** - `docs/value_m3_acceptance_criteria.md`  
  Зачем: acceptance criteria для VALUE-M3 (customer/proxy pack) как проверяемый контракт.  
  Где используется: выпуск и приёмка VALUE-M3, критерии GO/STOP для клиента.

- **REF-COST-LIFT** — `scripts/cost_lift.py`  
  Зачем: расчёт utility-артефакта cost&lift (Claim-2 money proof) по `summary.csv` + customer truth CSV.  
  Где используется: VALUE-M5-UTILITY-1 (публикация “денег”, а не только packs).

- **REF-DECISION-LOG** - `docs/05_decision_log.md`  
  Зачем: фиксируем принятые решения, инварианты, гейты, изменения метрик и интерпретаций.  
  Где используется: аудит изменений и восстановление причинно-следственной линии.

- **REF-LINEAGE** — `docs/90_lineage.md`  
  Зачем: append-only хронология релизов/мержей/управленческих изменений.  
  Где используется: аудит и восстановление “истины истории”.

- **REF-RELEASE-CHECKLIST** — `docs/95_release_checklist.md`  
  Зачем: checklist релиза evidence pack (состав pack, quality gates, registry).  
  Где используется: каждый publish workflow / релиз.

- **REF-ARTEFACTS-REGISTRY** — `docs/artefacts_registry.md`  
  Зачем: истина артефактов (asset_url + SHA256 + команда + outcome).  
  Где используется: любой факт VALUE/релиза.

- **REF-VERSION** — `VERSION`  
  Зачем: текущий baseline версии репо.  
  Где используется: provenance и релизные заметки.

- **REF-PILOT-QUICKSTART** — `docs/pilot_quickstart.md`  
  Зачем: быстрый сценарий “демо” и команды запуска.  
  Где используется: воспроизводимый пилот без магии.

- **REF-GATES** — `CONTEXT.md` (секция `## Gates (STOP/GO)`)  
  Зачем: единый протокол STOP/GO и “tested SHA” (merge-ref на PR).  
  Где используется: любой отчёт по гейтам и мердж-политика.

- **REF-COMMS-PROTOCOL** — `docs/20_comms_protocol.md`  
  Зачем: фиксированная форма обращений и отчётности Исполнителя.  
  Где используется: все рабочие коммуникации и отчёты.

## Contracts (обязательное)

- **REF-CONTRACTS-POLICY** — `docs/10_contracts_policy.md`  
  Зачем: правила версионирования контрактов (`schema_version`) и breaking change policy.  
  Где используется: все frozen contracts.

- **REF-STANDARD-CLAIMS** — `docs/standard_claims.md`  
  Зачем: канонический список “что гарантируем / что не гарантируем”, привязанный к frozen contracts и guardrail-тестам.  
  Где используется: VALUE-M4, аудит и внешняя коммуникация.

- **REF-CONTRACT-HETERO-SCORES-V1** — `docs/contracts/hetero_scores.v1.md`  
  Зачем: frozen контракт `hetero_scores.v1` для `score_mode=external_scores`.  
  Где используется: external scores + backward-compat тесты.

- **REF-CONTRACT-HETERO-AUDIT-V2** — `docs/contracts/hetero_audit.v2.md`  
  Зачем: семантика `audit.neg_controls.gate/slack/verdict` (VALUE-M2).  
  Где используется: оценка качества скоринга и separation facts.

- **REF-CONTRACT-CUSTOMER-TRUTH-V1** — `docs/contracts/customer_truth.v1.md`  
  Зачем: frozen CSV-контракт “expensive truth labels” для расчёта cost&lift.  
  Где используется: VALUE-M5-UTILITY-1 (customer/proxy truth → utility report).

- **REF-CONTRACT-COST-LIFT-V1** — `docs/contracts/cost_lift.v1.md`  
  Зачем: frozen контракт `cost_lift.v1` для `cost_lift_report.json` (utility report: cost&lift).  
  Где используется: VALUE-M5-UTILITY-1 (Claim-2 money proof; uplift + CI + coverage).

- **REF-CONTRACT-PHYSICS-OPERATOR-P0** — `docs/contracts/PHYSICS_OPERATOR_CONTRACT.md`  
  Зачем: контракт P0 для физического оператора `H=L+V`, SoT `atoms_db_v1.json`, и анти-иллюзия AUC через hardness curve.  
  Где используется: PHYSICS-P0 (operator foundation + audit-grade verdict по интерпретации AUC).

## Отчёты / релиз-заметки

- **REF-REPORTS** — `REPORT.md`, `REPORT_baseline.md`, `REPORT_wsZ1.md`  
  Зачем: отчёты/результаты.

- **REF-CHEM-RELEASE** — `docs/chem_validation_5_release_note.md`  
  Зачем: release note по chem-валидации/HETERO.
