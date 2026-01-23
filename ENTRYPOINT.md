# ENTRYPOINT

Единая точка входа по контексту: `CONTEXT.md`.

- Для ИИ/ассистента: начните с `CONTEXT.md` и следуйте **Порядку загрузки**.
- Для человека: `README.md` -> `CONTEXT.md` -> `docs/README.md`.

## Read order (you must follow)

1. `ENTRYPOINT.md` — как читать репо без блуждания.
2. `CONTEXT.md` — операционный entrypoint: STOP/GO, “как проверять истину”, как запускать.
3. `docs/ROADMAP.md` — управляющий документ: VALUE-first, вехи, DoD, Proof.
4. `docs/contracts/PHYSICS_OPERATOR_CONTRACT.md` — P0 контракт `H=L+V`, SoT `atoms_db_v1.json`, анти-иллюзия AUC.
5. `docs/30_config_layers.md` — слои A/B/C + канонический `experiment.yaml` (единая точка входа).
6. `docs/specs/accuracy_a3_phase_channel.md` — ACCURACY-A3: фазовый/магнитный канал (SPEC + invariant tests-only).
7. `src/hetero2/*` — реализация пайплайна (batch/pipeline/decoys/audit/operator).
8. `docs/artefacts_registry.md` — истина артефактов (asset URL + SHA256 + команда + outcome).
9. `docs/90_lineage.md` — append-only хронология мерджей/релизов и причинно-следственная линия.
