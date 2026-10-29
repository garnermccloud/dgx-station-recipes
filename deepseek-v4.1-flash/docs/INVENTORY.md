# Release-tree inventory

Generated from the staging tree. `docs/inventory.tsv` has one row for every file: `path`, `owner`, `role`, `note`.

**Total: 776 files.** Roles: build 131, docs 43, runtime 602.

## By top-level path

| path | files | owners |
|---|---:|---|
| `docker/` | 555 | rv-l2persist, rv-serve, upstream-vendor |
| `fastpath/` | 91 | rv-serve |
| `kernels/` | 80 | rv-serve |
| `verify/` | 16 | rv-serve |
| `./` | 8 | coordinator, rv-dspark, rv-megamoe, rv-serve |
| `tools/` | 8 | rv-dspark, rv-serve |
| `docs/` | 7 | rv-dspark, rv-megamoe |
| `scripts/` | 5 | rv-serve |
| `serve/` | 3 | rv-serve |
| `config/` | 2 | rv-serve |
| `hf/` | 1 | rv-megamoe |

## Dead files (owner action requested; not deleted here)

| file | owner | why |
|---|---|---|
| _none_ | — | — |

`NOTICE.md` was a byte-identical duplicate of `NOTICE`; it was deleted in this sweep (rv-dspark).

## Duplicate source trees/files (owner action requested; not deleted here)

| duplicate | files | canonical copy | owner |
|---|---:|---|---|
| `fastpath/peer3/csrc/` | 7 | `kernels/peer3/csrc/` | rv-serve |

The TSV notes every individual file in these groups. The duplicate trees are not all byte-identical: some have build scripts or tests only on one side, so the owner must choose the canonical tree, not blindly delete.

## Pending working-tree changes (not in HEAD)

These are owner edits in progress; they are not part of the committed inventory.

| path | status |
|---|---|
| _none_ | — |

## Build-time versus runtime

- **runtime**: mounted/shipped and read during `ds41 up` or serving (`ds41`, `serve/`, `config/`, `fastpath/` payload, prebuilt `.so`, `docker/b12x/`, `verify/`).
- **build**: Dockerfiles, overlays, kernel sources, setup/tune/profiling tools, test fixtures.
- **docs**: README, RESULTS, LIMITATIONS, NOTICE/LICENSE, architecture/build docs.
- **dead**: unused archives, committed bytecode, stale duplicate config.

## Known generated artifacts

- Prebuilt `.so` files under `fastpath/` and `fastpath/peer3/prebuilt/` are deliberate runtime artifacts (no compiler on the serving node), not dead outputs.
- The shipped tune file is `fiset-w6pf-post1-13cbba83.json`; `config/recipe.env` still names the old `a5ee9882` file and must be fixed before boot.
