# Resources/

Drop the model bundle the app needs at launch:

| File             | Where it comes from                          | Size    |
|------------------|----------------------------------------------|---------|
| `model.litertlm` | `gemma-3/model/model.litertlm` (exported by  | ~285 MB |
|                  | `gemma-3/export/export_gemma3_270m.py`)      |         |

`scripts/bootstrap.sh` copies this from `../../model/model.litertlm`
automatically if present.

The file is intentionally gitignored — it's bundled into the `.app` at
build time as a flat resource (not a folder reference; see the comment in
`project.yml`).
