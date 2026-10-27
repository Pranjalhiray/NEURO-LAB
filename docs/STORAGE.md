# Storage inventory and cleanup

The initial workspace audit found an empty D:\BTS project directory, about 109 GB free on D:, and about 21 GB free on C:. These figures are a snapshot from 2026-10-09; run the inventory script for current values.

The initial audit preceded data and model setup. The current inventory (2026-10-10) is:

| Item | Size / count |
|---|---:|
| Python environment with CUDA PyTorch | 3.1 GB, 26,346 files |
| Frontend dependencies | 93.2 MB, 5,993 files |
| BRISC v3 archive plus extracted dataset | 496.1 MB, 15,587 files |
| Model packages | 6.4 MB, 8 files |
| Data cache | 34.9 MB, 207 files |
| Temporary scratch | empty |
| Free space on C: / D: | 19.9 GB / 105.4 GB |

The complete local machine has 16 GB RAM and a 4 GB RTX 3050 Laptop GPU. Batch size 4 at 256 pixels fits the current segmentation training setup. Full 3D BraTS training is not a local target. The resource inventory script can refresh these figures.

Inspect project storage and available disk space:

~~~powershell
python scripts\resource_inventory.py
~~~

The cleanup tool defaults to a dry run and only lists files inside the project's temporary scratch folder. Cache removal is a separate opt-in. It never targets datasets, trained models, or source code.

~~~powershell
python scripts\cleanup_temp.py
python scripts\cleanup_temp.py --apply --confirm "DELETE TEMP DATA"
python scripts\cleanup_temp.py --include-cache
~~~

Review the listed paths before using the confirmed cleanup command. A dry run found no scratch files; no cleanup command has deleted files.
