# UFL-YOLOv10 Plugin (UAAM + SOEP + NAFL)

This package contains plugin modules to integrate into a YOLOv10 codebase:
- UAAM: Underwater-Aware Attention Module (models/your_custom/uaam.py)
- SOEP: Small Object Enhance Pyramid (models/your_custom/soep.py)
- NAFL: Noise-Aware Focal Loss + BackgroundBranch (models/losses/nafl.py)

Integration notes:
- Copy `models/your_custom/` and `models/losses/` to your YOLOv10 repo.
- Register/import UAAM and SOEP in the model builder; call SOEP with shallow P2 and mid P3 features.
- Add BackgroundBranch inside the decoupled head to predict bg_map and pass it to NoiseAwareFocalLoss during training.

Reference manuscript (source of design): include alongside code for reproducibility. fileciteturn1file0
