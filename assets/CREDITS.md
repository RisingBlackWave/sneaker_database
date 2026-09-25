# Credits

## `foot.bin` — 3D foot used by the FOOT MAP

**"Foot Topology Study"** by **Johnson Martin**
Source: https://sketchfab.com/3d-models/foot-topology-study-4aa96c9b0a9b4cb1b3740f8e0558ffbc
Author: https://sketchfab.com/Johnson-Martin
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**Changes made:** used the higher-resolution `skin.002` mesh only (the low-poly
topology cage was dropped); merged its four colour-group submeshes into one mesh;
discarded materials and UVs; rotated into the app's frame (Y up, sole on y=0,
toes toward +Z), scaled to a 26-unit foot length, centred; quantised to the
compact `SNKF` binary format. Produced with `tools/prep_foot.py`:

    python3 tools/prep_foot.py foot_topology_study.glb --node skin.002 --out assets/foot.bin
