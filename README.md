---
title: Face Mesh Workflow
emoji: 🐢
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 3.35.2
app_file: app.py
pinned: false
duplicated_from: verkaDerkaDerk/face-image-to-face-obj
---

Uses MediaPipe to detect a face in an image the allows you to combined it's depth estimation with those from Zoe and Midas.
The 3d viewer has Y pointing the opposite direction from Blender, so ya hafta spin it.

See https://huggingface.co/spaces/shariqfarooq/ZoeDepth and https://huggingface.co/spaces/drafff/dpt-depth-estimation

Caveat: I may be conflating "Intel/dpt-large" and "Midas"...

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
