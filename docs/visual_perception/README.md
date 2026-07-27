# Visual perception

`visual_perception` groups traditional vision inference that extracts
structured information from images. It is separate from `visual_generation`.

- [Instance perception and YOLO conversion](instance_perception.md)
- [Image classification and YOLOv8-CLS](image_classification.md)
- [Semantic segmentation conversion contract](semantic_segmentation.md)
- [Depth estimation conversion contract](depth_estimation.md)

Instance perception returns object instances with optional boxes, masks,
keypoints, or orientation. Semantic segmentation and depth estimation retain
their dense per-pixel result contracts. Image classification uses its own
whole-image Top-K contract under the same top-level category.
