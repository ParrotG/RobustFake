from transformers import AutoImageProcessor, AutoModel

model_path=r"D:\NTU_project\hackson\DINO-v2\DINO_v2_based"

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

processor.save_pretrained(model_path)
model.save_pretrained(model_path)