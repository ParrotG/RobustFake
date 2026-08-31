from transformers import CLIPModel, CLIPProcessor

model_path = r"D:\NTU_project\hackson\model_based_on_clip\clip-vit-base-patch32"

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.save_pretrained(model_path)
processor.save_pretrained(model_path)