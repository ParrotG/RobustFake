import kagglehub

path = kagglehub.dataset_download(
    "birdy654/cifake-real-and-ai-generated-synthetic-images",
    output_dir=r"D:\NTU_project\hackson\model_based_on_clip\dataset"
)

print("Path to dataset files:", path)