from pathlib import Path
import huggingface_hub as hf_hub
# pip install huggingface-hub
import openvino as ov

core = ov.Core()

# create a directory for resnet model file
MODEL_DIRECTORY_PATH = Path("model")
MODEL_DIRECTORY_PATH.mkdir(exist_ok=True)

model_name = "resnet50"


precision = "FP16"

model_path = MODEL_DIRECTORY_PATH / "ir_model" / f"{model_name}_{precision.lower()}.xml"

model = None
if not model_path.exists():
    hf_hub.snapshot_download("katuni4ka/resnet50_fp16", local_dir=model_path.parent)
    print("IR model saved to {}".format(model_path))
    model = core.read_model(model_path)
else:
    print("Read IR model from {}".format(model_path))
    model = core.read_model(model_path)