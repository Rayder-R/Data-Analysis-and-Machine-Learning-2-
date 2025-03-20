from pathlib import Path
import huggingface_hub as hf_hub
# pip install huggingface-hub
import openvino as ov
import time
import openvino.properties as props

# 設定模型目錄
MODEL_DIRECTORY_PATH = Path("model")
model_name = "resnet50"
precision = "FP16"
device = "NPU"
model_path = MODEL_DIRECTORY_PATH / "ir_model" / f"{model_name}_{precision.lower()}.xml"

# 物件偵測模型
core = ov.Core()

# Create cache folder
cache_folder = Path("cache")
cache_folder.mkdir(exist_ok=True)

start = time.time()
core = ov.Core()

# Set cache folder
core.set_property({props.cache_dir(): cache_folder})

# Compile the model
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model, device)
print(f"Cache enabled (first time) - compile time: {time.time() - start}s")

start = time.time()
core = ov.Core()

# Set cache folder
core.set_property({props.cache_dir(): cache_folder})

# Compile the model as before
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model, device)
print(f"Cache enabled (second time) - compile time: {time.time() - start}s")