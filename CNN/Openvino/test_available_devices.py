import openvino.properties as props
import openvino as ov

# 用來測試電腦是否支援 NPU 

core = ov.Core()
device = "NPU"

test = core.available_devices
print(test)

test = core.get_property(device, props.device.full_name)
print(test)
