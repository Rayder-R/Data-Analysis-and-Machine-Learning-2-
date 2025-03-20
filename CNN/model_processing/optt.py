import openvino as ov
ov_model = ov.convert_model('Mnist_saved_model')
ov.save_model(ov_model, "openvino_model.xml")