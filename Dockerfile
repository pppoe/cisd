FROM python:3.11-slim
RUN python3 -m pip install onnxruntime pillow diffusers transformers
RUN python3 -m pip install accelerate
COPY small-stable-diffusion-v0_onnx/ /sdv0
