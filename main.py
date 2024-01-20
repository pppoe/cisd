import argparse
from PIL import Image
from diffusers import OnnxStableDiffusionPipeline

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prompt', type=str, required=True)
    args.add_argument('--output', type=str, required=True)
    parsed_args = args.parse_args()
    onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(
        "/sdv0/",
        safety_checker=None,
        revision="onnx",
        provider="CPUExecutionProvider",
    )
    images = onnx_pipe(parsed_args.prompt)
    images[0][0].save(parsed_args.output)
