import gradio as gr
import numpy as np

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


def ToSepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


# slider01 = gr.Slider(value=2, minimum=1, maximum=10, step=1)
# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", slider01],
#     outputs=["text"],
# )

demo = gr.Interface(ToSepia, gr.Image(), "image")

demo.launch()
