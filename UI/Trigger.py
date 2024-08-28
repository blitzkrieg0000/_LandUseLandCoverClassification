"""
    import gradio as gr

    with gr.Blocks() as demo:
        name = gr.Textbox(label="Name")
        output = gr.Textbox(label="Output Box")
        greet_btn = gr.Button("Greet")
        trigger = gr.Textbox(label="Trigger Box")

        def greet(name, evt_data: gr.EventData):
            return "Hello " + name + "!", evt_data.target.__class__.__name__

        def clear_name(evt_data: gr.EventData):
            return ""

        gr.on(
            triggers=[name.submit, greet_btn.click],
            fn=greet,
            inputs=name,
            outputs=[output, trigger],
        ).then(clear_name, outputs=[name])

    demo.launch()
"""


import gradio as gr

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")

    @gr.on(triggers=[name.submit, greet_btn.click], inputs=name, outputs=output)
    def greet(name, evt_data: gr.EventData):
        return "Hello " + name + "!"

demo.launch()


#%%
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        num1 = gr.Slider(1, 10)
        num2 = gr.Slider(1, 10)
        num3 = gr.Slider(1, 10)
    output = gr.Number(label="Sum")

    @gr.on(inputs=[num1, num2, num3], outputs=output)
    def sum(a, b, c):
        return a + b + c

demo.launch()