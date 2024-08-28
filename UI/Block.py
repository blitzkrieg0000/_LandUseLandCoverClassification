import gradio as gr

with gr.Blocks() as demo:
    a = gr.Number(label="a")
    b = gr.Number(label="b")
    with gr.Row():
        add_btn = gr.Button("Add")
        sub_btn = gr.Button("Subtract")
    c = gr.Number(label="sum")

    @add_btn.click(inputs=[a, b], outputs=c)
    def add(num1, num2):
        return num1 + num2
    
    @sub_btn.click(inputs={a, b}, outputs=c)
    def sub(data):
        return data[a] - data[b]
    

demo.launch()
