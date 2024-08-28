import gradio as gr

input_textbox = gr.Textbox()


with gr.Blocks() as demo2:
    gr.Examples(["hello", "bonjour", "merhaba"], input_textbox)
    
    name_box = gr.Textbox(label="Name")
    input_textbox.render()
    age_box = gr.Number(label="Age", minimum=0, maximum=100)
    symptoms_box = gr.CheckboxGroup(["Cough", "Fever", "Runny Nose"])
    submit_btn = gr.Button("Submit")



    with gr.Column(visible=False) as output_col:
        diagnosis_box = gr.Textbox(label="Diagnosis")
        patient_summary_box = gr.Textbox(label="Patient Summary")

    
    def submit(name, age, symptoms):
        return {
            submit_btn: gr.Button(visible=False),
            output_col: gr.Column(visible=True),
            diagnosis_box: "covid" if "Cough" in symptoms else "flu",
            patient_summary_box: f"{name}, {age} y/o",
        }


    submit_btn.click(
        submit,
        [name_box, age_box, symptoms_box],
        [submit_btn, diagnosis_box, patient_summary_box, output_col],
    )


demo2.launch()