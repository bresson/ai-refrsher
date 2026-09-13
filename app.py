import os
import pandas as pd
import gradio as gr
from agent import BasicAgent
from evaluation import fetch_questions, run_agent_on_questions, submit_answers


def run_and_submit_all(profile: gr.OAuthProfile | None):
    if not profile:
        return "Please Login to Hugging Face with the button.", None

    space_id = os.getenv("SPACE_ID")
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "LOCAL_RUN"

    agent = BasicAgent()
    questions_data = fetch_questions()
    results_log, answers_payload = run_agent_on_questions(agent, questions_data)
    result = submit_answers(profile.username, agent_code, answers_payload)

    status = f"Score: {result.get('score')}% ({result.get('correct_count')}/{result.get('total_attempted')})"
    return status, pd.DataFrame(results_log)


with gr.Blocks() as demo:
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Status", lines=5)
    results_table = gr.DataFrame(label="Results", wrap=True)
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

demo.launch(debug=True, share=False, ssr_mode=False)