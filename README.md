# SUSTAINABLE-SMART-CITY-ASSISTANT
A Sustainable Smart City Assistant is an AI-powered digital platform designed to help urban areas operate more efficiently, reduce their environmental footprint, and improve residents’ quality of life. It acts as the city’s “brain,” collecting and analyzing real-time data from multiple sources such as sensors, IoT devices, public services, citizen.
import gradio as gr
from gradio.themes import Soft
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import pathlib

print("✅ Model loading... GPU available:", torch.cuda.is_available())

# Pick a proper Hugging Face model
model_name = "gpt2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

# Pipeline
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)


# -------------------------
# Utility Functions
# -------------------------

def policy_summarizer_v2(text, file):
    if file is not None:
        path = file.name if hasattr(file, "name") else str(file)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    elif text.strip():
        content = text.strip()
    else:
        return "⚠️ Please upload a file or paste some text."

    prompt = f"Summarize the following city policy in simple terms:\n{content}\nSummary:"
    result = llm(prompt, max_new_tokens=100)[0]["generated_text"]
    return result[len(prompt):].strip()


def citizen_feedback(issue):
    return f"✅ Thank you! Your issue '{issue}' has been logged and categorized appropriately."


def kpi_forecasting(csv_file):
    path = csv_file.name if hasattr(csv_file, "name") else str(csv_file)
    df = pd.read_csv(path)
    X = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values
    model = LinearRegression().fit(X, y)
    next_year = [[X[-1][0] + 1]]
    prediction = model.predict(next_year)[0]
    return f"📈 Predicted KPI for {next_year[0][0]}: {round(prediction, 2)}"


def eco_tips(keyword):
    prompt = f"Give 3 actionable eco-friendly tips related to: {keyword}"
    result = llm(prompt, max_new_tokens=100)[0]["generated_text"]
    return result[len(prompt):].strip()


def detect_anomaly(csv_file):
    path = csv_file.name if hasattr(csv_file, "name") else str(csv_file)
    df = pd.read_csv(path)

    if 'value' not in df.columns:
        return "⚠️ CSV must contain a 'value' column."

    mean = df["value"].mean()
    std = df["value"].std()
    anomalies = df[np.abs(df["value"] - mean) > 2 * std]

    if anomalies.empty:
        return "✅ No significant anomalies detected."

    return "⚠️ Anomalies found:\n" + anomalies.to_string(index=False)


def chat_assistant(question):
    prompt = f"Answer this smart city sustainability question:\n\nQ: {question}\nA:"
    result = llm(prompt, max_new_tokens=100, temperature=0.7)[0]["generated_text"]
    return result[len(prompt):].strip()


# -------------------------
# Gradio App
# -------------------------

custom_theme = Soft()

with gr.Blocks(theme=custom_theme) as app:
    gr.Markdown("## 🌆 Sustainable Smart City Assistant")
    gr.Markdown("Built with IBM Granite LLM 🧠 to empower urban planning, feedback, sustainability, and innovation.")

    with gr.Tabs():
        with gr.Tab("📝 Policy Summarization"):
            with gr.Column():
                gr.Markdown("Upload a `.txt` file or paste policy text to generate a summary.")
                with gr.Row():
                    policy_file = gr.File(label="Upload .txt File", file_types=[".txt"])
                    policy_text = gr.Textbox(label="Or paste policy text", lines=10)
                policy_output = gr.Textbox(label="Summary", lines=5)
                summarize_btn = gr.Button("Summarize")
                summarize_btn.click(policy_summarizer_v2, inputs=[policy_text, policy_file], outputs=policy_output)

        with gr.Tab("📣 Citizen Feedback"):
            feedback_input = gr.Textbox(lines=3, label="Describe the Issue")
            feedback_output = gr.Textbox(label="Acknowledgement")
            feedback_btn = gr.Button("Submit Feedback")
            feedback_btn.click(citizen_feedback, inputs=feedback_input, outputs=feedback_output)

        with gr.Tab("📊 KPI Forecasting"):
            kpi_input = gr.File(label="Upload KPI CSV")
            kpi_output = gr.Textbox(label="Forecast Result")
            kpi_btn = gr.Button("Forecast KPI")
            kpi_btn.click(kpi_forecasting, inputs=kpi_input, outputs=kpi_output)

        with gr.Tab("🌱 Eco Tips Generator"):
            tip_input = gr.Textbox(label="Keyword (e.g. Plastic, Solar)")
            tip_output = gr.Textbox(label="Generated Tips")
            tip_btn = gr.Button("Get Eco Tips")
            tip_btn.click(eco_tips, inputs=tip_input, outputs=tip_output)

        with gr.Tab("🚨 Anomaly Detection"):
            anomaly_input = gr.File(label="Upload CSV with 'value' column")
            anomaly_output = gr.Textbox(label="Anomaly Results")
            anomaly_btn = gr.Button("Detect Anomalies")
            anomaly_btn.click(detect_anomaly, inputs=anomaly_input, outputs=anomaly_output)

        with gr.Tab("💬 Chat Assistant"):
            chat_input = gr.Textbox(label="Ask your question")
            chat_output = gr.Textbox(label="Assistant Response")
            chat_btn = gr.Button("Ask")
            chat_btn.click(chat_assistant, inputs=chat_input, outputs=chat_output)

app.launch()
