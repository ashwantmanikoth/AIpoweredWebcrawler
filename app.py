import gradio as gr
import configparser
from components.chat_logic import server

def get_models_from_properties(file_path: str) -> list[str]:
    """Reads available models from a properties file."""
    config = configparser.ConfigParser()
    config.read(file_path)
    models_str = config.get("models_section", "models", fallback="")
    return [m.strip() for m in models_str.split(",") if m.strip()]

# Load models and retrieval modes
models = get_models_from_properties("models.properties")
modes = ["RAG Fusion", "Multi Query RAG"]

with gr.Blocks() as ui:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<br><br><br>")  # Moves sidebar down
            model_dropdown = gr.Dropdown(choices=models, label="Select a LLM model")
            temp_slider = gr.Slider(0.0, 1.0, step=0.05, value=0.0, label="Adjust the temperature")
            n_results_slider = gr.Slider(3, 20, step=1, value=5, label="Number of Search Results")
            search_type_radio = gr.Radio(choices=["Search", "Chat (Include History)"], label="Search Type")
            rag_type_dropdown = gr.Dropdown(choices=["RAG Fusion", ""], label="Select type of retrieval")

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=server,  # Dummy function for testing
                additional_inputs=[model_dropdown, temp_slider, n_results_slider, search_type_radio, rag_type_dropdown],
                title="AI-powered RAG Web Crawler",
                description="An AI-powered search assistant using Retrieval Augmented Generation (RAG).",
            )
    
    gr.Markdown("### Powered by Qdrant, Gradio, and LangChain")

if __name__ == "__main__":
    ui.launch()