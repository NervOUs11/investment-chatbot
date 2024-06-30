import gradio as gr
from chatbot import Chatbot

bot = Chatbot()


def response(question, chat_history):
    bot_answer = bot.answer(question)
    chat_history.append((question, bot_answer))
    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Ask the question")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[chatbot, msg], value="Clear console")

    btn.click(response, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(response, inputs=[msg, chatbot], outputs=[msg, chatbot])

gr.close_all()

demo.launch()