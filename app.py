# app.py
from shiny import App
from components.ui_components import create_ui
from components.chat_logic import server

app = App(create_ui(), server)
