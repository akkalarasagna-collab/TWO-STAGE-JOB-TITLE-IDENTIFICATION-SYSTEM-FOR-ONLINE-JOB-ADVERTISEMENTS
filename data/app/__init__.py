# app/__init__.py — Flask application factory

from flask import Flask
from config import Config
import database
from bert_classifier import load_bert_model


def create_app():
    """Create and configure the Flask application."""

    app = Flask(
        __name__,
        template_folder='../../templates',
        static_folder='../../static'
    )

    # Load configuration
    app.config.from_object(Config)

    # Initialise SQLite database (creates history.db + table if needed)
    database.init_db()

    # Pre-load the CrossEncoder NLI model once at startup so the first
    # prediction request is not penalised by model initialisation time.
    print("Loading BERT model... please wait")
    load_bert_model()
    print("BERT model loaded successfully")

    # Register the main blueprint
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
