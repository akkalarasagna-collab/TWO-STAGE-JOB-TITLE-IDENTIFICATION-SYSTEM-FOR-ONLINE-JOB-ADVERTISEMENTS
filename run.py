# run.py — Entry point for the Flask application

from data.app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
