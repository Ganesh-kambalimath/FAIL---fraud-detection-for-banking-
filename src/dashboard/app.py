import json
import os
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Path to the simulation metrics log file
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs', 'simulation_metrics.json')

@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/data')
def get_data():
    """Provides the simulation data as JSON."""
    if not os.path.exists(LOG_FILE):
        return jsonify({"error": "Log file not found."})

    try:
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format in log file."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
