from flask import Flask, request, render_template_string, flash, redirect, url_for
import torch
import logging
from typing import Dict, Any
from Network_structure import NeuralNetwork
from config import INPUT_FEATURES, OUTPUT_FEATURES, EVAL_MODEL_FILE


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flashing messages
logging.basicConfig(level=logging.INFO)

# Global history list to store input dictionaries.
# For production, consider storing history in a database or session.
history = []

def parse_input(form: Dict[str, Any]) -> Dict[str, float]:
    """
    Parses input values from the form and converts them to floats.
    If a value is missing or invalid, raises a ValueError.
    """
    inputs = {}
    for key in INPUT_FEATURES:
        raw_value = form.get(key, "").strip()
        if raw_value == "":
            inputs[key] = 0.0
        else:
            try:
                inputs[key] = float(raw_value)
            except ValueError as e:
                logging.error("Conversion error for feature '%s': %s", key, e)
                raise ValueError(f"Invalid input for {key}. Please enter a numeric value.") from e
    return inputs

# Load the trained model
model = NeuralNetwork()
try:
    model.load_state_dict(torch.load(EVAL_MODEL_FILE, map_location=torch.device('cpu')))
    model.eval()
    logging.info("Model loaded and set to evaluation mode.")
except Exception as e:
    logging.error("Error loading the model: %s", e)
    raise e

# Build HTML template string. In a real project, use separate template files.
html_template = """
<html>
    <head>
        <title>Model Prediction</title>
    </head>
    <body>
        <h1>Enter Inputs for Prediction</h1>
        {% with messages = get_flashed_messages() %}
          {% if messages %}
              <ul>{% for message in messages %}<li>{{ message }}</li>{% endfor %}</ul>
          {% endif %}
        {% endwith %}
        <form method="POST">
"""
for feature in INPUT_FEATURES:
    # Use Jinja2 templating for form values.
    html_template += (
        f'  {feature}: <input type="text" name="{feature}" value="{{{{ form_values.get("{feature}", "") }}}}" /><br/>\n'
    )
html_template += """
            <input type="submit" value="Predict"/>
        </form>
        <h2>Result: {{ result }}</h2>
        {{ history_html|safe }}
    </body>
</html>
"""

def generate_history_html() -> str:
    """
    Generates an HTML table for the input history.
    The table headers are determined by INPUT_FEATURES and OUTPUT_FEATURES.
    """
    if not history:
        return "<h3>History</h3><p>No history available.</p>"
    
    # Build table headers.
    headers = "".join(f"<th>{feature}</th>" for feature in (INPUT_FEATURES + OUTPUT_FEATURES))
    html_str = f"<h3>History</h3><table border='1'><tr>{headers}</tr>"
    
    # Generate table rows for each history entry.
    for entry in history:
        row = "".join(f"<td>{entry.get(feature, '')}</td>" for feature in (INPUT_FEATURES + OUTPUT_FEATURES))
        html_str += f"<tr>{row}</tr>"
    
    html_str += "</table>"
    return html_str

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handles form submission, prediction, and displaying the input history.
    """
    global history
    result = ""
    # Default form values for repopulation.
    form_values = {feature: "" for feature in INPUT_FEATURES}
    
    if request.method == "POST":
        try:
            # Parse and validate inputs.
            inputs = parse_input(request.form)
            # Create a tensor from the inputs.
            input_tensor = torch.tensor([list(inputs.values())])
            
            # Get prediction from the model.
            with torch.no_grad():
                prediction = model(input_tensor)
            result = f"{prediction.item():.3f}"
            
            # Append the result to the input dictionary.
            inputs.update({OUTPUT_FEATURES[0]: result})
            history.append(inputs)
            logging.info("Prediction successful: %s", result)
            
            # Update form values for repopulation.
            form_values = {k: str(v) for k, v in inputs.items()}
        except Exception as e:
            error_message = f"Error: {e}"
            flash(error_message)
            logging.error(error_message)
            # Redirect to clear the form while preserving flash messages.
            return redirect(url_for("index"))
    
    history_html = generate_history_html()
    return render_template_string(html_template, form_values=form_values, result=result, history_html=history_html)

if __name__ == "__main__":
    # Enable debug mode for development.
    app.run(debug=True)
