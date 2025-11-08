# BRAMPS Server

A minimal Flask web application for the BRAMPS (Bayesian Regression Analysis for Multiple Planetary Systems) project.

## Project Structure

```
nops/
├── app.py              # Main Flask application
├── static/             # Static files (CSS, JS, images)
├── templates/          # HTML templates
│   └── index.html      # Main page template
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

1. Make sure your virtual environment is activated

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

You should see the "BRAMPS Server running" page.

## Development Mode

The server runs in debug mode by default, which means:
- Automatic reloading when code changes
- Detailed error messages in the browser
- The server is accessible from any IP address (0.0.0.0)

For production deployment, you should disable debug mode and use a production WSGI server like Gunicorn or uWSGI.

## Dependencies

- **Flask**: Web framework for Python
- **NumPy**: Numerical computing library
- **SciPy**: Scientific computing library
- **Pandas**: Data analysis and manipulation library

## Next Steps

This is a minimal skeleton. You can extend it by:
- Adding more routes in `app.py`
- Creating additional templates in `templates/`
- Adding CSS/JavaScript files in `static/`
- Implementing BRAMPS-specific functionality
