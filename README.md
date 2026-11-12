Installation and Setup

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/artificially-disabled.git
cd artificially-disabled

2. Create a virtual environment

python -m venv .venv

3. Activate the environment

Windows PowerShell:

.venv\Scripts\Activate.ps1

4. Install dependencies

pip install -r requirements.txt

5. Configure the Gemini API key

Create a .env file in the project root and add:

GEMINI_API_KEY=your_gemini_api_key

Never publish your API key or commit your .env file.

6. Run the application

python -m streamlit run app.py

Open the local URL displayed in the terminal to access the application.