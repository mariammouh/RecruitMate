
# RecruitMate: AI-Powered Recruiting Assistant

![alt text](logo.png)

RecruitMate helps recruiters quickly screen resumes. Upload PDF CVs, set job requirements, and get ranked candidates with detailed matching insights.

---

## 🚀 Features

* Extracts key info from resumes (name, skills, experience, education, etc.).
* Define custom job requirements (skills, degree, experience, languages).
* Ranks candidates by match percentage.
* Expandable candidate view with details.
* Export results to PDF/CSV.
* Simple Streamlit web interface.

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/mariammouh/RecruitMate.git
cd RecruitMate

# Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
.\venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

* Enter job requirements
* Upload PDF resumes
* View ranked candidates
* Download results

---

## 📂 Project Structure

```
RecruitMate/
├── .streamlit/         # Streamlit config
│   └── config.toml
├── .vscode/            # VS Code settings
├── venv/               # Virtual environment (not shared in repo)
├── .gitignore
├── app.py              # Main Streamlit app
├── example_resume.pdf  # Example CV for testing
├── icon.png            # App icon
├── logo.png            # Project logo
├── README.md           # Documentation
└── requirements.txt    # Dependencies

```

---

## 🛠️ Built With

* Python 3.x
* Streamlit (UI)
* spaCy (NLP)
* pdfplumber (PDF parsing)
* pandas (data handling)

---

## 🌟 Future Ideas

* Smarter semantic matching
* Adjustable weights for requirements
* Support for DOCX resumes
* Dashboards & analytics
* Multi-language support

---

## 🤝 Contributing

Contributions are welcome! Open an issue or submit a pull request.

---

## 👩 Author

Developed by Mariam Mouh 💡
