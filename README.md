# 📄 PDF Question Paper Generator API

A Flask-based API service that receives JSON data and generates professionally styled A4 question papers in PDF format using ReportLab and LaTeX.

---

## 🚀 Features

- Accepts POST requests with JSON payloads.
- Supports rendering:
  - ✅ Plain text questions
  - ✅ Math expressions (LaTeX)
  - ✅ Tables
  - ✅ Images from URLs
  - ✅ **Multilingual text** (e.g., Hindi, Arabic, Tamil, French, etc.)
- Professional **A4 page formatting** with proper spacing, margins, and layout.
- Custom **font embedding** for multilingual text compatibility (e.g., Devanagari, Arabic script).
- Returns a **print-ready, high-quality PDF**.

---

## 🧱 Tech Stack

- **Python 3**
- **Flask** – Web framework for the API
- **ReportLab** – PDF rendering engine
- **LaTeX / MathText** – For math formula support
- **Pillow** – Image support
- **io** – In-memory PDF streaming
- **Unicode fonts** – For multilingual rendering

---

🌐 Multilingual Support

To ensure all languages render correctly, the app uses Unicode-compatible fonts (e.g., Noto Sans, DejaVu Sans, Amiri, etc.). Fonts are auto-selected based on character sets.

You may extend this by:

- Adding .ttf files for your custom fonts

- Using pdfmetrics.registerFont() to load them

- Dynamically selecting fonts based on language detection (already partially implemented)

# Example input

### [neet_question_json](Neet_question_json_1.json)

# Example output in pdf format

### [neet_question_pdf](Neet_question_demo.pdf)

👤 Author

Akintola Olamilekan
Built for schools, universities, educators, and e-learning platforms that need printable, multilingual question papers from dynamic content.
