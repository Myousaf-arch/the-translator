# The Translator | Enterprise Neural Studio

A high-throughput contextual neural localization and translation suite powered by Meta's **NLLB-200 (`facebook/nllb-200-distilled-600M`)** architecture and built with **Streamlit**.

## Key Features
- **Multilingual Neural Engine:** High-precision contextual translation across global languages including Urdu (RTL support), Arabic, French, Spanish, German, and Chinese.
- **Enterprise Telemetry:** Real-time inference latency tracking, token counter, and compute device monitoring (CPU/CUDA GPU).
- **Custom Decoding Controls:** Interactive adjustments for Beam Search Width, Max Output Tokens, and Length Penalty.
- **Session Audit Log:** Automatic history tracking with one-click `.txt` and `.json` export workflows.

## Quickstart Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Myousaf-arch/the-translator.git
   cd the-translator
   pip install -r requirements.txt
   streamlit run app.py
   ```
