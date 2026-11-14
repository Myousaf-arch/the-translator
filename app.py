from datetime import datetime
import json
import logging
import re
import time
from typing import Dict, List, Tuple

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ==============================================================================
# 1. ENTERPRISE LOGGING & SYSTEM CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [LINGUACORE-NLLB200] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="The Translator | Enterprise Neural Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. CUSTOM ORGANIC LUXURY CSS STYLING (Client Palette Integration)
# ==============================================================================
st.markdown(
    """
<style>
    /* Palette Constants:
       Mercury: #EDE7DA
       Shadow: #84785F
       Coffee: #A67956
       Nutmeg: #743C30
       Lunar Green: #3A4E37
    */

    .stApp {
        background-color: #EDE7DA;
        color: #3A4E37;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f7f4ee;
        border-right: 1px solid #84785F;
    }

    /* Premium Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #3A4E37 0%, #293826 100%);
        border: 1px solid #84785F;
        border-radius: 12px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(58, 78, 55, 0.15);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #EDE7DA;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #EDE7DA;
        opacity: 0.85;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #84785F;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(132, 120, 95, 0.1);
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #743C30;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #84785F;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* Output Box Styling */
    .output-box {
        background: #ffffff;
        border: 1px solid #84785F;
        border-radius: 10px;
        padding: 24px;
        font-size: 1.15rem;
        line-height: 1.8;
        color: #3A4E37;
        min-height: 200px;
        box-shadow: inset 0 2px 6px rgba(132, 120, 95, 0.08);
    }

    /* Input Textarea */
    textarea {
        background-color: #ffffff !important;
        color: #3A4E37 !important;
        border: 1px solid #84785F !important;
        border-radius: 8px !important;
    }

    /* Custom Button */
    div.stButton > button:first-child {
        background: #A67956;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: #743C30;
        box-shadow: 0 4px 12px rgba(116, 60, 48, 0.25);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6, label {
        color: #3A4E37 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. META NLLB-200 UNIFIED NEURAL ENGINE REGISTRY
# ==============================================================================
ENGINE_MODEL_ID = "facebook/nllb-200-distilled-600M"
SOURCE_LANG_CODE = "eng_Latn"

LANGUAGE_REGISTRY: Dict[str, Dict[str, str]] = {
    "Urdu (اردو)": {
        "nllb_code": "urd_Arab",
        "short_code": "UR",
        "direction": "rtl",
        "region": "South Asia",
    },
    "Hindi (हिन्दी)": {
        "nllb_code": "hin_Deva",
        "short_code": "HI",
        "direction": "ltr",
        "region": "South Asia",
    },
    "Arabic (العربية)": {
        "nllb_code": "arb_Arab",
        "short_code": "AR",
        "direction": "rtl",
        "region": "Middle East & North Africa",
    },
    "French (Français)": {
        "nllb_code": "fra_Latn",
        "short_code": "FR",
        "direction": "ltr",
        "region": "Europe",
    },
    "Spanish (Español)": {
        "nllb_code": "spa_Latn",
        "short_code": "ES",
        "direction": "ltr",
        "region": "Europe & Americas",
    },
    "German (Deutsch)": {
        "nllb_code": "deu_Latn",
        "short_code": "DE",
        "direction": "ltr",
        "region": "Central Europe",
    },
}

SAMPLE_TEMPLATES: Dict[str, str] = {
    "Custom Input": "",
    "Natural Conversational Check": (
        "Hey! How are you doing today, and how has your weekend been so far? I"
        " hope everything is going smoothly on your end."
    ),
    "Enterprise Business Proposal": (
        "We are pleased to submit our strategic proposal for scaling your cloud"
        " infrastructure. Our architecture guarantees 99.99% uptime, zero-trust"
        " security compliance, and automated failover across global regions."
    ),
    "Legal & Compliance Clause": (
        "Both parties agree to maintain strict confidentiality regarding all"
        " proprietary algorithms, source code repositories, and client"
        " datasets disclosed during the term of this agreement."
    ),
}


# ==============================================================================
# 4. CORE NEURAL ENGINE & HARDWARE OPTIMIZATION
# ==============================================================================
def get_compute_device() -> Tuple[torch.device, str]:
  if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    return torch.device("cuda"), f"GPU ({device_name})"
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return torch.device("mps"), "Apple Silicon GPU (MPS)"
  return torch.device("cpu"), "CPU (Multi-Threaded)"


@st.cache_resource(show_spinner=False)
def load_nllb_engine() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM, str]:
  device, device_label = get_compute_device()
  logger.info(
      f"Initializing Unified NLLB-200 Engine '{ENGINE_MODEL_ID}' on"
      f" {device_label}..."
  )

  tokenizer = AutoTokenizer.from_pretrained(
      ENGINE_MODEL_ID, src_lang=SOURCE_LANG_CODE
  )
  model = AutoModelForSeq2SeqLM.from_pretrained(ENGINE_MODEL_ID)
  model.to(device)
  model.eval()

  logger.info(f"NLLB-200 Engine loaded successfully on {device_label}.")
  return tokenizer, model, device_label


def smart_sentence_chunker(text: str, max_chars: int = 450) -> List[str]:
  paragraphs = text.split("\n")
  chunks: List[str] = []

  for para in paragraphs:
    para = para.strip()
    if not para:
      chunks.append("")
      continue

    if len(para) <= max_chars:
      chunks.append(para)
    else:
      sentences = re.split(r"(?<=[.!?])\s+", para)
      current_chunk = ""
      for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
          current_chunk = (
              f"{current_chunk} {sentence}".strip()
              if current_chunk
              else sentence
          )
        else:
          if current_chunk:
            chunks.append(current_chunk)
          current_chunk = sentence
      if current_chunk:
        chunks.append(current_chunk)

  return chunks


def execute_neural_translation(
    text: str,
    target_nllb_code: str,
    num_beams: int = 2,
    max_length: int = 256,
    length_penalty: float = 1.0,
) -> Tuple[str, float, int]:
  start_time = time.perf_counter()
  tokenizer, model, _ = load_nllb_engine()
  device, _ = get_compute_device()

  target_lang_id = tokenizer.convert_tokens_to_ids(target_nllb_code)
  chunks = smart_sentence_chunker(text)
  translated_chunks: List[str] = []
  total_tokens_processed = 0

  for chunk in chunks:
    if chunk == "":
      translated_chunks.append("")
      continue

    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    total_tokens_processed += inputs["input_ids"].shape[1]

    with torch.no_grad():
      output_tokens = model.generate(
          **inputs,
          forced_bos_token_id=target_lang_id,
          max_length=max_length,
          num_beams=num_beams,
          length_penalty=length_penalty,
          early_stopping=True,
          no_repeat_ngram_size=3,
      )

    decoded_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    translated_chunks.append(decoded_text)

  final_output = "\n".join(translated_chunks)
  elapsed_ms = (time.perf_counter() - start_time) * 1000
  return final_output, elapsed_ms, total_tokens_processed


# ==============================================================================
# 5. SESSION STATE INITIALIZATION
# ==============================================================================
if "translation_history" not in st.session_state:
  st.session_state.translation_history = []

if "last_output" not in st.session_state:
  st.session_state.last_output = ""

if "last_metrics" not in st.session_state:
  st.session_state.last_metrics = {
      "latency_ms": 0.0,
      "tokens": 0,
      "words": 0,
      "target_lang": "N/A",
  }


# ==============================================================================
# 6. SIDEBAR: ENGINE CONTROLS
# ==============================================================================
with st.sidebar:
  st.markdown("### Engine Controls")
  st.caption("Meta NLLB-200 Neural Architecture")

  selected_language = st.selectbox(
      "Target Language",
      options=list(LANGUAGE_REGISTRY.keys()),
      index=0,
  )

  lang_meta = LANGUAGE_REGISTRY[selected_language]
  st.info(
      f"Model: {ENGINE_MODEL_ID}\n\nTag: {lang_meta['nllb_code']} | Script:"
      f" {lang_meta['direction'].upper()}"
  )

  st.divider()
  st.markdown("### Decoding Parameters")

  num_beams = st.slider("Beam Search Width", min_value=1, max_value=6, value=2)
  max_tokens = st.slider("Max Output Tokens", min_value=64, max_value=512, value=256, step=32)
  length_penalty = st.slider("Length Penalty", min_value=0.6, max_value=1.6, value=1.0, step=0.1)

  st.divider()
  _, active_hw = get_compute_device()
  st.markdown(f"Compute Node: {active_hw}")

  if st.button("Clear Session History"):
    st.session_state.translation_history = []
    st.session_state.last_output = ""
    st.rerun()


# ==============================================================================
# 7. MAIN WORKSPACE UI
# ==============================================================================
st.markdown(
    """
<div class="hero-container">
    <div class="hero-title">The Translator</div>
    <p class="hero-subtitle">High-throughput contextual neural translation suite built for professional localization workflows.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Top Telemetry Bar
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
  st.markdown(
      f"""<div class="metric-card">
            <div class="metric-value">{st.session_state.last_metrics['latency_ms']:.0f} ms</div>
            <div class="metric-label">Inference Latency</div>
        </div>""",
      unsafe_allow_html=True,
  )
with m_col2:
  st.markdown(
      f"""<div class="metric-card">
            <div class="metric-value">{st.session_state.last_metrics['tokens']}</div>
            <div class="metric-label">Tokens Processed</div>
        </div>""",
      unsafe_allow_html=True,
  )
with m_col3:
  st.markdown(
      f"""<div class="metric-card">
            <div class="metric-value">{st.session_state.last_metrics['words']}</div>
            <div class="metric-label">Source Words</div>
        </div>""",
      unsafe_allow_html=True,
  )
with m_col4:
  st.markdown(
      f"""<div class="metric-card">
            <div class="metric-value">{len(st.session_state.translation_history)}</div>
            <div class="metric-label">Total Requests</div>
        </div>""",
      unsafe_allow_html=True,
  )

st.write("")

template_choice = st.selectbox(
    "Quick-Load Sample Text:", options=list(SAMPLE_TEMPLATES.keys())
)
default_text = SAMPLE_TEMPLATES[template_choice]

left_col, right_col = st.columns(2, gap="large")

with left_col:
  st.markdown("### Source Text (English)")
  source_text = st.text_area(
      label="Source Input",
      value=default_text,
      height=240,
      placeholder="Enter text here...",
      label_visibility="collapsed",
  )

  char_count = len(source_text)
  word_count = len(source_text.split()) if source_text.strip() else 0
  st.caption(f"Characters: {char_count} | Words: {word_count}")

  translate_clicked = st.button("Execute Neural Translation")

with right_col:
  st.markdown(f"### Translated Output — {selected_language}")

  if translate_clicked:
    if not source_text.strip():
      st.warning("Please enter source text before executing translation.")
    else:
      try:
        with st.spinner("Processing neural inference..."):
          translated_result, latency_ms, tokens_used = (
              execute_neural_translation(
                  text=source_text,
                  target_nllb_code=lang_meta["nllb_code"],
                  num_beams=num_beams,
                  max_length=max_tokens,
                  length_penalty=length_penalty,
              )
          )

        st.session_state.last_output = translated_result
        st.session_state.last_metrics = {
            "latency_ms": latency_ms,
            "tokens": tokens_used,
            "words": word_count,
            "target_lang": selected_language,
        }

        st.session_state.translation_history.insert(
            0,
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "target_language": selected_language,
                "model": ENGINE_MODEL_ID,
                "source": source_text,
                "translation": translated_result,
                "latency_ms": round(latency_ms, 2),
                "tokens": tokens_used,
            },
        )
        st.rerun()

      except Exception as exc:
        logger.error(f"Pipeline failure: {exc}")
        st.error(f"Pipeline Error: {exc}")

  text_dir = lang_meta["direction"]
  align_style = "right" if text_dir == "rtl" else "left"
  display_content = (
      st.session_state.last_output
      if st.session_state.last_output
      else "<span style='color:#84785F;'>Translated output will appear here...</span>"
  )

  formatted_html_content = display_content.replace("\n", "<br>")

  st.markdown(
      f"""<div class="output-box" dir="{text_dir}" style="text-align: {align_style};">
            {formatted_html_content}
        </div>""",
      unsafe_allow_html=True,
  )

  if st.session_state.last_output:
    st.write("")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
      st.download_button(
          label="Download TXT",
          data=st.session_state.last_output,
          file_name=f"translation_{lang_meta['short_code'].lower()}.txt",
          mime="text/plain",
          use_container_width=True,
      )
    with dl_col2:
      json_payload = json.dumps(
          st.session_state.translation_history[0],
          ensure_ascii=False,
          indent=2,
      )
      st.download_button(
          label="Export JSON",
          data=json_payload,
          file_name=f"translation_{lang_meta['short_code'].lower()}.json",
          mime="application/json",
          use_container_width=True,
      )

# ==============================================================================
# 8. AUDIT LOG
# ==============================================================================
if st.session_state.translation_history:
  st.divider()
  st.markdown("### Session Audit Log")

  for idx, entry in enumerate(st.session_state.translation_history[:5]):
    with st.expander(
        f"#{idx + 1} | [{entry['timestamp']}] English →"
        f" {entry['target_language']} ({entry['latency_ms']} ms)",
        expanded=(idx == 0),
    ):
      h_col1, h_col2 = st.columns(2)
      with h_col1:
        st.caption("Source (EN)")
        st.write(entry["source"])
      with h_col2:
        st.caption(f"Target ({entry['target_language']})")
        st.write(entry["translation"])