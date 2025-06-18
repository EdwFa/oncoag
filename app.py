import copy
import json
from typing import Iterable, Dict, Any
import streamlit as st
# import logging
import datetime
# st.set_page_config(
#     page_title="Ex-stream-ly Cool App",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://www.extremelycoolapp.com/help',
#         'Report a bug': "https://www.extremelycoolapp.com/bug",
#         'About': "# This is a header. This is an *extremely* cool app!"
#     }
# )

from streamlit_ace import st_ace
from groq import Groq
import os
import httpx

from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk, MOAgentConfig
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

import textwrap
from fpdf import FPDF



# from pathlib import Path
# from weasyprint import HTML
# import markdown

# Default configuration
default_main_agent_config = {
    "main_model": "qwen-qwq-32b",
    "cycles": 1,
    "temperature": 0.1,
    "system_prompt": SYSTEM_PROMPT,
    "reference_system_prompt": REFERENCE_SYSTEM_PROMPT
}

default_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": """\
        You are a qualified medical assistant with deep knowledge in the field of oncology. Your task is to assist oncologists in analyzing clinical situations, interpreting examination results, choosing treatment tactics and providing up-to-date scientific information.
        You must:
            Have knowledge of modern protocols for diagnosing and treating oncological diseases.
            Consider the recommendations of international oncological societies (e.g. NCCN, ESMO, ASCO).
            Be familiar with tumor classifications (ICD-10, TNM, histological types).
            Understand the methods of radiation, chemotherapy, targeted, immunotherapy and surgical treatment.
            Facilitate decision-making based on evidence-based medicine and the latest clinical trials.
            Provide information on possible side effects of therapy and ways to correct them.
            Analyze laboratory and instrumental data from the point of view of oncological pathology.
            Do not make a final diagnosis or replace the attending physician, but act as an auxiliary tool for the specialist.
        Your answer should be:
            Scientifically sound
            Clear and structured
            Up-to-date (as of 2024–2025)
            No unnecessary jargon, but with professional precision
            If necessary, with sources or links to recommendations
       
        Think through your response step by step. {helper_response}",
        """,
        "model_name": "llama-3.3-70b-versatile",
        "temperature": 0.2
    },
    "layer_agent_2": {
        "system_prompt": """\
        You are a qualified medical assistant with deep knowledge in the field of oncology. Your task is to assist oncologists in analyzing clinical situations, interpreting examination results, choosing treatment tactics and providing up-to-date scientific information.
        You must:
            Have knowledge of modern protocols for diagnosing and treating oncological diseases.
            Consider the recommendations of international oncological societies (e.g. NCCN, ESMO, ASCO).
            Be familiar with tumor classifications (ICD-10, TNM, histological types).
            Understand the methods of radiation, chemotherapy, targeted, immunotherapy and surgical treatment.
            Facilitate decision-making based on evidence-based medicine and the latest clinical trials.
            Provide information on possible side effects of therapy and ways to correct them.
            Analyze laboratory and instrumental data from the point of view of oncological pathology.
            Do not make a final diagnosis or replace the attending physician, but act as an auxiliary tool for the specialist.
        Your answer should be:
            Scientifically sound
            Clear and structured
            Up-to-date (as of 2024–2025)
            No unnecessary jargon, but with professional precision
            If necessary, with sources or links to recommendations
        Respond with a thought and then your response to the question. {helper_response}""",
        "model_name": "deepseek-r1-distill-llama-70b",
        "temperature": 0.2
    },
    "layer_agent_3": {
        "system_prompt": """\
        You are a qualified medical assistant with deep knowledge in the field of oncology. Your task is to assist oncologists in analyzing clinical situations, interpreting examination results, choosing treatment tactics and providing up-to-date scientific information.
        You must:
            Have knowledge of modern protocols for diagnosing and treating oncological diseases.
            Consider the recommendations of international oncological societies (e.g. NCCN, ESMO, ASCO).
            Be familiar with tumor classifications (ICD-10, TNM, histological types).
            Understand the methods of radiation, chemotherapy, targeted, immunotherapy and surgical treatment.
            Facilitate decision-making based on evidence-based medicine and the latest clinical trials.
            Provide information on possible side effects of therapy and ways to correct them.
            Analyze laboratory and instrumental data from the point of view of oncological pathology.
            Do not make a final diagnosis or replace the attending physician, but act as an auxiliary tool for the specialist.
        Your answer should be:
            Scientifically sound
            Clear and structured
            Up-to-date (as of 2024–2025)
            No unnecessary jargon, but with professional precision
            If necessary, with sources or links to recommendations
        You are an medical expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}
        """,
        "model_name": "gemma2-9b-it",
        "temperature": 0.2
    },
}

# Recommended Configuration
rec_main_agent_config = {
    "main_model": "qwen-qwq-32b",
    "cycles": 1,
    "temperature": 0.1,
    "system_prompt": SYSTEM_PROMPT,
    "reference_system_prompt": REFERENCE_SYSTEM_PROMPT
}

rec_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama-3.3-70b-versatile",
        "temperature": 0.1
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "deepseek-r1-distill-llama-70b",
        "temperature": 0.2,
        "max_tokens": 2048
    },
    "layer_agent_3": {
        "system_prompt": "You are an emedical xpert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "qwen-qwq-32b",
        "temperature": 0.4,
    },
    "layer_agent_4": {
        "system_prompt": "You are an medical expert planner agent. Create a plan for how to answer the human's query. {helper_response}",
        "model_name": "gemma2-9b-it",
        "temperature": 0.2
    },
}

# === Настройка логирования ===
LOG_FILE = "oncobot.log"
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(message)s",
#     handlers=[
#         logging.FileHandler(LOG_FILE, encoding="utf-8"),
#         # logging.StreamHandler()
#     ]
# )

# def log_query(question: str):
#     """Функция для записи вопроса и ответа в лог"""
#     logging.info(question)

def write_to_log(filename: str, text: str):
    """
    Записывает текст в файл с указанием даты и времени.

    :param filename: Имя файла, в который будет записан текст.
    :param text: Текст для записи.
    """
    # Получаем текущее время
    now = datetime.now()
    print(now)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Формируем строку для записи
    log_entry = f"[{timestamp}] {text}\n"

    # Открываем файл в режиме добавления и записываем текст
    with open(filename, "a", encoding="utf-8") as file:
        file.write(log_entry)

    # print(f"Сообщение записано в {filename}")


from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from streamlit.web.server.server import Server


def get_session_id():
    """Получает уникальный ID сессии пользователя"""
    ctx = get_script_run_ctx()
    if ctx is None:
        return "unknown_session"
    return ctx.session_id


def _get_session():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info

def get_client_ip():
    """Пытается получить IP-адрес клиента"""
    session_info = _get_session()
    if session_info is None:
        return "unknown_ip"
    print(session_info)
    return
    # Получаем IP из заголовков запроса
    headers = session_info.request.headers
    x_forwarded_for = headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    remote_ip = headers.get("Remote-Addr")
    return remote_ip or "unknown_ip"


def write_to_file_with_metadata(filename: str, question: str):
    """
    Записывает вопрос пользователя в файл вместе с метаданными:
    - Дата и время
    - Session ID
    - IP-адрес

    :param filename: Имя файла для записи
    :param question: Текст вопроса пользователя
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_id = get_session_id()
    ip_address = get_client_ip()

    log_entry = f"[{timestamp}] [IP: {ip_address}] [Session: {session_id}] Question: {question}\n"

    with open(filename, "a", encoding="utf-8") as file:
        file.write(log_entry)

    st.info(f"Вопрос записан в лог: {log_entry.strip()}")


def view_file_contents(filename: str):
    """
    Отображает содержимое текстового файла в интерфейсе Streamlit.
    :param filename: Имя файла, содержимое которого нужно отобразить.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        if content:
            st.text_area("Содержимое файла:", value=content, height=400)
        else:
            st.info("Файл пуст.")
    except FileNotFoundError:
        st.error(f"Файл '{filename}' не найден.")

from datetime import datetime
def download_log_button(log_file_path: str = "oncobot.log"):
    """
    Отображает кнопку 'Скачать лог' и позволяет скачать файл с логами.

    :param log_file_path: Путь к файлу лога (по умолчанию 'user_questions.log')
    """
    try:
        with open(log_file_path, "rb") as f:
            log_data = f.read()

        # Генерируем имя файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        downloadable_name = f"log_{timestamp}.txt"

        st.download_button(
            label="🔽 Скачать лог",
            data=log_data,
            file_name=downloadable_name,
            mime="text/plain",
            key="download_log_button"
        )

    except FileNotFoundError:
        st.warning("Файл лога не найден. Пока нет записей для скачивания.")


# Ключи интер=фейсов
# GROQ_API_KEY = "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" - старый
# GROQ_API_KEY = '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'
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", default=None)
# print(GROQ_API_KEY)

# client = Groq(
#     api_key=os.environ.get("GROQ_API_KEY"),
#     http_client=httpx.Client(verify=False)  # Disable SSL verification
# )
# os.environ['GROQ_API_KEY'] = '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'
# print(GROQ_API_KEY)
if 'api_key' not in st.session_state:
    st.session_state.api_key = GROQ_API_KEY
if 'groq' not in st.session_state:
    if GROQ_API_KEY:
        # st.session_state.groq = Groq()
        st.session_state.groq = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
            http_client=httpx.Client(verify=False)  # Disable SSL verification
        )


# Helper functions
def json_to_moa_config(config_file) -> Dict[str, Any]:
    config = json.load(config_file)
    moa_config = MOAgentConfig( # To check if everything is ok
        **config
    ).model_dump(exclude_unset=True)
    return {
        'moa_layer_agent_config':moa_config.pop('layer_agent_config', None),
        'moa_main_agent_config': moa_config or None
    }

def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message['response_type'] == 'intermediate':
            layer = message['metadata']['layer']
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message['delta'])
        else:
            # Display accumulated layer outputs
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, output in enumerate(outputs):
                    with cols[i]:
                        st.expander(label=f"Agent {i+1}", expanded=False).write(output)
            
            # Clear layer outputs for the next iteration
            layer_outputs = {}
            
            # Yield the main agent's output
            yield message['delta']

def set_moa_agent(
    moa_main_agent_config = None,
    moa_layer_agent_config = None,
    override: bool = False
):
    moa_main_agent_config = copy.deepcopy(moa_main_agent_config or default_main_agent_config)
    moa_layer_agent_config = copy.deepcopy(moa_layer_agent_config or default_layer_agent_config)

    if "moa_main_agent_config" not in st.session_state or override:
        st.session_state.moa_main_agent_config = moa_main_agent_config

    if "moa_layer_agent_config" not in st.session_state or override:
        st.session_state.moa_layer_agent_config = moa_layer_agent_config

    if override or ("moa_agent" not in st.session_state):
        st_main_copy = copy.deepcopy(st.session_state.moa_main_agent_config)
        st_layer_copy = copy.deepcopy(st.session_state.moa_layer_agent_config)
        st.session_state.moa_agent = MOAgent.from_config(
            **st_main_copy,
            layer_agent_config=st_layer_copy
        )

        del st_main_copy
        del st_layer_copy

    del moa_main_agent_config
    del moa_layer_agent_config

# App
st.set_page_config(
    page_title="Sechenov Onco-LLMAgents ",
    page_icon='static/favicon.ico',
         menu_items={
        'About': "## LLM-Onco-Agents-SU"
    },
    layout="wide",
    initial_sidebar_state="collapsed",
)

# valid_model_names = [model.id for model in Groq().models.list().data if not (model.id.startswith("whisper") or model.id.startswith("llama-guard"))]

valid_model_names = [
    'llama-3.3-70b-versatile',
    'gemma2-9b-it',
    'mistral-saba-24b',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
    'qwen-qwq-32b',
    'deepseek-r1-distill-llama-70b'
]


# st.markdown("<a href='https://groq.com'><img src='app/static/banner.png' width='500'></a>", unsafe_allow_html=True)
# st.write("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

set_moa_agent()

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    # upl_col, load_col = st.columns(2)
    st.download_button(
        "Download",
        data=json.dumps({
            **st.session_state.moa_main_agent_config,
            'moa_layer_agent_config': st.session_state.moa_layer_agent_config
        }, indent=2),
        file_name="moa_config.json"
    )

    # moa_config_upload = st.file_uploader("Choose a JSON file", type="json")
    # submit_config_file = st.button("Update config")
    # if moa_config_upload and submit_config_file:
    #     try:
    #         moa_config = json_to_moa_config(moa_config_upload)
    #         set_moa_agent(
    #             moa_main_agent_config=moa_config['moa_main_agent_config'],
    #             moa_layer_agent_config=moa_config['moa_layer_agent_config']
    #         )
    #         st.session_state.messages = []
    #         st.success("Configuration updated successfully!")
    #     except Exception as e:
    #         st.error(f"Error loading file: {str(e)}")

    with st.form("Agents ..", border=False):
        # Load and Save moa config file

        if st.form_submit_button("Use Recommended Config"):
            try:
                set_moa_agent(
                    moa_main_agent_config=rec_main_agent_config,
                    moa_layer_agent_config=rec_layer_agent_config,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

        # Main model selection
        new_main_model = st.selectbox(
            "Main Model",
            options=valid_model_names,
            index=valid_model_names.index(st.session_state.moa_main_agent_config['main_model'])
        )

        # Cycles input
        new_cycles = st.number_input(
            "Iterations",
            min_value=1,
            max_value=10,
            value=st.session_state.moa_main_agent_config['cycles']
        )

        # Main Model Temperature
        main_temperature = st.number_input(
            label="Main-Agent Temperature",
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )

        # Layer agent configuration
        tooltip = "Agents in the layer agent configuration run in parallel _per cycle_. Each layer agent supports all initialization parameters of [Langchain's ChatGroq](https://api.python.langchain.com/en/latest/chat_models/langchain_groq.chat_models.ChatGroq.html) class as valid dictionary fields."
        st.markdown("Layers", help=tooltip)
        new_layer_agent_config = st_ace(
            key="layer_agent_config",
            value=json.dumps(st.session_state.moa_layer_agent_config, indent=2),
            language='json',
            placeholder="Layer Agent Configuration (JSON)",
            show_gutter=False,
            wrap=True,
            auto_update=True
        )

        with st.expander("Optional Main Agent Params", expanded=True):
            tooltip_str = """\
Main Agent configuration that will respond to the user based on the layer agent outputs.
Valid fields:
- ``system_prompt``: System prompt given to the main agent. \
**IMPORTANT**: it should always include a `{helper_response}` prompt variable.
- ``reference_prompt``: This prompt is used to concatenate and format each layer agent's output into one string. \
This is passed into the `{helper_response}` variable in the system prompt. \
**IMPORTANT**: it should always include a `{responses}` prompt variable. 
- ``main_model``: Which Groq powered model to use. Will overwrite the model given in the dropdown.\
"""
            tooltip = tooltip_str
            st.markdown("Main Agent Config", help=tooltip)
            new_main_agent_config = st_ace(
                key="main_agent_params",
                value=json.dumps(st.session_state.moa_main_agent_config, indent=2),
                language='json',
                placeholder="Main Agent Configuration (JSON)",
                show_gutter=False,
                wrap=True,
                auto_update=True
            )

        if st.form_submit_button("Update Configuration"):
            try:
                new_layer_config = json.loads(new_layer_agent_config)
                new_main_config = json.loads(new_main_agent_config)
                # Configure conflicting params
                # If param in optional dropdown == default param set, prefer using explicitly defined param
                if new_main_config.get('main_model', default_main_agent_config['main_model']) == default_main_agent_config["main_model"]:
                    new_main_config['main_model'] = new_main_model
                
                if new_main_config.get('cycles', default_main_agent_config['cycles']) == default_main_agent_config["cycles"]:
                    new_main_config['cycles'] = new_cycles

                if new_main_config.get('temperature', default_main_agent_config['temperature']) == default_main_agent_config['temperature']:
                    new_main_config['temperature'] = main_temperature
                
                set_moa_agent(
                    moa_main_agent_config=new_main_config,
                    moa_layer_agent_config=new_layer_config,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

        if st.form_submit_button("Logview"):
            view_file_contents(LOG_FILE)
        # if st.form_submit_button("Download"):
    download_log_button(log_file_path="oncobot.log")

    # st.markdown("---")
    # st.markdown("""
    # ### Credits
    # - MOA: [Together AI](https://www.together.ai/blog/together-moa)
    # - LLMs: [Groq](https://groq.com/)
    # - Paper: [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
    # """)


def text_to_pdf(text, filename):
    a4_width_mm = 210
    pt_to_mm = 0.35
    fontsize_pt = 10
    fontsize_mm = fontsize_pt * pt_to_mm
    margin_bottom_mm = 10
    character_width_mm = 7 * pt_to_mm
    width_text = a4_width_mm / character_width_mm
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(True, margin=margin_bottom_mm)
    pdf.add_page()
    pdf.set_font(family='Arial', size=fontsize_pt)
    splitted = text.split('\n')
    for line in splitted:
        lines = textwrap.wrap(line, width_text)
        if len(lines) == 0:
            pdf.ln()
        for wrap in lines:
            pdf.cell(0, fontsize_mm, wrap, ln=1)
    pdf.output(filename, 'F')

def save_markdown(text: str, filename: str) -> bool:
    """
    Сохраняет текст в формате Markdown в файл с указанным именем.

    Параметры:
    text (str): Текст в формате Markdown для сохранения
    filename (str): Имя целевого файла (с расширением .md или без)

    Возвращает:
    bool: True если сохранение успешно, False в случае ошибки
    """
    try:
        # Добавляем расширение .md если отсутствует
        if not filename.endswith('.md'):
            filename += '.md'

        # Создаем директории если нужно
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Записываем содержимое в файл
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)

        return True

    except Exception as e:
        print(f'Ошибка сохранения: {str(e)}')
        return False

# Пример использования
# if __name__ == "__main__":
#     content = """# Пример Markdown
#
# ## Список дел
# - [x] Написать функцию сохранения
# - [ ] Протестировать код
# - [ ] Добавить в проект
#
# ```python
# print("Hello Markdown!")
# ```"""
#
#     if save_markdown(content, 'example_file'):
#         print("Файл успешно сохранен!")
#     else:
#         print("Не удалось сохранить файл")



# def md_to_pdf(input_md: str, output_pdf: str, css: str = None) -> bool:
#     """
#     Конвертирует Markdown-файл в PDF с сохранением форматирования
#
#     Параметры:
#     input_md (str): Путь к исходному .md файлу
#     output_pdf (str): Путь для сохранения .pdf файла
#     css (str): Опциональный путь к CSS-файлу для стилизации PDF
#
#     Возвращает:
#     bool: True при успешной конвертации, False при ошибке
#     """
#     try:
#         # Проверка расширений файлов
#         if not input_md.endswith('.md'):
#             raise ValueError("Input file must be a .md file")
#
#         if not output_pdf.endswith('.pdf'):
#             output_pdf += '.pdf'
#
#         # Создание директорий для выходного файла
#         Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
#
#         # Чтение Markdown-контента
#         with open(input_md, 'r', encoding='utf-8') as f:
#             md_content = f.read()
#
#         # Конвертация Markdown в HTML
#         html_content = markdown.markdown(md_content)
#
#         # Базовые CSS-стили
#         default_css = """
#         body { font-family: Arial, sans-serif; line-height: 1.6; margin: 2cm; }
#         h1, h2, h3 { color: #2c3e50; }
#         code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
#         pre { background: #333; color: #fff; padding: 15px; overflow-x: auto; }
#         a { color: #3498db; text-decoration: none; }
#         """
#
#         # Сборка полного HTML-документа
#         full_html = f"""
#         <!DOCTYPE html>
#         <html>
#             <head>
#                 <meta charset="utf-8">
#                 <style>{default_css}</style>
#                 {f'<link rel="stylesheet" href="{css}">' if css else ''}
#             </head>
#             <body>{html_content}</body>
#         </html>
#         """
#
#         # Генерация PDF
#         HTML(string=full_html).write_pdf(output_pdf)
#
#         return True
#
#     except Exception as e:
#         print(f"Ошибка конвертации: {str(e)}")
#         return False


# Пример использования
# if __name__ == "__main__":
#     if md_to_pdf('example_file.md', 'output.pdf'):
#         print("PDF успешно создан!")
#     else:
#         print("Ошибка при создании PDF")


# Main app layout
st.header("Sechenov Onco Assistant-&-Consultant", anchor=False, divider='blue')
st.write("Prototype Oncology medical AI-consultant, Powered by Sechenov University on GroQ")

# Display current configuration
with st.status("Current Configuration", expanded=False, state='complete') as config_status:
    # st.image("./static/moa_groq.svg", caption="Mixture of Agents Workflow", use_column_width='always')
    st.markdown(f"**Main Agent Config**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.moa_main_agent_config, indent=2),
        language='json',
        placeholder="Layer Agent Configuration (JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )
    st.markdown(f"**Layer Agents Config**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.moa_layer_agent_config, indent=2),
        language='json',
        placeholder="Layer Agent Configuration (JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )

if st.session_state.get("message", []) != []:
    st.session_state['expand_config'] = False

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # text_to_pdf(message["content"], 'reports/report.pdf')
        # print(message["content"])
        st.markdown(message["content"])


if query := st.chat_input("Ask a question"):
    config_status.update(expanded=False)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
        # write_to_file_with_metadata(LOG_FILE, query)
        write_to_log(LOG_FILE, query)
        # print(query)
        # logging.info(query)


    moa_agent: MOAgent = st.session_state.moa_agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        ast_mess = stream_response(moa_agent.chat(query, output_format='json'))
        response = st.write_stream(ast_mess)
        save_markdown(response, 'result.md')
        # if md_to_pdf('result.md', 'output.pdf'):
        #     print("PDF успешно создан!")
        # else:
        #     print("Ошибка при создании PDF")
        # print(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})