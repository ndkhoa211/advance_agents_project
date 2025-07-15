# Advanced Agents Project

> **Multi‑agent experiments in LangChain** – a router agent that delegates to specialised sub‑agents (Python REPL & CSV), plus a modern *tool‑calling* agent powered by OpenAI / Anthropic.

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://www.python.org/)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-9cf?logo=langchain)
[![Tavily](https://img.shields.io/badge/Powered%20by-Tavily-ffc72c?logo=tavily)](https://www.tavily.com/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-ff69b4?logo=langchain)](https://smith.langchain.com/o/856312b1-7816-4389-80cb-b01e398655be/projects/p/e96289ca-0c2a-45a0-8d45-4dafe8db2a92?timeModel=%7B%22duration%22%3A%227d%22%7D)
![License](https://img.shields.io/badge/license-MIT-lightgrey)


---

## What I built

* **Python REPL agent** – translates natural‑language tasks into runnable Python via `PythonAstREPLTool` and streams the results back. ([raw.githubusercontent.com](https://raw.githubusercontent.com/ndkhoa211/advance_agents_project/main/main.py))
* **CSV analytics agent** – answers data questions over *`episode_info.csv`* using Pandas (`create_csv_agent`). ([raw.githubusercontent.com](https://raw.githubusercontent.com/ndkhoa211/advance_agents_project/main/main.py))
* **Grand *router* agent** – a top‑level ReAct agent that chooses the right sub‑agent (`Tool` wrappers) for each user request. ([raw.githubusercontent.com](https://raw.githubusercontent.com/ndkhoa211/advance_agents_project/main/main.py))
* **Tool‑calling chat agent** – showcases OpenAI/Anthropic function‑calling with real tools (`TavilySearchResults`, custom `multiply`). ([raw.githubusercontent.com](https://raw.githubusercontent.com/ndkhoa211/advance_agents_project/main/main.py))
* **`main.py` demo script** – runs sample queries like QR‑code generation or live weather comparison Taipei vs Kaohsiung. ([raw.githubusercontent.com](https://raw.githubusercontent.com/ndkhoa211/advance_agents_project/main/main.py))

---

## Quick start

```bash
# 1️⃣ Clone & enter
$ git clone https://github.com/ndkhoa211/advance_agents_project.git
$ cd advance_agents_project

# 2️⃣ Create an isolated env (uv is recommended)
$ uv venv            # ➜ .venv/
$ source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3️⃣ Install runtime deps
$ uv pip install -e .

# 4️⃣ Set your secrets
$ cat > .env <<'EOF'
OPENAI_API_KEY=sk‑...
ANTHROPIC_API_KEY=sk‑anthropic‑...
TAVILY_API_KEY=tvly‑...
EOF

# 5️⃣ Run the demo script
$ python main.py
```

**Example output** (truncated):

```
Start...
:::Hello Tool Calling:::
| Taipei | 32 °C | Kaohsiung 29 °C |
...
```

---

## Repository structure

```text
advance_agents_project/
├── main.py                # All agents + demo calls
├── episode_info.csv       # Sample data for CSV agent
├── pyproject.toml         # Dependencies 
├── uv.lock                # Exact package versions (frozen)
└── README.md              # ← you are here
```

---

## Dependencies

The project targets **Python 3.12+** and pins versions via *uv*:

| Package                                                                    | Purpose                                 |
| -------------------------------------------------------------------------- | --------------------------------------- |
| `langchain`, `langchain‑openai`, `langchain‑anthropic`, `langchain‑tavily` | LLM back‑ends & tool‑calling            |
| `langchain‑experimental`                                                   | `PythonAstREPLTool`, `create_csv_agent` |
| `pandas`, `matplotlib`                                                     | Data wrangling & plotting               |
| `qrcode`                                                                   | QR‑code generation                      |
| `rich`                                                                     | Pretty console output                   |

See **`pyproject.toml`** for the full list. ([raw.githubusercontent.com](https://raw.githubusercontent.com/ndkhoa211/advance_agents_project/main/pyproject.toml))

---

## Environment variables

| Variable            | Role                                 |
| ------------------- | ------------------------------------ |
| `OPENAI_API_KEY`    | ChatGPT (router, Python, CSV agents) |
| `ANTHROPIC_API_KEY` | Claude (tool‑calling agent)          |
| `TAVILY_API_KEY`    | Web search tool                      |

Store them in a local `.env` – they are loaded automatically by `python‑dotenv`.

---

## License

This repository is released under the **MIT License** – see [`LICENSE`](LICENSE) for details.

