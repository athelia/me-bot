# me-bot

Personal bot project.

## Requirements

Platform-specific setup guides:

- [macOS](docs/requirements-macos.md)
- [Windows](docs/requirements-windows.md)

Both guides cover Python, Docker, and Ollama setup (including the Docker-based workflow shared with `household-llm`).

## To-Do

### Fix/Refactor

- [x] **Persona & prompt customization.** The persona name `spacepiratemog` (and nickname `Mog`) is hardcoded throughout:
  - `loaders/discord_chat.py`: `TARGET_AUTHOR` (used by `_is_target_author` to decide which messages are the bot's)
  - `chat.py`: both system prompts (`PROMPT_WITH_CONTEXT`, `PROMPT_NO_CONTEXT`), the labels in `format_pair_context`, `format_statement_context` and `format_context`, and the `Mog:` label in `append_chat_log` and `main()`
  - `eval_rag.py`, `preview_pairs.py`, `preview_discord.py`: report labels / headers
  - [x] move the persona name (and display nickname) into a setup/config file and reference that variable everywhere above, including author matching
  - [ ] extract the RAG prompts (`PROMPT_WITH_CONTEXT`, `PROMPT_NO_CONTEXT`) out of `chat.py` into a separate file, and make them customizable by the user
- [x] **Fix** `preview_discord.py`**.** It imports `load_discord_chats` from `loaders/discord_chat.py`, which was removed in `e1d1b66`, so the script fails on import. Switch it to the current loader functions (`load_discord_statements` matches what it previews: the bot's own messages with timestamp/source metadata).
- [x] **Dedupe settings.** Create one settings module (e.g. `settings.py`) and import it in `main.py`, `chat.py` and the other scripts. Currently duplicated:
  - `CHROMA_DIR`: `main.py` (`Path("data/chroma")`) and `chat.py` (`"data/chroma"` as a plain string; pick one type)
  - `COLLECTION_NAME`: `main.py`, `chat.py`
  - `OLLAMA_BASE_URL`, `EMBED_MODEL`, `LLM_MODEL` (env lookups + defaults): `main.py`, `chat.py`
- [ ] Dedupe settings continued
  - `DISCORD_DIR`: `main.py`, `export_finetune_data.py`, `preview_discord.py`, `preview_pairs.py`
  - `load_dotenv()`: `main.py`, `chat.py`, `eval_rag.py` (call it once, in the settings module)
  - embeddings construction: `create_embeddings()` in `main.py` vs. inline `OllamaEmbeddings(...)` in `chat.py`'s `get_vector_store()`; share one function
  - chat-only settings (`TOP_K`, `STATEMENT_K`, `FETCH_K`, `MAX_DISTANCE`, `MAX_REPLY_TOKENS`, `SHOW_SOURCES`, `LOG_PATH`) can move there too for consistency
- [ ] `generate_reply` **in** `chat.py`**: build clients once, retrieve once.**
  - [ ] Every call to `generate_reply` rebuilds `OllamaEmbeddings` + `Chroma` (via `get_retrieved_docs` -> `get_vector_store`) and `ChatOllama` (via `get_llm`). Move client building into a setup function that runs only when first needed (lazily) and reuse the clients for every message.
  - [ ] With `SHOW_SOURCES=true`, `main()` calls `get_retrieved_docs` to print sources, then `generate_reply` calls it again, so retrieval runs twice per message. `eval_rag.py --generate` has the same double retrieval (`get_retrieved_docs` + `chain.invoke`). Have `generate_reply` return the retrieved docs together with the reply (or accept already-retrieved docs), and update `main()`, `build_rag_chain` and `eval_rag.py` to use that.
  - [ ] Move the `SHOW_SOURCES` conditional (the block in `main()` that prints retrieved pairs/statements, or the "no results passed similarity threshold" note) into `generate_reply`, so sources are shown from the same retrieval that produced the reply.
  - [ ] Delete `build_rag_chain()` (it only wraps `generate_reply` in a `RunnableLambda` for `eval_rag.py`) and have `eval_rag.py` call `generate_reply` directly.
- [ ] **Extract file-walking in** `loaders/discord_chat.py`**.** `load_discord_pairs` and `load_discord_statements` contain the same loop (`is_dir` check, `sorted(rglob)`, suffix filter, `read_text`), so `load_discord_documents` reads and parses every export file twice. Extract one function that walks the directory and returns the file texts (or parsed messages), and have both loaders (and `load_discord_documents`) use its results.
- [ ] **Strip out unused code**
  - [ ] `main.py`: `PyPDFLoader` import, `PDF_PATH`, `LLM_MODEL`
  - [ ] `main.py`: replace `vector_store._collection.count()` (private attribute) with a public API or `len(documents)`
  - [ ] `chat.py`: `format_docs` (labelled "backward compatibility for eval_rag.py", but nothing calls it)
  - [ ] `chat.py`: `sanitize_response` / `_MARKDOWN_NOISE` (call is commented out in `generate_reply`); either restore it or delete it. If deleted, reconsider rendering replies with `rich.markdown.Markdown` in `main()`, since replies should be plain text.



### Improvements

Discord input: other users' messages become the incoming side of conversation pairs (`_format_incoming` in `loaders/discord_chat.py`).

- [ ] include sentiment analysis on other users' messages, per message
- [ ] remove links and images
- [ ] **Better filters for Discord export noise.** Filtering lives in `_clean_line` in `loaders/discord_chat.py` (per-line checks against `_EMBED_BOILERPLATE`, `_DISCORD_UI_LINES`, `_EMOJI_SHORTCODE`, `_STANDALONE_DATE`, plus "started a call" / "server tag:"). Examples that still get through:
  - [ ] `Game Invitation Whiteboard Game ended. Start a new one? Play`
  - [ ]      `Mog: get your phone spacepiratemog'`
  - [ ] three consecutive lines: `Vophren`, then `used`, then `Launch` (the first line is a username, so line-by-line exact matching can't catch this; it needs a multi-line pattern or a look-ahead at the following lines)
  - [ ] `You missed a call from synalle`



### Low Priority (Won't Fix)

Known weaknesses.

- [ ] **No tests for regex parsing** in `loaders/discord_chat.py`: `_MESSAGE_HEADER`, `_URL_PATTERN`, `_EMOJI_SHORTCODE`, `_STANDALONE_DATE`, `_clean_line`, `is_low_signal_content`. A small fixture file of sample export text would cover most of it.
- [ ] **Tag/comma username format.** `_normalize_author` only strips whitespace and a trailing comma, so a header like `synalle [PRTN],` leaves the author as `synalle [PRTN]`. The tag ends up in `incoming_authors` metadata and in the `author: text` lines embedded in pair documents, and author matching would fail if the target author ever had a tag.
- [ ] **Raw** `requirements.txt` **/** `requirements-windows.txt`**.** Both are a full `pip freeze` (~100 pinned packages). Most are transitive deps (e.g. `fastapi`, `kubernetes`, `onnxruntime` come from `chromadb`), and some direct ones are unused (`openai`, `langchain-openai`, `pypdf`, `langchain-community` once `PyPDFLoader` is gone). Replace with just the direct dependencies (`python-dotenv`, `langchain-core`, `langchain-chroma`, `langchain-ollama`, `chromadb`, `rich`), optionally with a separate lock file. Also, `requirements-windows.txt` is saved as UTF-16 (likely from a PowerShell `>` redirect); re-save it as UTF-8 so it diffs and greps normally.