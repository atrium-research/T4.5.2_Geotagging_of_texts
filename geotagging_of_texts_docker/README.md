# Geotagging of texts docker

A web app that finds historical place names in a PDF, TXT or TEI/XML file, links them to the ToposText knowledge base, shows them on a map, and exports a TEI/XML or JSON file.

It runs fully on your own computer, on **CPU or NVIDIA GPU**, from one Docker image. All AI models are inside the image and are loaded when the app starts, so nothing loads while you work.

## 1. Requirements

| You need | Why | Check it |
|---|---|---|
| **Docker** (Docker Desktop on Windows/macOS, Docker Engine + Compose plugin on Linux) | builds and runs the app | `docker --version` and `docker compose version` |
| **Git** | downloads the project | `git --version` |
| An internet connection | for the first build (about 7 GB of models and packages) and for the map tiles | |
| About **40 GB free disk** and **16 GB RAM** for Docker | the image is about 30 GB; all models stay loaded together | |

GPU (optional): an NVIDIA GPU with about 6 GB of memory, a recent NVIDIA driver, and on Linux the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Docker Desktop on Windows works with the WSL2 backend. The CPU mode needs nothing extra.

## 2. Get the project

```bash
git clone https://github.com/atrium-research/T4.5.2_Geotagging_of_texts.git
cd geotagging-of-texts/geotagging_of_texts_docker
```

## 3. Settings (`.env`)

Rename the `.env.example` to `.env` and edit the file in the project folder.

| Setting | What it does | Default |
|---|---|---|
| `APP_PORT` | port in your browser, `http://localhost:<port>` | `8000` |
| `CHAT_MODEL` | the LLM (Hugging Face GGUF, `owner/repo:quant`). Another model is downloaded the first time you start the app, and needs internet | `unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M` |
| `SAVE_RUNS` | `true` keeps uploads and results between restarts. `false` keeps them only until the container stops (runs older than an hour are deleted) | `false` |
| `LLAMA_CONTEXT` | context window of the models, in tokens. Raise it (for example `16384`) if a run stops with "paragraph is too long for the model's context window". Uses more memory | `4096` |
| `LLAMA_THREADS` | CPU threads per model, `0` = automatic | `0` |
| `MAX_UPLOAD_MB` | largest file you can upload | `50` |
| `BIND_ADDR` | `127.0.0.1` = only this computer. See the security note below | `127.0.0.1` |

Do not remove the `COMPOSE_PROFILES=cpu` line: it makes plain `docker compose up` start the CPU mode.

The OCR, reranker, embedding and place-recognition models are fixed and cannot be changed.

## 4. Build (once)

Run this inside the project folder:

```bash
docker compose build
```

The first build takes a while. It installs the packages, downloads the models into the image, and loads each one to check it works.

## 5. Start the app

**On the CPU:**

```bash
docker compose up
```

**On an NVIDIA GPU:**

```bash
docker compose --profile gpu up
```

Wait for these lines in the terminal:

```
Loading models, please wait...
...
The models are ready. Go to http://localhost:8000
```

Then open **http://localhost:8000** in your browser, upload a file, and press **Start workflow**. When it finishes, click a highlighted place or a map pin, review the results, and download the TEI XML or JSON.

To stop the app, press `Ctrl+C` in the terminal, or run `docker compose down` (add `--profile gpu` if you started the GPU mode).