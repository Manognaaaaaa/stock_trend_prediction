# Streamlit demo for Hugging Face Spaces (Docker SDK).
# Build from the repo root:  docker build -f app/Dockerfile -t tadawul-demo .
# Run:                       docker run -p 8501:8501 tadawul-demo
# On the Space, set `app_port: 8501` in the Space README front matter.
FROM python:3.11-slim

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/code

COPY --chown=user app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir --user -r app/requirements.txt

COPY --chown=user app/ app/

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
