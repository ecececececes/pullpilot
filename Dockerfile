FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pullpilot/ pullpilot/

# Bake in the 28-PR sample dataset so the Review page's example dropdown and
# the static engine work out of the box, fully offline.
RUN mkdir -p data/examples && python -m pullpilot.benchmark.make_example_dataset

EXPOSE 5000
CMD ["python", "-m", "pullpilot.web"]
