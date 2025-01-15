ve:
	python3.11 -m venv .ve
	. .ve/bin/activate
	pip install -r requirements.txt

check:
	ruff check .

fix:
	ruff check --fix .

format:
	ruff format .

docker_build:
	docker build -t llm-app .

docker_start:
	docker run -p 7860:7860 --name llm-app -d llm-app

run_v1:
	chainlit run 01_Prompt\ Engineering\ and\ Prototyping\ Best\ Practices/app_v1.py -w --port 8001

run_v2:
	chainlit run 01_Prompt\ Engineering\ and\ Prototyping\ Best\ Practices/app_v2.py -w --port 8002