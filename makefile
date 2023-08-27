install:
	python -m pip install --upgrade pip &&\
		pip install -r requirement.txt

# install-gcp:
# 	pip install --upgrade pip &&\
# 		pip install -r requirements-gcp.txt

# install-aws:
# 	pip install --upgrade pip &&\
# 		pip install -r requirements-aws.txt

# install-amazon-linux:
# 	pip install --upgrade pip &&\
# 		pip install -r amazon-linux.txt
lint:
	pylint --disable=R,C main.py

format:
	black *.py

test:
	python -m pytest -vv --cov=hello test.py
