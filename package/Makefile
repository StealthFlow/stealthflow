export PWD=`pwd`
export NAME_CONTAINER=stealthflow/pypi-uploader
export MAIN_VERSION=0.0.15

build: 
	docker build -f Dockerfile -t $(NAME_CONTAINER) .

void:
	docker run -it --rm -v $(PWD):/work $(NAME_CONTAINER) /bin/bash

upload-test:
	rm -rf dist StealthFlow.egg-info/ build dist
	docker run -it --rm -v $(PWD):/work $(NAME_CONTAINER) \
		python setup.py bdist_wheel --version $(MAIN_VERSION)
	docker run -it --rm -v $(PWD):/work $(NAME_CONTAINER) \
		twine upload --skip-existing --repository testpypi dist/*

clean:
	docker run -it --rm -v $(PWD):/work $(NAME_CONTAINER) \
		rm -rf build dist *.egg-info	
	
upload-main:
	# PyPI
	python /Users/n/.local/lib/python3.6/site-packages/twine upload --repository pypi dist/*