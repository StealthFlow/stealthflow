build-tf23:
	docker build -f Dockerfile.tf23sf0014 -t stealthflow/tf23-test .

docker run -it --rm -v $(PWD):/work $(NAME_CONTAINER) \
	python test.py