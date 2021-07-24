export PWD = `pwd`
export NAME_DEV = stealthflow/sf-test
export NAME_TF23 = stealthflow/tf23-test # 
export NAME_TF24 = stealthflow/tf24-test # kaggle / DeepLearning AMI
export NAME_TF25 = stealthflow/tf25-test # colab

pretest: ## run in local before upload modules
	docker build -f docker/Dockerfile.dev -t $(NAME_DEV) .
	docker run -it --rm \
		-v $(PWD):/work \
		-v $(PWD)/../:/sf \
		$(NAME_DEV) python3 scripts/test.py

tf23: ## build&run with tensorflow-gpu 2.3
	docker build -f docker/Dockerfile.tf23sf0014 -t $(NAME_TF23) .
	docker run -it --rm -v $(PWD):/work $(NAME_TF23) \
		scripts/test.py	

tf24: ## build&run with tensorflow-gpu 2.4
	docker build -f docker/Dockerfile.tf23sf0014 -t $(NAME_TF24) .
	docker run -it --rm -v $(PWD):/work $(NAME_TF24) \
		python scripts/test.py

tf25: ## build&run with tensorflow-gpu 2.5
	docker build -f docker/Dockerfile.tf23sf0014 -t $(NAME_TF25) .
	docker run -it --rm -v $(PWD):/work $(NAME_TF25) \
		python scripts/test.py

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
