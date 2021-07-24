export PWD = `pwd`
export NAME_DEV = stealthflow/sf-test
export NAME_TF23 = stealthflow/tf23-test # 
export NAME_TF24 = stealthflow/tf24-test # kaggle / DeepLearning AMI
export NAME_TF25 = stealthflow/tf25-test # colab

pretest:
	docker build -f docker/Dockerfile.dev -t $(NAME_DEV) .
	docker run -it --rm \
		-v $(PWD):/work \
		-v $(PWD)/../:/sf \
		$(NAME_DEV) /bin/bash

tf23:
	docker build -f docker/Dockerfile.tf23sf0014 -t $(NAME_TF23) .
	docker run -it --rm -v $(PWD):/work $(NAME_TF23) \
		/bin/bash		
#python test.py

tf24:
	docker build -f docker/Dockerfile.tf23sf0014 -t $(NAME_TF24) .
	docker run -it --rm -v $(PWD):/work $(NAME_TF24) \
		python test.py

tf25:
	docker build -f docker/Dockerfile.tf23sf0014 -t $(NAME_TF25) .
	docker run -it --rm -v $(PWD):/work $(NAME_TF25) \
		python test.py