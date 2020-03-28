all: docker

docker:
	docker build -f Dockerfile -t jek_img .
	docker run --name jek_cont --volume="${PWD}:/srv/jekyll" -p 3000:4000 -it --rm jek_img

test:
	docker build -f Dockerfile -t jek_img .
	docker run --name jek_cont --volume="${PWD}:/srv/jekyll" -p 3000:4000 -it --rm jek_img bash
