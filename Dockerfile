FROM jekyll/jekyll:4.0

RUN apk add --update vim

CMD jekyll serve
