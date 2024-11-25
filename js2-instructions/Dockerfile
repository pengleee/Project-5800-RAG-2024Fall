FROM bitnami/kubectl
USER root
WORKDIR /home
COPY js2.sh /home/
EXPOSE 5800
ENTRYPOINT ["bash", "js2.sh"]
