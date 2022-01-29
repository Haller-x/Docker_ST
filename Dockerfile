#imagem base
FROM python:3.8-slim
#porta de exposição 
EXPOSE 8501
#ambiente de trabalho
WORKDIR /app
#copia dos requisitos
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
#copiando tudo
COPY . .
#rodando aplicação
CMD streamlit run app.py

#configurações streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
