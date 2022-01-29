#imagem base
FROM python:3.8-slim
#ambiente de trabalho
WORKDIR /app
#copia dos requisitos
COPY requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
#copiando tudo
COPY . .
#porta de exposição 
EXPOSE 8501
#rodando aplicação
CMD streamlit run  --server.port $PORT app.py


#configurações streamlit
RUN mkdir -p ~/.streamlit/
RUN bash -c echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml
