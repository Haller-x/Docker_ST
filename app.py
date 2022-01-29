import os
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from io import BytesIO
import streamlit as st
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

@st.cache(allow_output_mutation=True)
def load_h5():
    """
    Carrega o modelo em .h5 e armazena em cache
    NO ARGS
    RETURN:
    modelo preditivo
    """
    model = tf.keras.models.load_model('model.h5')
    model.make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    return model

def load_and_save(img_bytes):
    '''
    Abre a imagem e salva num jpg padrão
    ARGS:
    img_bytes <- imagem em bytes
    '''
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')
    img.save("sample.jpg")

def get_img_array(img_path, size):
    '''
    Transforma a imagem em um tensor!
    ARGS:
    img_path <- string contendo o caminho da imagem
    size <- tupla com dois inteiros
    RETURN:
    array <- np.array
    '''
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    '''
    img_array <- np.array da imagem
    model <- objeto do modelo
    last_conv_layer_name <- nome da ultima layer de convolução (str)
    pred_index <- int - index da predição
    '''
    # Primeiro, criamos um modelo que mapeia a imagem de entrada para as ativações
    # da última camada de convolução, bem como as previsões de saída.
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )


    # Em seguida, calculamos o gradiente da classe predita (com maior probabildiade) 
    # para nossa imagem de entrada com relação às ativações da última camada de convolução
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradiente do output do neurônio de saida em relação ao mapa de feature da ultima
    #camada de convolução
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vetor para cada entrada é a média da intensidade do gradiente 
    # em relação ao mapa de feature
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))


    # Multiplicamos cada canal na matriz do mapa de features
    # pelo gradiente (importância do canal) em relação à classe predita com maior probabilidade
    # então soma-se os canais para obter a ativação da classe do mapa de calor
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizando entre 0 e 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    '''
    Salva e realiza a exibição do gradiente em relação a imagem (sobreposição)
    ARGS:
    img_path <- caminho da imagem (str)
    heatmap <- mapa de calor numpy (np.array)
    cam_path < argumento para salvar a imagem
    alpha = ajuste de transparência
    RETURN:
    imagem sobreposta(apenas display)
    '''
    
    # imagem original
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # rescalando a imagem entre até 255
    heatmap = np.uint8(255 * heatmap)

    # jet para colorizar o "gradiente"
    jet = cm.get_cmap("jet")

    # usando os valores RGB do heatmap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Criando uma imagem com RGB colorido com o heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Sobreposição das imagens
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Salvando a imagem para realizar o display
    superimposed_img.save(cam_path)

    # Display 
    st.write('GRAD-CAM:')
    st.image(cam_path)
    # Excluí a imagem do caminho salvo
    os.remove(cam_path)


def load_image(image_path):
    '''
    Carrega imagem no formato ideal realiza o pré-processamento
    ARGS:
    image_path <- str: caminho da imagem
    RETURN:
    img <- imagem processada
    '''
    img = tf.io.decode_image(image_path, channels=3)
    img = tf.image.resize(img, (300, 300))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

## funções de estilo apenas - são códigos snippet padrão ##
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Feito com ",
        image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(25), height=px(25)),
        " com ❤️ por ",
        link("https://github.com/haller-x", "@Haller-x"),
        br(),
    ]
    layout(*myargs)
## 

if __name__ == "__main__":
    #confidurações da página
    st.set_page_config(page_title='CERTI - Classificador de cachorro ou gato')
    st.title("TensorFlow Classificador de Cães e gatos + Streamlit! ✨🖼️")
    st.header("UI para usar o modelo em TensorFlow  criado usando 1400 images para treino do dataset The Oxford-IIIT Pet")
    hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
    #lógica para carregar o modelo e footer
    st.markdown(hide_footer_style, unsafe_allow_html=True)
    footer()
    model = load_h5()
    uploaded_file = st.file_uploader(label="Faça o upload de imagem para gerar a classificação ",
                                    type=["png", "jpeg", "jpg"])

    if not uploaded_file:
        st.warning("Por favor, adicione uma imagem antes de continuar")
        st.stop()
    else:
        image_as_bytes = uploaded_file.read()
        st.image(image_as_bytes, use_column_width=True)
        pred_button = st.button("Predizer")

    if pred_button:
        #aplicando as funções definidas acima
        img = tf.expand_dims(load_image(image_as_bytes), 0)
        load_and_save(image_as_bytes)
        img = get_img_array('sample.jpg',size=(300,300))
        with st.spinner('Calculando resultados...'):
            pred = model.predict(img)
            classes = ['cat (gato)','dog (cachorro)']
            heatmap = make_gradcam_heatmap(img, model, 'top_conv', pred_index=np.argmax(pred))
        save_and_display_gradcam('sample.jpg', heatmap)
        st.write(f"Predição: {classes[np.argmax(pred)]}, Probabilidade {max(pred[0])}")
    
