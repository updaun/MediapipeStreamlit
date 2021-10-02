from mediapipe.python.solutions.face_mesh import FACE_CONNECTIONS
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

DEMO_IMAGE = './demo/demo.png'
DEMO_VIDEO = './demo/demo.mp4'

st.title('Mediapipe Streamlit App')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,

    unsafe_allow_html=True,
)

st.sidebar.title('Mediapipe Streamlit Sidebar')
st.sidebar.subheader('parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter =cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))

    
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


app_mode = st.sidebar.selectbox('Choose the App mode', ['About App', 'Run on Image', 'Run on Video'])


if app_mode == 'About App':
    st.markdown('In this Application we are using **Mediapipe** for creating a Holistic App. **Streamlit** is to create the Web Graphical User Interface (GUI)')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:350px
            margin-left: -350px
        }
        </style>
        """,

        unsafe_allow_html=True,
    )
    st.video('https://youtu.be/wyWmWaXapmI')

    st.markdown('''
                # About Our Team \n
                Hey this is ** Hi-Pipe ** from ** GIS ** \n

                If you are interested in playing game with Mediapipe. \n

                -
    ''')

elif app_mode == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:350px
            margin-left: -350px
        }
        </style>
        """,

        unsafe_allow_html=True,
    )

    st.markdown("**Detected Faces, Hands and Pose**")
    kpi1_text = st.markdown("0")

    max_faces = st.sidebar.number_input('Maximum Number of Face', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE 
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    
    face_count = 0

    ##Dashboard
    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence = detection_confidence
        ) as holistic:
    
        results = holistic.process(image)
        out_image = image.copy()

        ## Holistic Landmark Drawing
        # for holistic_landmarks in results.face_landmarks:
        face_count += 1

        mp_drawing.draw_landmarks(
            image = out_image,
            landmark_list = results.pose_landmarks,
            connections = mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec = drawing_spec
        ) 

            # kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)