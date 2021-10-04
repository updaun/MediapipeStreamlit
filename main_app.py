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

        # mp_drawing.draw_landmarks(
        #     image = out_image,
        #     landmark_list = results.pose_landmarks,
        #     connections = mp_holistic.POSE_CONNECTIONS,
        #     landmark_drawing_spec = drawing_spec
        # ) 
        if results.pose_landmarks:
            face_count += 1
            mp_drawing.draw_landmarks(
                out_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                out_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                out_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                out_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)

elif app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")

    if record:
        st.checkbox("Recording", value=True)


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

    # st.markdown("**Detected Faces, Hands and Pose**")
    # kpi1_text = st.markdown("0")
    


    max_faces = st.sidebar.number_input('Maximum Number of Face', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    st.markdown('## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])

    tffile = tempfile.NamedTemporaryFile(delete=False)

    ## We get out input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO

    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # Recording Part
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text("Input Video")
    st.sidebar.video(tffile.name)
    
    fps = 0
    i = 0

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Holistic**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    ## Dashboard
    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence = detection_confidence,
        min_tracking_confidence = tracking_confidence
        ) as holistic:
    
        prevTime = 0

        while vid.isOpened():
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue

            results = holistic.process(frame)
            frame.flags.writeable = True

            face_count = 0

            if results.pose_landmarks:
                face_count += 1
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # mp_drawing.draw_landmarks(
                # image = frame,
                # landmark_list = results.pose_landmarks,
                # connections = mp_holistic.POSE_CONNECTIONS,
                # landmark_drawing_spec = drawing_spec,
                # connection_drawing_spec = drawing_spec) 

            # FPS Counter logic
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            if record:
                out.write(frame)

            ## Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame, channels = 'BGR', use_column_width = True)

        ## Holistic Landmark Drawing
        # for holistic_landmarks in results.face_landmarks:
        # face_count += 1

        

            # kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        # st.image(out_image, use_column_width=True)