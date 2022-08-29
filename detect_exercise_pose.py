import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

DEMO_VIDEO = 'squart.mp4'
DEMO_IMAGE = 'img.jpg'
DEMO_IMAGE_RESULT = 'result.jpg'

def detect_squat(arm_angle, hip_angle, knee_angle, knee_r, knee_l):
    if  knee_angle < 100 and hip_angle < 100:
        res = "Squart" 
        
    elif hip_angle > 150 and knee_angle > 150 :
        res = "Wide Squart"
    elif abs(shoulder_r[1]-knee_r[1] - (shoulder_l[1]-knee_l[1])) > 0.2 :
        res = "Lunge"
    else:
        res = "False"
    return res

def calculate_angle(a,b,c):
    a = np.array(a) # 첫번째
    b = np.array(b) # 두번쩨
    c = np.array(c) # 마지막
    
    # 아크탄젠트-> 탄젠트의 역함수
    # -180 < =arctan2 <= 180
    # 1 degree = π / 180 radian (180도 = π radian)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)  
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

st.title('Mediapipe를 활용한 운동 자세 인식')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('운동 횟수 세기 Sidebar')
app_mode = st.sidebar.selectbox('Choose the App Mode',['About App','Run on Image','Use Wepcam : Detect Squart','Use Wepcam : Detect Arm'])

if app_mode =='About App':
    st.markdown('''
          ## 소개 \n 
            미디어 파이프를 활용하여 운동 횟수를 세는 프로젝트를 맡은 김보라 입니다. \n
           
            이 프로젝트에서는 웹캠을 통한 영상이나 이미지의 운동을 판별합니다.\n
            ''')


    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.markdown('#### 1. 이미지를 통해 스쿼트, 와이드 스쿼트, 런지를 판별하고 그 외는 False 처리합니다.')
    st.text('Input Imaget')
    st.image('img.jpg')     
    st.text('Output Image')
    st.image('result.jpg')

    st.markdown('''
            
           \n
            #### 2. 웹캠으로 전달받은 영상의 '왼쪽, 오른쪽팔 구부리기'와 '스쿼트' 를 감지하여 횟수를 세어줍니다. \n
            
            - '왼쪽, 오른쪽팔 구부리기'는 각 팔의 어깨 - 팔꿈치 - 손목의 각도 변화를 감지하여 횟수를 셉니다. \n

            - '스쿼트'의 경우도 비슷한 원리로 '어깨 - 골반 - 무릎'의 각도와 '골반 - 무릎 -  발목'의 각도를 감지하여 앉았다 일어서는 변화를 감지하며 \n

            - '팔꿈치 - 어깨 - 골반', '어깨 - 골반 - 무릎', '골반 - 무릎 -  발목' 세 가지 각도를 통해 스쿼트 자세가 올바른지 확인합니다.\n
            \n

            ''')
    #st.video('squart.mp4')

elif app_mode =='Use Wepcam : Detect Squart':
    st.markdown('#### 웹캠을 활용한 스쿼트 동작 인식')
    run = st.button('Use Wepcam')
    st.write('웹캠을 사용하려면 \'Use Webcam\' 버튼을 누르세요.')
    FRAME_WINDOW = st.image([])
    cam = cv2.VideoCapture(0)\
    # coumt 변수
    counter = 0 
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence = 0.5 ) as pose :
        while run:
            
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            image.flags.writeable = True 
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

            try : 
                landmarks = results.pose_landmarks.landmark
                
                # 정규화된 3차원 좌표번호 얻기(함수로 처리해도 됨)
                #def get_cordinate(landmarks, part_name):
                    #return [landmarks[mppose.PoseLandmark[part_name].value].x,
                    #landmarks[mppose.PoseLandmark[part_name].value].y,
                    #landmarks[mppose.PoseLandmark[part_name].value].z,]
                
                wrist_l = [ landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y ]
                wrist_r = [ landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y ]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_l= [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # 평균좌표
                elbow = [ (elbow_r[i] + elbow_l[i])/2 for i in range(2) ]
                shoulder = [ (shoulder_r[i] + shoulder_l[i])/2 for i in range(2) ]
                hip = [ (hip_r[i] + hip_l[i])/2 for i in range(2) ]
                knee = [ (knee_r[i] + knee_l[i])/2 for i in range(2) ]
                ankle = [ (ankle_r[i] + ankle_l[i])/2 for i in range(2) ]
                
                
                # 좌표 기반 각도 계산
                # Elbow : wrist - elbow- shoulder
                elbow_angle_l = round(calculate_angle(wrist_l,elbow_l,shoulder_l))
                elbow_angle_r = round(calculate_angle(wrist_r,elbow_r,shoulder_r))
                # arm : elow - shoulder -hip
                arm_angle = round(calculate_angle(elbow, shoulder, hip),2)
                # hip :  shoulder -hip -knee
                hip_angle = round(calculate_angle(shoulder, hip, knee),2)
                # knee :  hip - knee -ankle
                knee_angle = round(calculate_angle(hip, knee, ankle),2)

        

                # 각도 시각화
                cv2.putText(image,str(arm_angle), tuple(np.multiply(shoulder,[640,480]).astype(int)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA )
                cv2.putText(image,str(hip_angle), tuple(np.multiply(hip,[640,480]).astype(int)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA )
                cv2.putText(image,str(knee_angle), tuple(np.multiply(knee,[640,480]).astype(int)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA )
                
                
                # 스쿼트 판별 및 개수 세기
                result = detect_squat(arm_angle,hip_angle,knee_angle,knee_r, knee_l)

                #print(result)
                if knee_angle < 130 and hip_angle < 130 : 
                    stage = "down"
                elif hip_angle > 150 and knee_angle > 150 and stage =='down':
                    stage="up"
                    counter +=1
                    #print(counter)
                    
            except :
                pass
            
            #상태 박스 만들기
            cv2.rectangle(image, (5,5), (270,70), (178,200,223), -1)
            
            # detect squart
            cv2.putText(image, 'Squart', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(result), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Rep data
            cv2.putText(image, 'REPS', (110,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(f'{counter:^5d}'), (110,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Stage data
            cv2.putText(image, 'STAGE', (190,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image,str(stage),(190,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

            FRAME_WINDOW.image(image)
        

elif app_mode == 'Use Wepcam : Detect Arm' :
    st.markdown('#### 웹캠을 활용한 **팔** 구부리기 동작 인식')
    run = st.button('Use Wepcam')
    st.write('웹캠을 사용하려면 \'Use Webcam\' 버튼을 누르세요.')
    FRAME_WINDOW = st.image([])
    cam = cv2.VideoCapture(0)\

    # coumt 변수
    counter_l = 0
    counter_r = 0
    stage_l = None
    stage_r = None


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence = 0.5 ) as pose :
        while run:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            image.flags.writeable = True 
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

            try : 
                landmarks = results.pose_landmarks.landmark
                shoulder_l = [ landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y ]
                elbow_l = [ landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y ]
                wrist_l = [ landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y ]
                shoulder_r = [ landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y ]
                elbow_r = [ landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y ]
                wrist_r = [ landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y ]
                                
                # 좌표 기반 각도 계산
                elbow_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            
                if elbow_angle_l > 160:
                    stage_l = "down"
                elif elbow_angle_l < 30 and stage_l =='down':
                    stage_l="up"
                    counter_l +=1
                
                if elbow_angle_r > 160:
                    stage_r = "down"
                elif elbow_angle_l < 30 and stage_r =='down':
                    stage_r="up"
                    counter_r +=1
     
            except :
                pass
                
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            cv2.rectangle(image, (0,0), (190,60), (178,200,223), -1)
            cv2.putText(image, 'REPS_l', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter_r), (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE_l', (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage_r,(100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)                            
                

            cv2.rectangle(image, (450,0), (710,60), (178,200,223), -1)
            cv2.putText(image, 'REPS_r', (460,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter_l),(460,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE_r', (545,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage_l, (545,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            FRAME_WINDOW.image(image)
                                              


elif app_mode =='Run on Image':
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        st.markdown('#### Input Image')
        st.image(image)
        st.markdown('#### Output Image')
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence = 0.5 ) as pose :
            #  RGB로 색 변환 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # BGR로 다시 색 변환
            image.flags.writeable = True 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Render detections
            mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                                )     
            # Landmarks 추출하기, Landmark 탐지 실패하면 pass
            try :
                landmarks = results.pose_landmarks.landmark
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_l= [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # 평균좌표 (x,y -> range(2))
                elbow = [ (elbow_r[i] + elbow_l[i])/2 for i in range(2) ]
                shoulder = [ (shoulder_r[i] + shoulder_l[i])/2 for i in range(2) ]
                hip = [ (hip_r[i] + hip_l[i])/2 for i in range(2) ]
                knee = [ (knee_r[i] + knee_l[i])/2 for i in range(2) ]
                ankle = [ (ankle_r[i] + ankle_l[i])/2 for i in range(2) ]

                # 좌표 기반 각도 계산

                # arm : elow - shoulder -hip
                arm_angle = round(calculate_angle(elbow, shoulder, hip),2)
                # hip :  shoulder -hip -knee
                hip_angle = round(calculate_angle(shoulder, hip, knee),2)
                # knee :  hip - knee -ankle
                knee_angle = round(calculate_angle(hip, knee, ankle),2)
            
                # 스쿼트 판별
                result = detect_squat(arm_angle, hip_angle, knee_angle,knee_r,knee_l)
                #상태 박스 만들기
                cv2.rectangle(image, (5,5), (150,70), (178,200,223), -1)

                # Rep data
                cv2.putText(image, str(result), (10,25),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA) 
                cv2.putText(image, str("arm_angle"), (10,45),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA) 
                cv2.putText(image, str(arm_angle), (100,45),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA) 
                cv2.putText(image, str("hip_angle"), (10,55),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(hip_angle), (100,55),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str("knee_angle"), (10,65),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(knee_angle), (100,65),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

                st.image(image) 
                st.markdown(f'#### 결과 : {result}')

            except :
                pass
                            
    else:
        demo_image = DEMO_IMAGE
        demo_image_result = DEMO_IMAGE_RESULT
        image = np.array(Image.open(demo_image))
        image_result = np.array(Image.open(demo_image_result))
        st.markdown('#### Sample')
        st.text('Input Image')
        st.image(image)
        st.text('Output Image')
        st.image(image_result)
        st.markdown('#### 결과 : Squart')
