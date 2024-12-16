from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from collections import deque

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    frame_skip = 20  # 每 20 帧检测一次
    frame_count = 0
    emotions, gender_display, age = "N/A", "N/A", "N/A"  # 初始化值

    # 设置情绪变化趋势图
    emotion_trend = deque([7] * 30, maxlen=30)  # 这里初始值设置为7，表示"neutral"情绪
    emotion_mapping = {
        "angry": 1, "disgust": 2, "fear": 3, "happy": 4,
        "sad": 5, "surprise": 6, "neutral": 7  #1:生气；2：反感厌恶；3：害怕畏惧；4：开心；5：悲伤难过；6：惊讶震惊；7：中立，不带感情色彩
    }
    plt.ion()
    fig, ax = plt.subplots()
    emotion_line, = ax.plot(emotion_trend)
    ax.set_ylim(0, 8)
    ax.set_title("Emotion Trend")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Emotion")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 缩小帧大小
        frame = cv2.resize(frame, (640, 480))

        # 人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if frame_count % frame_skip == 0 and len(faces) > 0:
            x, y, w, h = faces[0]  # 使用检测到的第一个人脸
            face_img = frame[y:y+h, x:x+w]

            try:
                # 使用 DeepFace 进行分析
                analysis = DeepFace.analyze(face_img, actions=['emotion', 'gender', 'age'], enforce_detection=False)
                
                # 检查分析结果是否非空
                if analysis:
                    emotions = analysis[0].get('dominant_emotion', "N/A")
                    # 将性别的概率值转换为百分比格式
                    gender = analysis[0].get('gender', {})
                    if isinstance(gender, dict):
                        gender_display = f"Man: {round(gender.get('Man', 0), 1)}%, Woman: {round(gender.get('Woman', 0), 1)}%"
                    else:
                        gender_display = "N/A"
                    # 将年龄减去10岁
                    age = max(0, analysis[0].get('age', 0) - 10)
                    
                    # 更新情绪趋势
                    emotion_value = emotion_mapping.get(emotions, 7)  # 默认值为 neutral 的值 7
                    emotion_trend.append(emotion_value)
                    emotion_line.set_ydata(emotion_trend)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            except Exception as e:
                print("分析出错:", e)
                emotions, gender_display, age = "N/A", "N/A", "N/A"

        # 在图像上绘制红框和检测结果
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 在图像上绘制情绪信息
            cv2.putText(frame, f"Emotion: {emotions}", (x, y - 10), font, 0.7, (255, 0, 0), 2)
            
            # 在图像上分两行显示性别信息
            cv2.putText(frame, "Gender:", (x, y + h + 20), font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, gender_display, (x, y + h + 50), font, 0.7, (0, 255, 0), 2)

            # 显示调整后的年龄信息
            cv2.putText(frame, f"Age: {age}", (x, y + h + 80), font, 0.7, (0, 0, 255), 2)

        # 显示视频帧
        cv2.imshow("Face Analysis", frame)
        frame_count += 1

        # 检查窗口是否关闭
        if cv2.getWindowProperty("Face Analysis", cv2.WND_PROP_VISIBLE) < 1:
            break

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_webcam()
