import streamlit as st
import cv2
from deepface import DeepFace
from openai import OpenAI
import tempfile
import os

# Load API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)

st.title("🎭 Virtual Health Assistant")
st.markdown("Detect your emotion in real-time and receive supportive responses.")

capture_btn = st.button("Start Camera & Detect Emotion")

if capture_btn:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')

    cap = cv2.VideoCapture(0)
    st.info("Press 'Q' in the video window to capture the frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break
        cv2.imshow("Press Q to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(temp_file.name, frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    st.image(temp_file.name, caption="Captured Frame")

    try:
        result = DeepFace.analyze(img_path=temp_file.name, actions=['emotion'])
        emotion = result[0]['dominant_emotion']
        st.success(f"Detected Emotion: **{emotion.upper()}**")

        prompt = f"I am feeling {emotion}. Respond empathetically and try to improve my mood."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )

        st.markdown("### 💬 Virtual Assistant Says:")
        st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"Error during detection: {e}")
