import cv2
import streamlit as st
from deepface import DeepFace
from collections import Counter
import matplotlib.pyplot as plt

def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Set up Streamlit layout
    st.title("Real-time Emotion Detection")
    video_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Initialize counters for emotion detection
    emotions_counter = Counter()

    while True:
        # Read the frame
        _, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        try:

            # Draw the rectangle around each face and analyze emotions
            for (x, y, w, h) in faces:
                try:
                    # Analyze emotions using DeepFace
                    emotion = DeepFace.analyze(img, actions=["emotion"])
                    emotion_label = emotion[0]['dominant_emotion']
                    emotions_counter[emotion_label] += 1
                    cv2.putText(img, str(emotion_label), (x, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                except:
                    print("No face detected")

            # Display the frame in Streamlit
            video_placeholder.image(img, channels="BGR")

            # Update and display the emotion density graph
            update_emotion_density_chart(chart_placeholder, emotions_counter)
        except:
            print("its alright fam")

        # Check for key press event
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()

def update_emotion_density_chart(chart_placeholder, emotions_counter):
    # Prepare data for the chart
    labels, values = zip(*emotions_counter.items())

    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar(labels, values)

    # Customize the chart
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Density')
    ax.set_title('Emotion Density')
    ax.grid(True)

    # Update the chart in Streamlit
    chart_placeholder.pyplot(fig)

if __name__ == "__main__":
    main()
