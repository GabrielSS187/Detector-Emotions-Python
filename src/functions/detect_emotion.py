from deepface import DeepFace

from src.objects.emotions_object import emotion_translation

def detect_emotion(face_roi):
    # Analisar a emoção na região do rosto
    result = DeepFace.analyze(face_roi, actions=["emotion"])
    emotions = result[0]["emotion"]
    dominant_emotion = max(emotions, key=emotions.get)
    return emotion_translation[dominant_emotion]