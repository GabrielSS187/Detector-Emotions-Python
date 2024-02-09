import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")

emotion_translation = {
  "angry": "zangado",
  "disgust": "nojo",
  "fear": "medo",
  "happy": "feliz",
  "sad": "triste",
  "surprise": "surpreso",
  "neutral": "normal"
}

cap = cv2.VideoCapture(0)

while True:
  try:
    net, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for ( x, y, w, h ) in faces:
      face_roi = frame[y:y+h,  x:x+w]
      
      result = DeepFace.analyze(face_roi, actions=["emotion"])
      emotions =result[0]["emotion"]
      dominant_emotion = max(emotions, key=emotions.get)
      
      cv2.rectangle(frame, ( x, y ), ( x+w, y+h ), ( 255, 0, 0 ), 2 )
      
      text = f"{emotion_translation[dominant_emotion]}"
      cv2.putText(frame, text, ( x, y-10 ), cv2.FONT_HERSHEY_SIMPLEX, 2.0, ( 255, 0, 0 ), 2, cv2.LINE_AA)
    
    cv2.imshow("Analise de emoções", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
    
  except: 
    pass
  
  
cap.release()
cv2.destroyAllWindows()