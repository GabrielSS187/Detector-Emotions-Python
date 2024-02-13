import cv2

from src.functions.detect_emotion import detect_emotion


# Carregar o classificador de cascata de rosto pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def main():
    cap = cv2.VideoCapture(0)

    while True:
        try:
            # Capturar frame da câmera
            ret, frame = cap.read()
            if not ret:
                break
            
            # Espelhar o frame horizontalmente
            frame = cv2.flip(frame, 1)

            # Converter para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar rostos na imagem em escala de cinza
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Para cada rosto detectado
            for (x, y, w, h) in faces:
                # Extrair região de interesse (ROI) do rosto
                face_roi = frame[y:y+h, x:x+w]

                # Detectar emoção na região do rosto e obter a emoção dominante
                emotion = detect_emotion(face_roi)

                # Desenhar retângulo ao redor do rosto
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Exibir a emoção detectada sobre o rosto
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

            # Exibir o frame com as emoções detectadas
            cv2.imshow("Análise de emoções", frame)

            # Se a tecla 'q' for pressionada, sair do loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            # Se ocorrer um erro, continue com a próxima iteração

    # Liberar os recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
