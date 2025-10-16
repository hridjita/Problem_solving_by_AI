import cv2
import numpy as np
import cvzone
import google.generativeai as genai
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1,
                        detectionCon=0.5, minTrackCon=0.5)
def gethands(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand1 = hands[0]
        lmlist = hand1["lmList"]
        finger1 = detector.fingersUp(hand1)
        return finger1, lmlist, img
    else:
        return None, None, img
def draw(info, prev_pos, canvas, img):
    fingers, lmlist = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Only index finger up (drawing mode)
        current_pos = tuple(lmlist[8][0:2])
        if prev_pos is not None:
            cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)
        prev_pos = current_pos

    elif fingers == [1, 1, 1, 1, 1]:  # All fingers up (clear canvas)
        canvas = np.zeros_like(img)
        prev_pos = None

    else:
        # 👇 Reset drawing when not in drawing mode
        prev_pos = None

    return prev_pos, canvas

from PIL import Image
genai.configure(api_key="YOUR API KEY")

# Load Gemini Vision Model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
def sendtoAI(model, canvas, fingers):
    if fingers == [0, 1, 1, 0, 1]:
        # Convert canvas (NumPy) to PIL image
        pil_image = Image.fromarray(canvas)

        # Send to Gemini Vision with prompt
        response = model.generate_content(["Solve this math problem:", pil_image])
        answer = response.text

        # Create a white blank image to display the answer
        height, width = 600, 800
        white_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

        # Write the AI answer text line by line
        y0, dy = 50, 30  # Starting y position, line spacing
        for i, line in enumerate(answer.split('\n')):
            y = y0 + i * dy
            # Put text on the white image (black color)
            cv2.putText(white_img, line.strip(), (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Show the window with the answer
        cv2.imshow("AI Answer", white_img)
        cv2.waitKey(0)  # Wait until key press
        cv2.destroyWindow("AI Answer")
prev_pos = None
canvas = None
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    fingers, lmlist, img = gethands(img)

    if fingers is not None and lmlist is not None:
        prev_pos, canvas = draw((fingers, lmlist), prev_pos, canvas, img)

        # ✅ Check gesture and send to AI
        sendtoAI(model, canvas, fingers)

    combined = cv2.addWeighted(img, 0.8, canvas, 0.5, 0)
    cv2.imshow("Virtual Drawing", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
