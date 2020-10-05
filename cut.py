import cv2
import numpy as np
import pytesseract as tess
from flask import Flask, request, Response, jsonify
from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey
from waitress import serve
import json
import os

# Credenciais do cloudant
serviceUsername =  os.getenv("DB_USERNAME") 
servicePassword =  os.getenv("DB_PASSWORD") 
serviceURL =  os.getenv("DB_URL") 


# Conecta ao serviço do CLoudant
client = Cloudant.iam(serviceUsername, servicePassword, url=serviceURL)
client.connect()

# Abre a database 
db = client['dev']

app = Flask("API_IMAGE_RECOGNITION_TCC")

@app.route("/solution", methods=["POST"])
def solution_post():
    
    deviceID = request.form.get('deviceID')
    childInfo = request.form.get('childInfo')
    try:
        solution = request.files['image']
    except:
        return Response(json.dumps({"error":"404","msg": "Missing Image"}), status=404,mimetype='application/json')

    if( not bool(deviceID) or not bool(childInfo)):
        return Response(json.dumps({"error":"404","msg": "Missing Info"}), status=404,mimetype='application/json')

    childInfo = json.loads(childInfo)

    nparr =  np.fromstring(solution.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
    
    #purple color
    low_purple = np.array([80, 80, 75])
    high_purple = np.array([190, 220, 150])
    purple_mask = cv2.inRange(hsv_frame, low_purple, high_purple)
    purple = cv2.bitwise_and(frame, frame, mask=purple_mask)

    #orange color
    low_orange = np.array([5, 120, 150])
    high_orange = np.array([15, 255, 230])
    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
    orange = cv2.bitwise_and(frame, frame, mask=orange_mask)

    #yellow color
    low_yellow = np.array([20,30,135])
    high_yellow = np.array([40,255,255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    #Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([115, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    #Green color
    low_green = np.array([40, 70, 40])
    high_green = np.array([100, 255, 200])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    #Every color except white
    low = np.array([0, 0, 0])
    high = np.array([0, 0, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    blocos = []

    # Creating contour to track blue color 
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in (contours): 
        area = cv2.contourArea(contour) 
        if(area > 25000): 

            x, y, w, h = cv2.boundingRect(contour)  
            crop_img_blue = blue[x:w, y:h]
            blocos.append(tuple((x, 'Turn')))
            continue

    # Creating contour to track yellow color 
    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in (contours): 
        area = cv2.contourArea(contour) 
        if(area > 8000):     

            x, y, w, h = cv2.boundingRect(contour)  
            crop_img = yellow[x:w, y:h]
            blocos.append(tuple((x, 'Walk')))
            continue

    # Creating contour to track orange color 
    contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in (contours): 
        area = cv2.contourArea(contour) 
        if(area > 25000): 

            x, y, w, h = cv2.boundingRect(contour)  
            crop_img = orange[x:w, y:h]
            blocos.append(tuple((x, 'Wait')))
            continue

    # Creating contour to track purple color 
    contours, hierarchy = cv2.findContours(purple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in (contours): 
        area = cv2.contourArea(contour) 
        if(area > 15000): 
        
            x, y, w, h = cv2.boundingRect(contour)  
            crop_img = purple[x:w, y:h]
            blocos.append(tuple((x, 'Loop')))
            continue

    # Creating contour to track green color 
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in (contours): 
        area = cv2.contourArea(contour) 
        if(area > 20000):

            x, y, w, h = cv2.boundingRect(contour)  
            #crop_img = green[y:y+h, int(x-((w-x)/6)):int(x+w+((w-x)/6))]
            crop_img = green[y:y+h, int(x):int(x+w)]
            #cv2.drawContours(green, contours, 0, (0, 230, 255), 6)
            #if (crop_img.size > 0):
                #cv2.imshow("teste verde", crop_img)
                #key = cv2.waitKey(0)

            text = tess.image_to_string(crop_img, lang='eng', config='--tessdata-dir "/home/vcap/deps/0/apt/usr/share/tesseract-ocr/4.00/tessdata" --psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
            #print(text[0])
            
            try: 
                int(text[0])
                blocos.append(tuple((x, 'Number' + str(text[0]))))
                
            except ValueError:
                print("Error converting text to number")
            
            continue

    blocos_ordenados = (sorted(blocos))
    blocks = [{"name":  x[1]} for x in blocos_ordenados]

    data = {
        'child': childInfo,
        'device': deviceID,
        'solution': blocks,
        'correctSolution' : False
    }

    # Cria o documento no banco de dados
    document = db.create_document(data)

    # Verifica se o documento foi incluido no banco
    if not document.exists():
        return Response(json.dumps({"error":"500","msg": "Erro ao salvar no banco de dados"}), status=500,mimetype='application/json')

    document_formatted = {
        'solutionID' : document['_id'],
        'blocks': document['solution']
    }

    return Response(json.dumps(document_formatted), status=200,mimetype='application/json')

@app.route("/solution", methods=["PUT"])
def solution_put():
    solutionID = request.form.get('solutionID')
    correctSolution = request.form.get('correctSolution')
    childInfo = request.form.get('childInfo')

    if(not bool(solutionID) or not bool(correctSolution) or not bool(childInfo)):
        return Response(json.dumps({"error":"404","msg": "Missing Info"}), status=404,mimetype='application/json')

    childInfo = json.loads(childInfo)
    document = db[solutionID]
    document['correctSolution'] = (correctSolution == 'true' or correctSolution == 'True')
    document['child'] = childInfo
    document.save()
    return Response(json.dumps({"msg": "Solution updated with success"}), status=200,mimetype='application/json')

port = int(os.getenv("PORT", 8080))
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=port)