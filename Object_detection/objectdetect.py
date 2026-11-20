import cv2 as cv
import numpy as np
from pathlib import Path



def main():
    cap = None
    try:
        project_dir = Path(__file__).resolve().parent

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Unable to open webcam.')
        whT = 608
        confThreshold =0.5
        nmsThreshold= 0.2

        #### LOAD MODEL
        ## Coco Names
        classesFile = project_dir / "coco.names"
        classNames = []
        with open(classesFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')
        print(classNames)

        ## Model Files
        modelConfiguration = project_dir / "yolov3-320.cfg"
        modelWeights = project_dir / "yolov3.weights"
        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]

        #new function to find object
        def findObjects(outputs,img):
            hT, wT, cT = img.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        w,h = int(det[2]*wT) , int(det[3]*hT)
                        x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                        bbox.append([x,y,w,h])
                        classIds.append(classId)
                        confs.append(float(confidence))

            indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

            if len(indices) > 0:
                for i in np.array(indices).flatten():
                    box = bbox[i]
                    x, y, w, h = box
                    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                              (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        try:
            while True:
                success, img = cap.read()

                if not success:
                    print("Unable to read webcam frame.")
                    break

                blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                outputs = net.forward(outputNames)
                findObjects(outputs, img)

                cv.imshow('Image', img)

                if cv.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            print("Interrupted by user.")


    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()
