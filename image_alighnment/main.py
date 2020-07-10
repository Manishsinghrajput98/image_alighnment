import argparse
from flask import Flask, request, jsonify
from threading import Thread
import os
import cv2 
import numpy as np 

class Manish(Flask):
    def __init__(self, host, name):
        super(Manish, self).__init__(name,static_url_path='')
        self.host = host
        self.define_uri()

    def define_uri(self):
        self.provide_automatic_option = False
        self.add_url_rule('/start', None, self.start, methods = [ 'POST' ] )

    def image_alighnment(self,image):
        print("Call api For Image Alighnment")
        im2 = cv2.imread("Feature/matching.jpg", cv2.IMREAD_COLOR)
        im1 = cv2.imread(image, cv2.IMREAD_COLOR)
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        MAX_MATCHES = 500
        GOOD_MATCH_PERCENT = 0.15
        orb = cv2.ORB_create(MAX_MATCHES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))
        cv2.imwrite("output.jpg", im1Reg)
        
    def start(self):

        print("Call api For Image Alighnment")
        print("args :",request.args)

        if request.method == "POST":
            body = request.get_json()
            print(body)
            image_path = body['image_path']
            self.image_alighnment(image_path)
            res = dict()
            res['status'] = '200'
            res['result'] = "image Alighnment Sucessfull"
            return jsonify(res)
        else:
            res = dict()
            res['status'] = '500'
            res['msg'] = 'Some problem in video'
    
def importargs():
    parser = argparse.ArgumentParser('This is the server of image alighnment')
    parser.add_argument("--host", "-H", help = "host name running server",type=str, required=False, default='localhost')
    parser.add_argument("--port", "-P", help = "port of runnning server", type=int, required=False, default=3001)
    args = parser.parse_args()
    return args.host, args.port

def main():
    host, port = importargs()
    server = Manish(host, 'new_server')
    print("server run")
    server.run(host=host, port=port)


if __name__ == "__main__":
    main()
