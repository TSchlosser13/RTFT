#!/usr/bin/env python3.7


'''
@file   RTFT.py
@author Tobias Schlosser (tobias@tobias-schlosser.net)
@date   May 23, 2018
@brief  RTFT - Real Time Face Tracking Using face_recognition

TODO
'''


import sys

import face_recognition
import cv2

from datetime import datetime
from psutil   import virtual_memory


tracking_secs = 16


if len(sys.argv) != 2:
	print('Usage: %s <input>' % sys.argv[0])

	video_capture = cv2.VideoCapture(0)
else:
	video_capture = cv2.VideoCapture(sys.argv[1])


# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

tobias_img      = face_recognition.load_image_file('Testset/Tobias.png')
tobias_img_encs = face_recognition.face_encodings(tobias_img)[0]

face_encs_all = [tobias_img_encs]
face_last_all = [[]]
face_disp_all = [face_last_all[0]]
face_seen     = 8192 * [False] # TODO: size
time_last_all = [datetime.now().strftime('%H:%M:%S')]
time_disp_all = [time_last_all[0]]


while True:
	tmp, frame  = video_capture.read()
	frame_small = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)

	face_locs = face_recognition.face_locations(frame_small)
	# face_locs = face_recognition.face_locations(frame_small, 0, 'cnn')
	face_encs = face_recognition.face_encodings(frame_small, face_locs)

	face_disp  = []
	face_names = []
	time_disp  = []


	for i, tmp in enumerate(time_last_all):
		if datetime.now().second - int(time_last_all[i].split(':')[2]) > tracking_secs:
			face_disp_all[i] = face_last_all[i]
			time_disp_all[i] = time_last_all[i]


	for i, tmp in enumerate(face_encs):
		for j, tmp in enumerate(face_encs_all):
			match = face_recognition.compare_faces([face_encs_all[j]], face_encs[i])

			if match[0]:
				try:
					face_last_all[j] = frame_small[
						face_locs[i][0] : face_locs[i][2],
						face_locs[i][3] : face_locs[i][1]].copy()

					if not face_seen[j]:
						face_disp_all[j] = face_last_all[j]
						face_seen[j]     = True
				except:
					pass # TODO

				time_last_all[j] = datetime.now().strftime('%H:%M:%S')

				face_disp.append(face_disp_all[j])
				time_disp.append(time_disp_all[j])

				if j > 0:
					face_names.append('Person_' + str(j + 1))
				else:
					face_names.append('Tobias')

				break
			elif j == len(face_encs_all) - 1:
				face_encs_all.append(face_encs[i])
				time_last_all.append(datetime.now().strftime('%H:%M:%S'))
				time_disp_all.append(time_last_all[j + 1])

				try:
					face_last_all.append(frame_small[
						face_locs[i][0] : face_locs[i][2],
						face_locs[i][3] : face_locs[i][1]].copy())

					face_disp_all.append(face_last_all[j + 1])
					face_seen[j + 1] = True
				except:
					# TODO
					face_last_all.append([])
					face_disp_all.append(face_last_all[j + 1])

				face_disp.append(face_last_all[j + 1])
				face_names.append('Person_' + str(j + 1))
				time_disp.append(time_last_all[j])


	for (top, right, bottom, left), face, name, time in zip(face_locs, face_disp, face_names, time_disp):
		top    *= 2
		right  *= 2
		bottom *= 2
		left   *= 2

		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1, (255, 255, 255), 1)
		cv2.putText(frame, 'Last time seen: ' + time, (left + 6, bottom + 26), font, 1, (255, 255, 255), 1)

		try:
			frame[top + 16 : top + 16 + face.shape[0], right + 16 : right + 16 + face.shape[1]] = face
			cv2.rectangle(frame, (right + 16, top + 16), (right + 16 + face.shape[1], top + 16 + face.shape[0]), \
				(255, 255, 255), 2)
		except:
			pass # TODO


	cv2.namedWindow('RTFT - Real Time Face Tracking Using face_recognition', cv2.WND_PROP_FULLSCREEN)
	# cv2.setWindowProperty('RTFT - Real Time Face Tracking Using face_recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.imshow('RTFT - Real Time Face Tracking Using face_recognition', frame)

	if cv2.waitKey(1) & 0xFF == 27:
		break


	if virtual_memory().percent > 90:
		face_encs_all     = [face_encs_all[0]]
		face_last_all     = [face_last_all[0]]
		face_disp_all     = [face_disp_all[0]]
		face_seen[1:8191] = [False] # TODO: size
		time_last_all     = [time_last_all[0]]
		time_disp_all     = [time_disp_all[0]]


video_capture.release()
cv2.destroyAllWindows()

