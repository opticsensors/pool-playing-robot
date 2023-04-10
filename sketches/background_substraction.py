import cv2

bg_subtractor=cv2.createBackgroundSubtractorMOG2(detectShadows=False,varThreshold = 20)
bg = cv2.imread('./data/background.jpg')
#bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg = cv2.GaussianBlur(bg, (25,25), 0)
fg_mask = bg_subtractor.apply(bg,learningRate=0)

random_config=cv2.imread('./data/config_1.jpg')
#random_config = cv2.cvtColor(random_config, cv2.COLOR_BGR2GRAY)
random_config = cv2.GaussianBlur(random_config, (25,25), 0)
fg_mask = bg_subtractor.apply(random_config,learningRate=0)
cv2.imwrite('./results/fg_mask.png', fg_mask)
